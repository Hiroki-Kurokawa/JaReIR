#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Annotation server using Flask.
"""

from __future__ import annotations

from flask import (
    Flask,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
    jsonify,
)


import json
import hashlib
import tempfile
import os
import pandas as pd
from html import escape
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import re
from src.utils.configs import load_configs
CFG = load_configs("configs")

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
secret = os.environ.get("FLASK_SECRET_KEY")
if not secret:
    secret = "dev-secret" # In production, set FLASK_SECRET_KEY environment variable
app.secret_key = secret

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
INDEX_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, 'index.html')
BASE_URL = os.environ.get("ANNOTATION_BASE_URL", "http://localhost:5001")

os.makedirs(OUTPUT_DIR, exist_ok=True)

_TASK_CACHE: Dict[str, List[Dict[str, str]]] = {}


AUDIT_LOG_PATH = os.path.join(OUTPUT_DIR, "audit_log.jsonl")

def atomic_write_json(path: str, obj: dict):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic-ish
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def audit_log(event: str, **fields):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def load_tasks(csv_path: str) -> List[Dict[str, str]]:
    """Load tasks from a CSV file.
    Supports two CSV formats:
      1) Flat: one task per row with columns like topic/psg/did (synonyms accepted)
      2) Numbered: multiple tasks per row with columns like topic1, psg1, did1, qid1, topic2, ...
    """
    csv_path = os.path.abspath(csv_path)
    if csv_path in _TASK_CACHE:
        return _TASK_CACHE[csv_path]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    tasks: List[Dict[str, str]] = []
    df = pd.read_csv(csv_path, dtype=str)
    for _, row in df.iterrows():
        task_row = row.to_dict()
        task_row["psg"] = re.sub(r'_dup\d+', '', task_row["psg"])
        tasks.append(task_row)

    _TASK_CACHE[csv_path] = tasks
    return tasks


@app.route('/')
def root():
    return render_template('none.html')

@app.route('/complete', methods=['GET', 'POST'])
def complete():
    worker_id = session.get('worker_id')
    if not worker_id:
        return redirect(url_for('worker_default', _external=True)) 

    ended_at_iso = datetime.now(timezone.utc).isoformat()
    started_at_iso = session.get('started_at_iso')
    return render_template('complete.html', worker_id=worker_id, started_at_iso=started_at_iso, ended_at_iso=ended_at_iso)

@app.route('/worker', methods=['GET', 'POST'])
def worker_default():
    """Redirect to default task set worker page."""
    return render_template('none.html')

@app.route('/worker/<task_set_id>', methods=['GET', 'POST'])
def worker(task_set_id):
    
    # Construct CSV path from <task_set_id> in URL
    csv_name = f"{task_set_id}.csv"
    csv_path = os.path.join(CFG["data_config"]["paths"]["first_annotation_csv_paths"], csv_name)

    defaults = {
        'tasks_per_page': 3,
        'post_submit_action': 'next',
    }

    if request.method == 'POST':
        worker_id = (request.form.get('worker_id') or '').strip()
        tasks_per_page = int(request.form.get('tasks_per_page') or '3')
        post_submit_action = (request.form.get('post_submit_action') or 'next').strip()
        
        if tasks_per_page <= 0:
            tasks_per_page = 3
        tasks_per_page = min(tasks_per_page, 3)

        if not worker_id:
            return render_template('worker.html', 
                                 error='ユーザー名 を入力してください', 
                                 defaults=defaults, 
                                 task_set_name=csv_name)

        if not os.path.exists(csv_path):
            return render_template('worker.html', 
                                 error=f'タスクファイル {csv_name} が見つかりません。', 
                                 defaults=defaults, 
                                 task_set_name=csv_name)
        
        try:
            tasks = load_tasks(csv_path)
        except Exception as e:
            return render_template('worker.html', 
                                 error=f'CSV 読み込みに失敗しました: {e}', 
                                 defaults=defaults, 
                                 task_set_name=csv_name)
        
        if not tasks:
            return render_template('worker.html', 
                                 error='CSV にタスクがありません', 
                                 defaults=defaults, 
                                 task_set_name=csv_name)

        # Save to session
        session['worker_id'] = worker_id
        session['csv_path'] = csv_path  # 構築した CSV パスをセッションに保存
        session['post_submit_action'] = post_submit_action
        session['page_index'] = 0
        session['started_at_iso'] = datetime.now(timezone.utc).isoformat()
        
        session['task_set_id'] = task_set_id
        session['consented'] = False
        return redirect(url_for('consent', task_set_id=task_set_id, _external=True))


    return render_template('worker.html', 
                         error=None, 
                         defaults=defaults, 
                         task_set_name=csv_name)

@app.route('/consent/<task_set_id>', methods=['GET', 'POST'])
def consent(task_set_id):
    
    worker_id = session.get('worker_id')
    if not worker_id:
        return redirect(url_for('worker', task_set_id=task_set_id, _external=True))

    if request.method == 'POST':
        if request.form.get('agree') != 'yes':
            return render_template('agreement.html', error='同意が必要です。', task_set_id=task_set_id, worker_id=worker_id)
        session['consented'] = True
        session['consented_at_iso'] = datetime.now(timezone.utc).isoformat()
        audit_log(
            "consent_accepted",
            worker_id=worker_id,
            task_set_id=task_set_id,
            user_agent=request.headers.get("User-Agent"),
            ip=request.headers.get("X-Forwarded-For", request.remote_addr),
        )
        return redirect(url_for('task_page', page=1, _external=True))

    return render_template('agreement.html', error=None, task_set_id=task_set_id, worker_id=worker_id)



@app.route('/task')
def task_page():

    worker_id = session.get('worker_id')
    csv_path = session.get('csv_path')
    if not worker_id or not csv_path:
        return redirect(url_for('worker', task_set_id='0', _external=True))
    if not session.get('consented'):
        return redirect(url_for('consent', task_set_id=session.get('task_set_id', '0'), _external=True))

    tasks = load_tasks(csv_path)
    qids = tasks[0]["qid"]
    psgs = [t["psg"] for t in tasks]
    dids = [t["did"] for t in tasks]

    audit_log(
        "task_served",
        worker_id=worker_id,
        task_set_id=session.get("task_set_id"),
        qid=qids,
        dids=dids[:5],             
        did_count=len(dids),
        psg_hashes=[hash_text(p) for p in psgs],  
    )

    session['tasks_per_page'] = 1

    try:
        page = int(request.args.get('page', '1'))
    except ValueError:
        page = 1
 
    with open(INDEX_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    assert len(set([t['topic'] for t in tasks])) == 1, "All topics must be the same for index2.html"
    if '{$topic}' in html or '{$psg}' in html or '{$did}' in html or '{$qid}' in html:
        topics = tasks[0]["topic"]
        psgs = "|||".join([t["psg"] for t in tasks])
        dids = "|||".join([t["did"] for t in tasks])
        qids = tasks[0]["qid"]
        html = html.replace('{$topic}', escape(topics))
        html = html.replace('{$psg}', escape(psgs))
        html = html.replace('{$did}', dids)

    submit_url = f"{BASE_URL}/submit"
    print("Submit URL:", submit_url)
    submit_url_json = json.dumps(submit_url)

    submit_ui = f"""
    <div id="submit-area" style="margin: 8px; right: 16px; background: #fff; border: 1px solid #ddd; padding: 10px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
        <div>作業者: <strong>{worker_id}</strong> / ページ {page} </div>
        <div id="timer">経過時間: <span id="elapsed">0</span> 秒</div>
        <div id="incompleteNotice" style="margin-top:6px; color:#d00; font-size:12px; display:none;">未完了のタスクがあります。全ての項目を入力してください。</div>
        <div style="margin-top:8px; display:flex; gap:8px;">
            <button id="submitBtn" style="padding:6px 12px;" disabled>送信</button>
            <button id="stopTimerBtn" style="padding:6px 12px;">時間停止</button>
        </div>
    </div>

    <script>
    (function() {{
        const startedAt = Date.now();
        let timerId = null;
        let pauseEvents = [];
        let pauseIntervals = []; // 停止期間の配列
        let totalPausedMs = 0;
        let pauseStartedAt = null;

        function updateTimer() {{
            const now = Date.now();
            let effectiveElapsed = now - startedAt - totalPausedMs;
            if (pauseStartedAt !== null) {{
                effectiveElapsed = pauseStartedAt - startedAt - totalPausedMs;
            }}
            const s = Math.floor(effectiveElapsed / 1000);
            const el = document.getElementById('elapsed');
            if (el) el.textContent = String(s);
        }}

        timerId = setInterval(updateTimer, 1000);
        updateTimer();

        const stopBtn = document.getElementById('stopTimerBtn');
        if (stopBtn) {{
            stopBtn.addEventListener('click', function() {{
                if (pauseStartedAt === null) {{
                    // 停止開始
                    pauseStartedAt = Date.now();
                    clearInterval(timerId);
                    timerId = null;
                    stopBtn.textContent = "時間再開";
                    pauseEvents.push({{ action: "pause", at: new Date(pauseStartedAt).toISOString() }});
                }} else {{
                    // 再開
                    const resumedAt = Date.now();
                    totalPausedMs += resumedAt - pauseStartedAt;
                    pauseIntervals.push({{
                        start: new Date(pauseStartedAt).toISOString(),
                        end: new Date(resumedAt).toISOString(),
                        duration_sec: Math.floor((resumedAt - pauseStartedAt) / 1000)
                    }});
                    pauseStartedAt = null;
                    timerId = setInterval(updateTimer, 1000);
                    stopBtn.textContent = "時間停止";
                    pauseEvents.push({{ action: "resume", at: new Date(resumedAt).toISOString() }});
                }}
                updateTimer();
            }});
        }}

        window.__TASK_PAGE__ = {page};
        window.__TOTAL_PAGES__ = {1};

        function collectAnswers() {{
            const results = [];
            const docs = document.querySelectorAll('.doc-item');
            docs.forEach(doc => {{
                const idx = doc.getAttribute('data-doc-index');
                function val(name) {{
                    const nodes = document.getElementsByName(name);
                    for (const n of nodes) if (n.checked) return n.value;
                    return '';
                }}
                const relevance = val(`data[Result][results][doc${{idx}}_relevance]`);
                const canDescribe = val(`data[Result][results][doc${{idx}}_can_describe]`);
                const ta = document.querySelector(`textarea[name="data[Result][results][doc${{idx}}_reason_text]"]`);
                const reason = ta ? ta.value : '';
                results.push({{
                    slot: 1,
                    global_index: Number(idx),
                    qid: (document.getElementById('qid')?.textContent || '').trim(),
                    did: doc.getAttribute('data-did') || '',
                    relevance: relevance,
                    can_describe: canDescribe,
                    reason_text: reason,
                    doc_index: Number(idx)
                }});
            }});
            return results;
        }}

        function pageComplete() {{
            if (typeof window.__pageCompletionStatus === 'function') {{
                const st = window.__pageCompletionStatus();
                return st.complete;
            }}
            return true;
        }}

        function updateSubmitState() {{
            const btn = document.getElementById('submitBtn');
            const notice = document.getElementById('incompleteNotice');
            const complete = pageComplete();
            if (btn) btn.disabled = !complete;
            if (notice) notice.style.display = complete ? 'none' : 'block';
        }}

        window.addEventListener('page-completion-changed', updateSubmitState);
        setTimeout(updateSubmitState, 50);

        async function submit() {{
            const btn = document.getElementById('submitBtn');
            if (btn) btn.disabled = true;
            try {{
                if (!pageComplete()) {{
                    updateSubmitState();
                    return;
                }}
                const now = Date.now();
                let effectiveElapsed = now - startedAt - totalPausedMs;
                if (pauseStartedAt !== null) {{
                    effectiveElapsed = pauseStartedAt - startedAt - totalPausedMs;
                    // 終了時にまだ停止中なら interval として保存
                    pauseIntervals.push({{
                        start: new Date(pauseStartedAt).toISOString(),
                        end: new Date(now).toISOString(),
                        duration_sec: Math.floor((now - pauseStartedAt) / 1000)
                    }});
                }}

                const submissionId = (crypto?.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now());
                const payload = {{
                    submission_id: submissionId,
                    worker_id: {json.dumps(worker_id)},
                    page: window.__TASK_PAGE__,
                    started_at_client: new Date(startedAt).toISOString(),
                    finished_at_client: new Date().toISOString(),
                    effective_elapsed_sec: Math.floor(effectiveElapsed / 1000),
                    pause_events: pauseEvents,
                    pause_intervals: pauseIntervals,  // 停止区間も送信
                    results: collectAnswers(),
                }};
                console.log("Submitting payload:", payload);
                const res = await fetch({submit_url_json}, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload),
                }});
                if (res.ok) {{
                    const data = await res.json();
                    const next = data.next_url;
                    if (next) window.location.href = next;
                }} else {{
                    alert('保存に失敗しました');
                }}
            }} finally {{
                if (btn) btn.disabled = false;
            }}
        }}

        const btn = document.getElementById('submitBtn');
        if (btn) btn.addEventListener('click', function(e) {{ e.preventDefault(); submit(); }});
    }})();
    </script>
    """

    insert_at = html.rfind("</body>")
    #print(html)
    if (insert_at != -1):
        html = html[:insert_at] + submit_ui + html[insert_at:]
    else:
        html += submit_ui

    session['page'] = page
    session['page_started_at_iso'] = datetime.now(timezone.utc).isoformat()
    #print(html)
    resp = make_response(html)
    return resp


@app.post('/submit')
def submit_results():

    if not session.get("consented"):
        return jsonify({'error': 'not consented'}), 403
    worker_id = session.get('worker_id')
    if not worker_id:
        return jsonify({'error': 'no worker session'}), 400

    try:
        payload = request.get_json(force=True)
    except Exception:
        payload = None
    if not payload:
        return jsonify({'error': 'invalid payload'}), 400

    if payload.get('worker_id') and payload.get('worker_id') != worker_id:
        # Ensure the same worker
        return jsonify({'error': 'worker_id mismatch'}), 400

    page = int(payload.get('page') or session.get('page') or 1)
    started_at_client = payload.get('started_at_client')
    finished_at_client = payload.get('finished_at_client')
    started_at_server = session.get('page_started_at_iso')
    finished_at_server = datetime.now(timezone.utc).isoformat()

    pause_events = payload.get('pause_events', [])
    pause_intervals = payload.get('pause_intervals', [])
    effective_elapsed_sec = payload.get('effective_elapsed_sec', None)

    # compute elapsed seconds if possible
    def _elapsed_s(start_iso: Optional[str], end_iso: Optional[str]) -> Optional[int]:
        try:
            if not start_iso or not end_iso:
                return None
            from datetime import datetime
            s = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
            e = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))
            return int((e - s).total_seconds())
        except Exception:
            return None

    elapsed_client = _elapsed_s(started_at_client, finished_at_client)
    elapsed_server = _elapsed_s(started_at_server, finished_at_server)

    results: List[Dict[str, Any]] = payload.get('results') or []

    # Persist as JSON file and append to CSV
    submission_id = (payload.get("submission_id") or "").strip()
    if not submission_id:
        return jsonify({'error': 'missing submission_id'}), 400

    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base = f"{ts}_{worker_id}_page{page}_{submission_id}"
    json_path = os.path.join(OUTPUT_DIR, base + '.json')

    existing = [fn for fn in os.listdir(OUTPUT_DIR) if fn.endswith(f"_{submission_id}.json")]
    if existing:
        return jsonify({'status': 'ok', 'deduped': True}), 200

    task_set_id = session.get("task_set_id")
    csv_path_in_session = session.get("csv_path")
    csv_name = os.path.basename(csv_path_in_session) if csv_path_in_session else None


    doc = {
        "task_set_id": task_set_id,
        "csv_name": csv_name,
        'worker_id': worker_id,
        'page': page,
        'started_at_client': started_at_client,
        'finished_at_client': finished_at_client,
        'elapsed_client_seconds': elapsed_client,
        'started_at_server': started_at_server,
        'finished_at_server': finished_at_server,
        'elapsed_server_seconds': elapsed_server,
        'pause_events': pause_events,
        'pause_intervals': pause_intervals,
        'effective_elapsed_seconds': effective_elapsed_sec,
        'results': results,
    }
    atomic_write_json(json_path, doc)

    total_pages = 1  
    next_url = None
    if total_pages <= 0:
        next_url = url_for('worker', _external=True)
    elif page >= total_pages:
        next_url = f"{BASE_URL}/complete"
    else:
        next_url = url_for('worker_default', _external=True)

    completed_all = (total_pages <= 0) or (page >= total_pages)

    return jsonify({'status': 'ok', 'next_url': next_url, 'completed_all': completed_all})



if __name__ == '__main__':
    # Allow overriding port via env
    port = int(os.environ.get('PORT', '5001'))
    app.run(host='0.0.0.0', port=port, debug=True)

