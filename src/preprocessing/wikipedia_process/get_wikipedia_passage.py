#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get Wikipedia passages by section from Wikipedia dump.
"""

from lxml import etree
from bs4 import BeautifulSoup
import mwparserfromhell
from multiprocessing import Pool, cpu_count
import os
import json
import re
from tqdm import tqdm
from collections import defaultdict
from JaReIR.src.utils.configs import load_configs
CFG = load_configs("configs")
os.makedirs(CFG["data_config"]["paths"]["wikipedia_psg_path"], exist_ok=True)

def local_name(tag):
    return etree.QName(tag).localname

def recursive_text_extract(part_tree, part_text, current_title, current_text, texts, titles):
    """
    Extract texts recursively by following section hierarchy.
    """
    if current_title not in part_tree or current_title not in part_text:
        raise ValueError(f"Title {current_title} not found in part_tree or part_text.")
    if len(part_tree[current_title]["child"]) == 0:
        texts.append({
            "title" : titles,
            "text" : current_text + [part_text[current_title]]
        })
        return
    current_text = current_text + [part_text[current_title]]
    for child_title in part_tree[current_title]["child"]:
        recursive_text_extract(part_tree, part_text, child_title, current_text, texts, titles + [child_title])

def title_format(title):
    title = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', title)  # [[リンク|表示名]] -> 表示名
    title = re.sub(r'\[\[(.*?)\]\]', r'\1', title)  # [[リンク]] -> リンク
    title = re.sub(r'<!--.*?-->', '', title)
    title = re.sub(r'<ref.*?>.*?</ref>', '', title, flags=re.DOTALL)
    title = re.sub(r'<ref.*?/>', '', title)
    title = re.sub(r'<ref.*', '', title)
    title = re.sub(r'^.*?</ref>', '', title)
    title = title.strip()
    return title

def remove_wiki_tables(text):
   
    while '{|' in text:
        start = text.find('{|')
        if start == -1:
            break
        depth = 1
        pos = start + 2
        while pos < len(text) and depth > 0:
            if text[pos:pos+2] == '{|':
                depth += 1
                pos += 2
            elif text[pos:pos+2] == '|}':
                depth -= 1
                pos += 2
            else:
                pos += 1
        text = text[:start] + text[pos:]
    return text

def remove_files_and_images(text):

    pattern = re.compile(r'\[\[(File|ファイル|Image|画像):[^\]]*\]\]', flags=re.IGNORECASE)

    text = pattern.sub('', text)
    
    while True:
        match = re.search(r'\[\[(File|ファイル|Image|画像):', text, flags=re.IGNORECASE)
        if not match:
            break
        start = match.start()
        depth = 0
        pos = start
        while pos < len(text):
            if text[pos:pos+2] == '[[':
                depth += 1
                pos += 2
            elif text[pos:pos+2] == ']]':
                depth -= 1
                pos += 2
                if depth == 0:
                    break
            else:
                pos += 1
        text = text[:start] + text[pos:]
    return text

def clean_wikitext(text):

    # HTMLタグ系
    text = re.sub(r'<gallery[^>]*>.*?</gallery>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<timeline[^>]*>.*?</timeline>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<math[^>]*>.*?</math>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<score[^>]*>.*?</score>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<mapframe[^>]*>.*?</mapframe>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Wikiテーブル
    text = remove_wiki_tables(text)
    
    # ファイル・画像
    text = remove_files_and_images(text)
    
    return text

def extract_section_hierarchy(wikitext):

    pattern = re.compile(r'(<ref\b[^>]*?)\s*/>\s*<ref\b[^>]*?>',
                         flags=re.IGNORECASE | re.DOTALL)
    wikitext = pattern.sub(r'\1></ref>', wikitext)
    pattern = re.compile(
            r'<!--.*?-->'              # コメント
            r'|<ref\b[^>]*?\s*/>'      # 自己終了タグ
            r'|<ref\b[^>]*?>.*?</\s*ref\s*>'  # 開始+終了タグ
            r'|<ref\b[^>]*?>',         # 壊れた開タグ
            flags=re.DOTALL | re.IGNORECASE
        )
    wikitext = pattern.sub('', wikitext)

    wikicode = mwparserfromhell.parse(wikitext)
    heading = wikicode.filter_headings()
    part_tree = {}
    if heading:
        pre_parent_num = -1
        pre_parent_name = None
        duplicate_count = defaultdict(int)
        for head in heading:
            if head.count("=") > 2 and head.count("==") == 0:
                # =を含む文
                continue
            if pre_parent_num == -1:
                pre_parent_num = head.count("=") // 2
                title = head.replace("=", "").strip()
                title = title_format(title)
                part_tree[title] = {"parent" : None, "child" : [] , "tag" : pre_parent_num}
                pre_parent_name = title
            else:
                current_count = head.count("=") // 2
                title = head.replace("=", "").strip()
                title = title_format(title)
                if title in part_tree:
                    duplicate_count[title] += 1
                    title = title + "_dup" + str(duplicate_count[title])
                if pre_parent_num == current_count:
                    tmp_parent = part_tree[pre_parent_name]["parent"]
                    if tmp_parent is not None:
                        part_tree[tmp_parent]["child"].append(title)
                    part_tree[title] = {"parent" : tmp_parent, "child" : [], "tag" : current_count}
                    pre_parent_name = title
                elif pre_parent_num < current_count:
                    part_tree[pre_parent_name]["child"].append(title)
                    part_tree[title] = {"parent" : pre_parent_name, "child" : [], "tag" : current_count}
                    pre_parent_num = current_count
                    pre_parent_name = title
                else:
                    while_counter = 0
                    while pre_parent_num > current_count and pre_parent_name is not None:
                        #print("Going up from:", pre_parent_name,  pre_parent_num, current_count, duplicate_count)
                        pre_parent_num = part_tree[pre_parent_name]["tag"]
                        pre_parent_name = part_tree[pre_parent_name]["parent"]
                        while_counter += 1
                        if while_counter > 10000:
                            raise RuntimeError("Infinite loop detected in section hierarchy parsing.")

                    tmp_parent = part_tree[pre_parent_name]["parent"] if pre_parent_name is not None else None
                    if tmp_parent is not None:
                        part_tree[tmp_parent]["child"].append(title)
                    part_tree[title] = {"parent" : tmp_parent, "child" : [], "tag" : current_count}
                    pre_parent_num = current_count
                    pre_parent_name = title
        #print(part_tree)
        wikitext_split = wikitext.split("\n")
        
        # 導入部（最初の見出し前のテキスト）を抽出
        intro_text = []
        ignore_pattern = re.compile(r"thumb|ファイル:|File:|^Category:|^カテゴリ:", flags=re.IGNORECASE)
        intro_done = False
        
        part_text = {}
        tmp_text = []
        current_title = None
        duplicate_count = defaultdict(int)
        current_text = None
        seen_titles = set()
        for text in wikitext_split:
            current_text = text
            #print(current_text, current_title)
            current_text = current_text.strip()
            current_text = title_format(current_text)
  
            if current_text.startswith("=") and current_text.endswith("="):
                tmp_title = current_text.replace("=", "").strip()
                tmp_title = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', tmp_title)
                # 保存を先に行う
                if current_title is not None:
                    part_text[current_title] = "\n".join(tmp_text)
                    tmp_text = []
                # 重複チェック（seen_titlesを使用）
                if tmp_title in seen_titles:
                    duplicate_count[tmp_title] += 1
                    tmp_title = tmp_title + "_dup" + str(duplicate_count[tmp_title])
                seen_titles.add(tmp_title)
                intro_done = True  # 最初の見出しに到達
                
                current_title = tmp_title
            else:
                #current_text = mwparserfromhell.parse(current_text).strip_code().strip()
                if len(current_text) == 0:
                    continue
                elif ignore_pattern.search(current_text):
                    continue
                if current_title is not None:
                    tmp_text.append(current_text)
                elif not intro_done:
                    # まだ見出しに到達していない → イントロ部分
                    intro_text.append(current_text)
            current_text = None
        if len(tmp_text) >= 0 and current_title is not None:
            part_text[current_title] = "\n".join(tmp_text)
        # print(strip_codes)
        texts = []
        for key, value in part_tree.items():
            parent = value["parent"]
            if parent is None:
                recursive_text_extract(part_tree, part_text, key, [], texts, [key])
        # イントロテキストを結合して返す
        intro_joined = "\n".join(intro_text) if intro_text else None
        return (intro_joined, texts)
        
    # 見出しがない場合でもイントロを抽出
    wikitext_split = wikitext.split("\n")
    intro_text = []
    ignore_pattern = re.compile(r"thumb|ファイル:|File:|^Category:|^カテゴリ:", flags=re.IGNORECASE)
    for line in wikitext_split:
        stripped = line.strip()
        stripped = title_format(stripped)
        if stripped and not ignore_pattern.search(stripped):
            intro_text.append(stripped)
    
    intro_joined = "\n".join(intro_text) if intro_text else None
    return (intro_joined, None)

def tasks(args):
    page_id, title, text = args
    res = []
    # print("Title:", title)
    # print(page_id)
    try:
        result = extract_section_hierarchy(text)
    except Exception as e:
        print(f"Error processing page {page_id} ({title}): {e}")
        with open("error_log.txt", "a", encoding="utf-8") as f:
            f.write(title + "\n")
        return []
    
    intro_text, texts = result
    section_blacklist = ["脚注", "参考文献", "外部リンク", "関連項目", "注釈"]
    idx = 0
    
    # イントロ（導入部）を追加
    if intro_text:
        saved_intro = clean_wikitext(intro_text)
        saved_intro = str(mwparserfromhell.parse(saved_intro).strip_code())
        saved_intro = re.sub(r'[\ud800-\udfff]', '', saved_intro)
        saved_intro = saved_intro.replace("\x00", "")
        if saved_intro.strip():
            save_data = {
                "id": f"{page_id}_{idx}",
                "title": title,
                "section_titles": [title],  # Wikipediaタイトルのみ
                "text": saved_intro
            }
            res.append(save_data)
            idx += 1
    
    # セクションを追加
    if texts is not None:
        for data in texts:
            # ブラックリストチェック（section_titlesにWikipediaタイトルは含めない）
            if any(blk in sec_title for sec_title in data["title"] for blk in section_blacklist):
                continue
            saved_text = "\n".join(data["text"])
            saved_text = clean_wikitext(saved_text)
            saved_text = str(mwparserfromhell.parse(saved_text).strip_code())
            saved_text = re.sub(r'[\ud800-\udfff]', '', saved_text) 
            saved_text = saved_text.replace("\x00", "")
            # 空のテキストをスキップ
            if not saved_text.strip():
                continue
            save_data = {
                "id": f"{page_id}_{idx}",
                "title": title,
                "section_titles": [title] + data["title"],  # Wikipediaタイトル + セクション見出し
                "text": saved_text
            }
            res.append(save_data)
            idx += 1
    return res
def main():
    context = etree.iterparse(CFG["data_config"]["paths"]["wikipedia_dump_path"], events=("end",))
    #print(context)
    black_title_list = ["Template", "Wikipedia:", "プロジェクト:", "ファイル:"]
    batch = []
    batch_count = 0
    for event, elem in context:
        title = None
        text = None
        page_flag = False
        for child in elem.iter():
            ln = local_name(child.tag)
            if ln == "title":
                title = child.text
            elif ln == "text":
                text = child.text
            elif ln == "page":
                page_flag = True
            if page_flag and ln == "id":
                page_id = child.text
                page_flag = False
        if title is None or text is None:
            #elem.clear()
            continue
        if any([blk in title for blk in black_title_list]):
            continue
        if title == "柔道整復師":
            text = text.replace("柔道整復師の扱う対象ではないとしている<ref", "")
        elif title == "ギャグ":
            text = text.replace('<ref name=":0" /><ref>{{Cite book|和書 |title=世界の喜劇人 |date=2016年3月18日 |publisher=[[新潮社]] |author=[[小林信彦]] |edition=電子版 |chapter=序章　遥かなる喝采 |quote=荻昌弘氏は、ギャグを大きく三つに分けて、①音響と物体の映像によるギャグ②セリフによるギャグ③アクションによるギャグとし、これらのすべては、人間の不幸の上に成立する、と指摘している。そして、これらのほかに、マルクス兄弟のそれのような純ナンセンスのギャグがある、というわけだが、（中略）③のアクションによるものと、ナンセンス性の結びついたものが、サイレントの傑作ギャグである。}}</ref>', "")
        elif title == "Myunとyayo〜":
            text = text.replace("=== シングル ===みゅｎ", "=== シングル ===")
        batch.append((page_id, title, text))
        if len(batch) >= 10 ** 6:
            save_datas = []
            with Pool(cpu_count()) as p:
                for result in tqdm(p.imap_unordered(tasks, batch), total=len(batch)):
                    if result is not None:
                        save_datas.extend(result)
            batch = []
            for save_data in save_datas:
                save_path = os.path.join(CFG["data_config"]["paths"]["wikipedia_psg_path"], f"{batch_count}.jsonl")
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(save_data, ensure_ascii=False) + "\n")
            batch_count += 1

    if len(batch) > 0:
        save_datas = []
        with Pool(cpu_count()) as p:
            for result in tqdm(p.imap_unordered(tasks, batch), total=len(batch)):
                if result is not None:
                    save_datas.extend(result)
        batch = []
        for save_data in save_datas:
            save_path = os.path.join(CFG["data_config"]["paths"]["wikipedia_psg_path"], f"{batch_count}.jsonl")
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(save_data, ensure_ascii=False) + "\n")
        batch_count += 1

if __name__ == "__main__":
    main()
