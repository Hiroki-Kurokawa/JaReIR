# Manual for Lancers Annotation Task Setup
This document outlines the steps for conducting annotation tasks on the Lancers platform.
## Preparation
- Set URL for annotation server in lancers.html
  - Replace `YOUR_SERVER_URL` with the actual server URL
- Create a Lancers account for posting tasks.
- Configure server access
 - Ensure that the server is accessible
- Workflow: Provide workers with a URL to access the site on Lancers -> Workers access the URL -> Data collection (on the server)
 - It is acceptable for workers to perform tasks outside of Lancers
## Work Procedure
1. Access https://www.lancers.jp/work_form/add?ref=header_menu and create a job posting
2. Step1: Choose "データ収集・入力・リスト作成"
3. Step2:
 - Select "タスク方式"
4. Job Posting Creation Screen:
 - Job Title
  - "質問に対する文章の推論理由の作成"
 - AI Usage Permission
  - "許可する"
 - Client Information
  - "大学・学校"
 - Feature of Job
  - "指定なし"
 - Job Description
  - ある文章と質問の関連性確認と必要な理由の作成をしていただく作業です。大量に募集しています。
 - Style : "だ、である調"
 - Job Details
  - Click "以前の入力形式"
  - Select source -> Copy and paste the content of lancers.html (in this directory)
 - Job Price and Quantity
  - Job Price: 275円
  - Quantity: 2 (2 workers per task)
 - Job Visibility: "ゲストを含むすべてのランサーに作業詳細を公開する"
 - Restriction per worker: "制限なし"
 - Deadline for Job Completion
  - "21日後に締め切る"
5. Confirm Job Details and Create Job Posting
6. Variable File Upload
 - 「変数ファイルをアップロードし、依頼を公開する」をクリック
 - Click "変数ファイルをアップロードし、依頼を公開する"
 - Upload tasks.csv
 - Confirm that the task preview displays the correct work location
7. Advance Payment
 - Set the payment method and make an advance payment
 - This will publish the task and allow workers to participate
## After Posting
 - Monitor progress from task management
 - Respond to questions from workers
 - After work completion, review the submitted content and download the results
 - If there are issues, request rework or reject the submission

