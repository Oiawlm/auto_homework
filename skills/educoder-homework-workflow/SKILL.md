---
name: educoder-homework-workflow
description: Use when completing authorized EduCoder-style intranet homework, AI training, programming challenge, or course workflow tasks where Codex must read visible task requirements, optionally use DeepSeek-style model assistance, submit through the browser UI, and save reusable answers or status notes.
---

# Educoder Homework Workflow

## Overview

Use this workflow for authorized course practice tasks on EduCoder-like platforms. The goal is to finish the task through the visible UI, use external model help when useful, and record reusable answers without bypassing authentication, CAPTCHA, platform checks, or hidden APIs.

This skill is intended for Codex. It is not a standalone bot, scraper, credential manager, or platform bypass tool.

## Workflow

1. If the user is working on this known course or a repeated account, search `references/answers.md` by task title, task URL, or visible progress label before generating a new answer.
   - If the answer bank contains a passed answer, use it as the first candidate.
   - If it records a no-pass task or platform anomaly, skip it unless the recorded retry condition has changed.
2. Enter the task through the deepest page that is already open. Use exactly one transition at a time:

| Current page | Action | Success signal |
|---|---|---|
| Homework list | Find the row with the exact task title and click that row's `查看作品` once. | A tab with `/shixun_homework/<id>/detail` exists. |
| Work/detail or overall-evaluation page | Focus the `/detail` tab and click the right-side `进入实训` once. | A task tab with `/tasks/` exists. |
| Task page | Read the visible requirements and continue with the answer or evaluation. | The task title, editor, questions, or environment controls are visible. |

   After each transition, inspect controlled and user-visible tabs before clicking again. If the click reports a timeout but the success-signal tab exists, continue from that tab. Prefer an existing task page over a detail page, and an existing detail page over the homework list.
3. Identify the task type:
   - Code challenge: starter code, required output, tests, current failure.
   - Multiple-choice or short-answer: question text and options.
   - Experience task: environment launch, external website requirement, whether a platform evaluation button exists.
   - Special no-pass task: record the answer or workflow note if the page says pass is not required.
4. Capture only task-relevant material for model assistance:
   - Title and URL.
   - Requirements and examples.
   - Starter code or current code.
   - Visible error, raw actual output, expected output if visible.
   - Do not send usernames, passwords, cookies, tokens, personal files, or private browser state to external sites.
5. If the user requested DeepSeek assistance or the task is nontrivial, paste the captured material into the external model and ask for a concise answer. Enable the strongest reasoning/search mode available if the site exposes one.
6. Verify the proposed answer locally or by reasoning before submitting. External model output is a draft, not proof.
7. Submit only through the visible platform UI. Do not inspect cookies, local/session storage, hidden internal APIs, or attempt to forge completion.
8. After each submission, read the result panel:
   - If passed, immediately record the title, task URL, final code/answer, and progress in `answers.md`.
   - If failed, use systematic debugging: compare raw output, visible expected output, reference answer, records, comments, and current editor content.
   - Stop blind guessing after repeated failures and document the evidence.

## Reusing Known Answers

- Use `references/answers.md` as the bundled answer bank for this course snapshot.
- Search by exact task title first, then by task URL or level number.
- Treat stored code as the first candidate answer, but still inspect the live task page because platform starter code, hidden tests, or environment behavior can drift.
- When a stored answer still passes, keep it unchanged. When it needs adjustment, update its existing task section in both the workspace `answers.md` and the bundled reference copy.

## Adding Missing Answers

When `references/answers.md` does not contain the current task:

1. Solve the task through the normal workflow.
2. Use DeepSeek only with task text, current code, and visible evaluation feedback.
3. Verify by submitting through the visible platform UI.
4. Add one canonical task section, or update the existing section in place, with the passed answer or diagnosed platform anomaly.
5. Copy the updated answer entry into `skills/educoder-homework-workflow/references/answers.md`.
6. Commit and push the update, or open a PR, with a message like `docs: add answers for <task name>`.
7. Before publishing, scan the worktree and Git history for credentials or token-like strings.

## Public Repository Safety

- Keep credentials out of the repository: no usernames, passwords, API keys, tokens, cookies, localStorage/sessionStorage dumps, or browser profiles.
- Keep the answer bank limited to task text, task URLs, code, output notes, and platform behavior notes.
- Before making a copy public, scan both the worktree and Git history for secret-like strings.

## Browser Editing Rules

- Use stored bare `/tasks/...` URLs only to identify an answer-bank entry. Enter the live task through the current classroom detail page because stored task URLs may return 403 for another classroom.
- If an old tab is unresponsive, open one fresh classroom list page and resume the same list -> detail -> task sequence.
- For Monaco-style editors, click the editor, press `Control+A`, press `Backspace`, then paste/fill the new code. Filling without clearing can append code to stale content.
- After editing, inspect the visible code section and confirm required functions, class names, or changed lines are present before clicking evaluation. A plausible answer in memory is not proof that Monaco accepted it.
- Prefer the platform's `评测` button for final checking. Use `自测运行` only for cheap local feedback.
- After a mismatch, inspect the raw actual-output panel, including hidden diff containers and stderr warnings. Correct stdout can still fail because a library warning was included in the judged output.
- If actual output matches the public sample, compare the reference answer, records, and comments. When the reference implementation also fails or multiple users report the same mismatch, record a platform anomaly instead of continuing blind code changes.
- For multi-level tasks, prefer the visible `上一关`/`下一关` anchor `href` from the current course. Text clicks may not route, and stored level URLs may belong to another classroom.
- If a reference answer is unlocked, compare it with the current editor content before submitting.

## External Model Prompt

Use a focused prompt like this:

```text
请只根据下面的课程题面、 starter code 和评测错误给出可提交答案。
要求：
1. 保留平台要求的输入输出格式。
2. 不要解释太长，先给最终代码/答案。
3. 如果样例输出和正确算法冲突，请指出冲突并优先满足平台评测。

任务标题：
[title]

任务要求：
[visible requirements]

当前代码：
[code]

评测反馈：
[error/raw output/expected output if visible]
```

## Recording Answers

Maintain one canonical section per task or course-specific variant. Update that section in place instead of adding account, date, session, or retry-history blocks.

````markdown
## [task title]

任务页面：
```text
[level/task URLs]
```

评测结果：`x/y`

代码/答案：
```python
[final code]
```

关键限制：
[only the durable condition needed to reproduce the result]

平台状态：
[only for no-pass tasks or anomalies: final evidence and the condition that justifies a future retry]
```
````

Write final-state facts only. Exclude dates, account identifiers, attempt counts, chronological narration, discarded approaches, and phrases such as "this time", "previously", or "after trying". For non-code experience tasks, replace the code block with the exact successful operation sequence or final platform limitation.

## Completion Criteria

Before saying the workflow is done:

- Rescan all homework pages and list every remaining `0/n` item.
- Distinguish real remaining work from tasks marked no-pass or platform tasks with no evaluation entry.
- Do not re-enter known no-pass/zero-completion experience tasks during the final rescan; verify their list status only.
- Confirm `answers.md` contains the latest passed answers and anomaly notes.
- Run `scripts/validate-docs.ps1 -WorkspaceRoot <repo-root>` to confirm the answer-bank copies match and no session-history wording was introduced.
- State which tasks passed, which remain blocked by platform behavior, and what evidence supports that conclusion.
