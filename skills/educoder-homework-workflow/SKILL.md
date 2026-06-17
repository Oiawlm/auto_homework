---
name: educoder-homework-workflow
description: Use when completing authorized EduCoder-style intranet homework, AI training, programming challenge, or course workflow tasks where Codex must read visible task requirements, optionally use DBC/DeepSeek-style model assistance, submit through the browser UI, and save reusable answers or status notes.
---

# Educoder Homework Workflow

## Overview

Use this workflow for authorized course practice tasks on EduCoder-like platforms. The goal is to finish the task through the visible UI, use external model help when useful, and record reusable answers without bypassing authentication, CAPTCHA, platform checks, or hidden APIs.

This skill is intended for Codex. It is not a standalone bot, scraper, credential manager, or platform bypass tool.

## Workflow

1. If the user is working on this known course or a repeated account, search `references/answers.md` by task title, task URL, or visible progress label before generating a new answer.
2. Open the current task in the browser and read the visible page from top to bottom.
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
5. If the user requested DBC/DeepSeek assistance or the task is nontrivial, paste the captured material into the external model and ask for a concise answer. Enable the strongest reasoning/search mode available if the site exposes one.
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
- When a stored answer passes for another account, keep it unchanged. When it needs adjustment, update both the live `answers.md` in the workspace and this bundled reference copy.

## Public Repository Safety

- Keep credentials out of the repository: no usernames, passwords, API keys, tokens, cookies, localStorage/sessionStorage dumps, or browser profiles.
- Keep the answer bank limited to task text, task URLs, code, output notes, and platform behavior notes.
- Before making a copy public, scan both the worktree and Git history for secret-like strings.

## Browser Editing Rules

- For Monaco-style editors, click the editor, press `Control+A`, press `Backspace`, then paste/fill the new code. Filling without clearing can append code to stale content.
- After editing, inspect the visible code section before clicking evaluation.
- Prefer the platform's `评测` button for final checking. Use `自测运行` only for cheap local feedback.
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

Append to `answers.md` after each completed or diagnosed task:

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

备注：
[only include non-obvious platform behavior, failed traps, or external-environment notes]
```
````

For non-code experience tasks, replace the code block with the exact operation sequence and observed platform status.

## Completion Criteria

Before saying the workflow is done:

- Rescan all homework pages and list every remaining `0/n` item.
- Distinguish real remaining work from tasks marked no-pass or platform tasks with no evaluation entry.
- Confirm `answers.md` contains the latest passed answers and anomaly notes.
- State which tasks passed, which remain blocked by platform behavior, and what evidence supports that conclusion.
