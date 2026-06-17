# EduCoder Homework Workflow for Codex

这是一个给 Codex 使用的课程任务工作流仓库，适用于已获授权的 EduCoder 风格内网实践课、AI 作业流程验证、题面读取、外部模型辅助、网页 UI 提交和答案复用记录。

## 内容

- `answers.md`：已整理的题目答案、代码片段、评测状态和平台异常说明。
- `skills/educoder-homework-workflow/`：Codex Skill，可在 Codex 中作为可复用工作流调用。
- `skills/educoder-homework-workflow/references/answers.md`：随 Skill 打包的答案库副本，便于多账号或重复任务复用。

## 使用方式

1. 在 Codex 中安装或引用 `skills/educoder-homework-workflow`。
2. 打开已授权的课程任务页面。
3. 让 Codex 使用该 Skill 读取题面、检索答案库、必要时调用 DeepSeek 类外部模型辅助生成答案。
4. 只通过平台可见网页 UI 提交和评测。
5. 通过后把新答案同步写回 `answers.md` 和 Skill 的 `references/answers.md`。

## 补充缺失答案

答案库不会覆盖所有课程、班级或后续新增题目。遇到答案库没有的题目时，使用者应让自己的 Codex 按 Skill 流程完成题目，并把通过后的答案贡献回来：

1. 先在 `answers.md` 和 `skills/educoder-homework-workflow/references/answers.md` 中按题目标题或 URL 搜索。
2. 如果没有命中，读取可见题面、starter code、评测错误和输出。
3. 必要时把题面、代码和错误粘贴给 DeepSeek 辅助生成答案；不要粘贴账号、密码、token、Cookie 或浏览器存储。
4. 只通过平台网页 UI 提交，确认评测通过或记录平台异常。
5. 把新答案追加到根目录 `answers.md`。
6. 同步复制到 `skills/educoder-homework-workflow/references/answers.md`。
7. 提交 PR 或 push 到本仓库，提交信息建议使用 `docs: add answers for <task name>`。

## 安全边界

- 本仓库不包含 API key、GitHub token、平台账号、平台密码、Cookie 或浏览器会话数据。
- 不要把账号、密码、token、Cookie、浏览器存储、个人文件或私有日志提交到仓库。
- 外部模型只应接收题面、当前代码、可见错误和评测输出，不应接收凭据。
- 该工作流只面向授权课程/实验环境；不要用于绕过验证码、伪造身份、攻击平台或绕过平台评测。

## 平台说明

目标环境是学校内网中的 EduCoder 风格实践平台。答案库里保留了部分任务 URL 用于定位题目；这些 URL 不包含登录凭据，访问仍需要合法网络和账号权限。

## GitHub 仓库设置

这个仓库主要作为 Codex Skill 和答案库的公开分发包使用。Issues、Wiki、Projects 和 Actions 可以关闭，避免把它误当成通用软件项目维护。
