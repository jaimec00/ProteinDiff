---
name: review
description: Review recent code changes for quality, performance, and alignment with requirements
---

Immediately invoke the Task tool with `subagent_type: "code-review-critic"` to review recent code changes.

The prompt should include:
- Summary of what code changes were made in this session
- The original user request (if applicable)
- Any files that were modified

Do not ask clarifying questions - launch the agent right away to review the work done so far.
