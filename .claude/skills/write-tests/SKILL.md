---
name: write-tests
description: Create or update unit tests for code
---

Immediately invoke the Task tool with `subagent_type: "unit-test-writer"` to write tests.

The prompt should include:
- The file(s) or function(s) that need tests
- Whether to create new tests or update existing ones
- Any specific edge cases or scenarios to cover
- Reference to existing test patterns in `tests/` directory

After the unit-test-writer agent completes, suggest running `/test` to verify the new tests pass.
