# CLAUDE.md

- keep your responses short and concise. sacrifice grammar for conciseness
- to run commands, always use `pixi run -e {cpu,gpu}...` to ensure you are in the environment
- use `dataclasses` for configs, and when setting defaults for custom classes, use `field(default_factory = ...)`
- keep the code clean and concise, use and/or create helper functions to do this