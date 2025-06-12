## Writing Rules

- Write concisely; omit needless modifiers and embellishments.

## Git

- Before commiting, run code checks and tests:
    ```
    uv run ruff check --fix && uv run ruff format && uv run pytest --cov
    ```
- Use concise single sentence commit messages

## Memory Bank

- Read the memory bank files in this order (skipping those not present):
  - projectArchitecture.md
  - currentContext.md
  - progress.md

## Implementation Plans

Follow these rules when asked to create an **implementation plan**.

- Create new implementation plans by overwriting the contents of `memory-bank/progress.md`.
- Structure plans using level-2 markdown headers for high level tasks and checklists ("- [ ]") for individual tasks.
- Each task should be self contained and correspond to a single commit.
- When applicable, explicitly describe tests to be written for each task.
- Always end your turn after writing an implementation plan.
