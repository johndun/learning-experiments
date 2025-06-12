## Python Tech Stack

- **Python Version:** Python 3.12
- **Package Management:** Use `uv` for managing dependencies.
- **Code Quality & Formatting:** Use `ruff` for both linting and formatting with 100-character line length and black-compatible style.
- **Testing Framework:** `pytest` with coverage reports.
- **Typing and Documentation:** Type hints are required. Google-style docstrings should be used.
- **Data Structures:** Use `dataclasses` for defining structured data.
- **Code Quality Guidelines:**
    - Use `ruff check` for linting and `ruff format` for formatting.
    - Line length maximum: 100 characters.
    - Follow black-compatible formatting style.
    - Automatic import sorting and organization.
    - Comprehensive linting rules for code quality and consistency.
- **Testing Guidelines:**
    - Tests should be written using `pytest`.
    - Aim for high coverage, with coverage reports generated automatically during CI.
    - Tests should be isolated, repeatable, and cover all critical paths.
    - When working on a script (in `scripts/`), do not write tests

## Coding Guidelines

- Keep file size small. Write functions in their own files.
- Create new files for new tests.

## Script Guidelines

- When working on a script (in `scripts/`), do not write tests