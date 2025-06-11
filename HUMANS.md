Run tests with coverage:

```bash
uv run pytest --cov
```

Check for linting issues:

```bash
uv run ruff check
```

Check and automatically fix issues where possible:

```bash
uv run ruff check --fix
```

Format code:

```bash
uv run ruff format
```

Check formatting without making changes:

```bash
uv run ruff format --check
```

Run everything:

```bash
uv run ruff check --fix && uv run ruff format && uv run pytest --cov
```
