
# Contributing to NextRec

First off, thank you for taking the time to improve NextRec. Every bug report, pull request, tutorial, or documentation fix makes the project more useful to the recommendation systems community. This document describes how to get involved and what we expect from contributors so that collaboration stays efficient and respectful.

## Ways to Contribute

- **Report bugs** – Let us know about incorrect behavior, crashing experiments, or unexpected metrics by opening a GitHub issue that includes reproduction steps, environment details, and logs.
- **Suggest enhancements** – Propose new models, metrics, datasets, or tooling improvements with a short problem statement and why the feature matters.
- **Improve documentation** – Clarify tutorials, docstrings, READMEs, or notebooks. Screenshots, diagrams, or small examples are welcome.
- **Triage issues** – Help verify reports, reproduce bugs, or close stale issues after confirmation with maintainers.
- **Implement features and fixes** – Pick up open issues (or propose new ones) and submit clean pull requests with tests when possible.

## Before You Start

1. Review `README.md`/`README_zh.md` to understand the project scope, dependencies, and current limitations.
2. Search the [issue tracker](https://github.com/zerolovesea/NextRec/issues) to avoid duplicates.
3. For larger work (new models, API changes, data loaders), open an issue or discussion to confirm direction before investing significant time.

## Development Environment

NextRec uses Python 3.10+. To set up a local environment:

```bash
git clone https://github.com/zerolovesea/NextRec.git
cd NextRec
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]        # or `pip install -r requirements.txt` for runtime deps
```

Helpful commands:

- `pytest` – runs the test suite (configured via `pytest.ini`)
- `pytest -m "not slow"` – skip long-running tests
- `pytest --cov=nextrec` – collect coverage when touching core modules
- `python -m nextrec.scripts.<name>` – run utilities located in `scripts/` (see README/docstrings for usage)

Please keep dependencies minimal and discuss any new requirement beforehand.

## Coding Standards

- Follow Python best practices with type hints when practical.
- Maintainers format code with tools like `black`, `ruff`, or `isort`. Run your preferred formatter before submitting.
- Document new public APIs with docstrings and update relevant markdown files under `docs/`, `tutorials/`, or examples.
- Keep functions and modules focused; add comments only when logic is non-obvious.
- Place datasets and checkpoints outside of the repository when possible. For new sample data, prefer lightweight fixtures under `test/` or documented download scripts.

## Testing

Every code change should be covered by unit or integration tests when feasible. If adding or modifying a model, include:

- Shape/typing checks for inputs and outputs
- at least one deterministic evaluation (e.g., on synthetic data)
- regression tests for training utilities or metrics

If tests are not practical (e.g., due to GPU-only code), explain the manual validation steps in your pull request so reviewers know how to reproduce results.

## Pull Request Process

1. Fork the repository and create a feature branch named after the issue or purpose, e.g., `feature/mtl-loss-refactor`.
2. Commit changes in logical chunks with meaningful messages. Reference issues using `Fixes #<id>` or `Closes #<id>` when appropriate.
3. Update documentation, changelog entries, or tutorial notebooks if behavior changes.
4. Run the full test suite locally before opening the PR.
5. Fill out the pull request template, summarizing the motivation, changes, and testing.
6. Respond to review feedback promptly and keep discussions respectful. Use GitHub suggestions or follow-up commits to address comments.

We prefer smaller, focused pull requests over large, multi-purpose ones. If a change spans multiple areas, consider splitting it into incremental PRs.

## Reporting Security Issues

Please do **not** file public GitHub issues for security vulnerabilities. Instead, email `zyaztec@gmail.com` with details and reproduction instructions. We will coordinate a fix and disclosure timeline with you.

## Community Expectations

All contributors and maintainers must follow the [NextRec Code of Conduct](CODE_OF_CONDUCT.md). Be patient, give others credit, and remember that we are building this project in our spare time. Thoughtful communication helps us make NextRec better for everyone.
