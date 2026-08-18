# Contributing

Thanks for your interest in improving this project.

## Development setup

```bash
git clone https://github.com/srikala-g/credit-risk-analyzer.git
cd credit-risk-analyzer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make dev            # installs the package + dev tools + pre-commit hooks
cp .env.example .env   # then add your Gemini key
```

## Before you open a pull request

```bash
make format         # auto-format
make lint           # check style
make test           # run tests
```

Pre-commit hooks run these automatically on commit once `make dev` has installed
them.

## Guidelines

- Keep changes focused; one logical change per pull request.
- Add or update tests for behaviour changes.
- Keep functions small and documented; this codebase favours readability so each
  component can be explained clearly.
- Update the relevant docs in `docs/` when you change behaviour or architecture.
