# Security Policy

## Handling secrets

This project talks to the Gemini API and therefore requires an API key. To keep
keys safe:

- The key is supplied via the `GEMINI_API_KEY` environment variable (locally via
  a `.env` file, on Hugging Face Spaces via a Space secret).
- `.env` is git-ignored; only `.env.example` (with no real values) is committed.
- **Never commit a real key.** If one is ever committed, rotate it immediately —
  removing the commit is not sufficient, because git history retains it.
- Enable GitHub secret scanning and push protection on the repository
  (Settings → Code security and analysis).

## Handling documents

- Uploaded documents are processed in memory for the session and are not
  persisted by the application.
- Do not commit confidential documents to the repository. Public filings
  (e.g. 10-Ks) are fine for demos and tests.

## Reporting a vulnerability

If you find a security issue, please open a private report via GitHub Security
Advisories, or contact the maintainer directly rather than opening a public
issue.
