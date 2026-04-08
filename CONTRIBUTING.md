# Contributing

Thanks for contributing to Ark Chatbot Context Engine.

## Scope

This repository is focused on one thing: persistent memory for conversational chatbots.

Good contributions usually improve one of these areas:
- Context assembly and token budgeting
- Memory extraction, deduplication, and retrieval
- Provider integrations
- Database schema and query performance
- API reliability and developer experience
- Tests, docs, and reproducible setup

Please avoid broad platform changes that turn the project into a general agent framework.

## Development Setup

Requirements:
- Python 3.12+
- PostgreSQL with `pgvector`

Local setup:

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d db
```

You can also use `./start.sh` for a local dev flow.

## Running Tests

Run the full suite:

```bash
python -m pytest tests/ -v
```

Skip live-provider tests:

```bash
python -m pytest tests/ -m "not live"
```

If your change affects providers, retrieval, or memory extraction behavior, add or update tests.

## Before Opening a PR

- Keep changes focused.
- Update docs when behavior changes.
- Add regression coverage for bug fixes.
- Call out any migration or configuration impact clearly.

## Design Notes

- The default path should remain easy to run locally.
- Memory quality matters more than adding many new abstractions.
- Provider support should be explicit and testable.
- Performance-sensitive paths should stay simple and inspectable.

## Reporting Issues

When reporting a bug, include:
- What you expected
- What happened instead
- Reproduction steps
- Relevant `.env` settings if configuration matters
- Whether you are using local Postgres, Docker Compose, or a custom deployment

