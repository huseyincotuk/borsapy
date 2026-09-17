# AGENTS.md

## Project

This repository is `borsapy`, a Python library for Turkish financial market
data and analysis, with a focus on Borsa İstanbul (BIST).

The project is being extended to support a comprehensive BIST equity
research workflow. Existing `borsapy` functionality must remain stable
unless a change is explicitly required.

## Core Principles

1. Do not invent financial or market data.
2. Do not silently fabricate, estimate, or fill missing financial data.
3. Preserve the provenance of externally sourced data.
4. Prefer primary sources when available, especially KAP for company
   disclosures and official market/regulatory sources for market data.
5. Treat dates, reporting periods, units, currencies, and point-in-time
   information carefully. Do not mix periods without explicitly handling it.
6. Preserve backward compatibility of the existing public API whenever
   reasonably possible.
7. Prefer small, focused, reversible changes over broad refactors.
8. Do not change unrelated files or functionality.

## Data Sources

For every new external data integration:

- Identify the source.
- Document how the data is retrieved.
- Handle unavailable data explicitly.
- Validate and normalize the returned data.
- Preserve important source metadata such as dates, periods, identifiers,
  and document references when applicable.
- Do not treat secondary-source data as equivalent to primary-source data.

Source-specific implementation details belong in project documentation,
not in this file.

## Financial Data Integrity

Financial calculations must be deterministic and reproducible.

Pay particular attention to:

- reporting periods;
- fiscal year versus calendar year;
- quarterly versus trailing-twelve-month values;
- adjusted versus unadjusted prices;
- TL versus other currencies;
- millions/thousands/unit scaling;
- stock splits and capital actions;
- missing values;
- restated financial statements.

Never hide a missing or conflicting value by silently substituting another
value.

## Code Changes

Before modifying code:

1. Inspect the existing implementation and related tests.
2. Understand the current public API and data flow.
3. Identify the smallest appropriate change.
4. Add or update tests for the changed behavior.

For new data adapters or integrations, include tests covering:

- normal/valid data;
- empty or missing data;
- malformed responses;
- relevant error conditions;
- regression cases for important bugs.

Network operations must have appropriate timeout and error handling.
Do not introduce infinite retries or unbounded waits.

## Testing and Quality

Use the repository's `uv` environment.

Run:

    uv run pytest -q
    uv run ruff check .

Do not weaken, delete, skip, or modify tests merely to make a change pass.

If an existing test or lint failure is unrelated to the current change,
identify it explicitly rather than silently fixing unrelated code.

## Git

Work from `develop`.

Use a dedicated feature branch for substantive changes:

    feature/<short-description>

Keep commits focused and descriptive.

Do not commit generated caches, virtual environments, credentials,
API keys, secrets, or local configuration.

Before completing a change:

- inspect `git diff`;
- inspect `git status`;
- run relevant tests;
- run the full test suite when practical.

## Agent Behavior

When asked to implement a feature:

1. First inspect the relevant repository code and documentation.
2. Explain the intended approach briefly.
3. Implement the smallest complete change.
4. Test the change.
5. Report what changed, what was tested, and any remaining limitations.

Do not make unrelated improvements unless explicitly requested.

When requirements are ambiguous and the ambiguity could affect data
correctness, API compatibility, or architecture, stop and ask for
clarification rather than guessing.

## Scope

The long-term objective is to build a reliable BIST research stack around
`borsapy`, including additional market/company data sources and research
capabilities.

However, new functionality must be introduced incrementally and should
not be added merely because it might be useful.

Existing functionality is the baseline and must be protected.
