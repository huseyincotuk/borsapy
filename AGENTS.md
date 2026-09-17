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

## Evidence-First Workflow

Repository analysis must be evidence-driven.

Do not infer repository capabilities from filenames, class names, comments,
documentation, search results, or partial code inspection alone.

For repository-wide analysis, use the following workflow:

1. Discover the relevant files.
2. Read the implementation that is necessary to support each claim.
3. Record concrete evidence before drawing conclusions.
4. Separate verified facts from unresolved or unverified items.
5. Only then produce the final synthesis.

A search result identifies where to investigate; it is not evidence by
itself that a feature or behavior exists.

Do not claim that a provider, API, endpoint, feature, fallback, cache,
retry mechanism, authentication method, or data field exists unless it
has been verified in the relevant implementation.

When examining external data sources:

- Copy hostnames and endpoint paths only from code actually inspected.
- Never reconstruct, autocomplete, normalize, or guess a URL.
- Never infer an API provider solely from a module or class name.
- Distinguish between URLs actually requested by code and URLs that appear
  only in comments, examples, tests, or documentation.
- If an endpoint or source cannot be verified, report it as `unverified`.
- If code dynamically constructs an endpoint, describe only the verified
  components and state that the final URL is constructed dynamically.

When reporting repository findings, use these confidence labels where
appropriate:

- `VERIFIED` — directly supported by inspected implementation.
- `PARTIAL` — some relevant implementation was inspected, but the complete
  behavior was not established.
- `UNVERIFIED` — insufficient evidence was inspected to support the claim.

Never convert `PARTIAL` or `UNVERIFIED` findings into factual statements
in summaries.

## Large Analysis Tasks

Do not attempt to understand the entire repository in one reasoning step.

For broad analysis:

1. Build a file inventory.
2. Divide the investigation into logical components.
3. Inspect one component at a time.
4. Keep intermediate findings concise.
5. Re-check important claims against implementation before the final report.

Do not stop repository exploration merely because enough information seems
available to produce a plausible answer.

If the requested scope is too large to verify reliably in one pass, say so
and propose or perform a staged investigation instead of filling gaps with
assumptions.

Prefer targeted `Glob`, `Grep`, and `Read` operations over broad,
unstructured searches.

Use only tools that are actually available in the current environment.
If a tool call fails because the tool does not exist, do not invent another
tool name. Continue using the available tools.

## Reporting Requirements

For technical repository analysis:

- Reference important findings by repository-relative file path.
- Distinguish observed implementation from interpretation.
- Do not state repository-wide negatives such as "no implementation exists",
  "all providers behave this way", or "there are no circular dependencies"
  unless the relevant scope was systematically inspected.
- Do not describe external endpoint availability unless it was actually
  tested and such testing was part of the requested task.
- Do not present malformed, incomplete, or uncertain URLs as valid endpoints.
- Prefer omission or `UNVERIFIED` over speculation.

Before submitting a final analysis, perform a consistency check:

1. Is every important factual claim supported by inspected code?
2. Did I infer anything from a filename or search result alone?
3. Did I reconstruct or guess any external URL?
4. Did I generalize from one provider or module to the whole repository?
5. Did I claim absence without systematically checking the relevant scope?

If any answer indicates insufficient evidence, revise the report before
submitting it.

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
