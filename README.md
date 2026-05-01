# HackerRank Orchestrate — Support Triage Agent

A terminal-based AI agent that reads support tickets and produces structured triage decisions (reply or escalate) grounded entirely in a local support corpus — no live web calls for answers.

Built for the **HackerRank Orchestrate** hackathon (May 1–2, 2026).

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Agent](#running-the-agent)
   - [Batch mode](#batch-mode-default)
   - [Sample / test mode](#sample--test-mode)
   - [Interactive mode](#interactive-mode)
   - [Custom input and output paths](#custom-input-and-output-paths)
7. [Input Format](#input-format)
8. [Output Format](#output-format)
9. [Pipeline Walkthrough](#pipeline-walkthrough)
10. [Corpus Structure](#corpus-structure)
11. [TF-IDF Index and Caching](#tf-idf-index-and-caching)
12. [Escalation Rules](#escalation-rules)
13. [LLM Integration](#llm-integration)
14. [Module Reference](#module-reference)
15. [Tuning and Configuration](#tuning-and-configuration)
16. [Troubleshooting](#troubleshooting)
17. [Submission Checklist](#submission-checklist)

---

## How It Works

For each support ticket the agent runs the following pipeline:

```
Ticket (issue + subject + company)
         │
         ▼
 [1] Out-of-scope check          ← regex rules, no LLM call
         │ out-of-scope → ESCALATED immediately
         ▼
 [2] TF-IDF retrieval            ← top-5 most relevant corpus chunks
         │                          company-filtered, falls back to all-corpus
         ▼
 [3] Prompt construction         ← ticket + retrieved context → markdown prompt
         │
         ▼
 [4] Groq LLM call               ← llama-3.1-8b-instant, temp=0
         │
         ▼
 [5] JSON parsing                ← raw → fenced block → bare object
         │
         ▼
 [6] Rule-based validation       ← hard escalation patterns override LLM
         │                          missing fields filled from fallback
         ▼
    Structured output row
    (status / product_area / response / justification / request_type)
```

The agent **never invents answers**. If the local corpus does not contain relevant documentation, the ticket is automatically escalated to a human agent.

---

## Repository Layout

```
.
├── AGENTS.md                          ← Technical spec for AI coding tools
├── README.md                          ← This file
├── .gitignore
├── .env.example              ← template for environment variables
├── code/
│   ├── .env                          ← Created AFTER copying from .env.example and store secrets (GROQ_API_KEY here)
│   ├── agent.py                       ← Core triage pipeline (main logic)
│   ├── config.py                      ← All constants, paths, and allowed values
│   ├── corpus_loader.py               ← Corpus reader, chunker, TF-IDF builder + cache
│   ├── escalation.py                  ← Rule-based escalation (pre- and post-LLM)
│   ├── main.py                        ← CLI entry point
│   ├── prompts.py                     ← System prompt and per-ticket prompt builder
│   ├── retriever.py                   ← Cosine similarity search and context formatter
│   ├── requirements.txt               ← Python dependencies
├── data/
│   ├── hackerrank/                    ← HackerRank help-center articles
│   ├── claude/                        ← Claude Help Center export
│   └── visa/                          ← Visa consumer and small-business support docs
└── support_tickets/
    ├── sample_support_tickets.csv     ← Dev/test tickets (with expected signals)
    ├── support_tickets.csv            ← Evaluation input (no labels)
    └── output.csv                     ← Agent predictions written here
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 or later |
| pip | any recent version |
| Groq API key | Free tier at [console.groq.com](https://console.groq.com) |
| Internet access | Only needed for Groq API calls during inference |

No GPU required. No PyTorch. No ONNX. Runs on any OS.

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:interviewstreet/hackerrank-orchestrate-may26.git
cd hackerrank-orchestrate-may26
```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r code/requirements.txt
```

Dependencies installed:

| Package | Version | Purpose |
|---|---|---|
| `groq` | ≥0.9.0 | Groq LLM API client |
| `scikit-learn` | ≥1.4.0 | TF-IDF vectorizer and cosine similarity |
| `pandas` | ≥2.0.0 | CSV handling |
| `python-dotenv` | ≥1.0.0 | `.env` file loading |
| `tqdm` | ≥4.66.0 | Progress bar for batch mode |
| `numpy` | ≥1.26.0 | Numerical operations |

### 4. Set up your environment file

```bash
cp .env.example code/.env
```

Edit `code/.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_your_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com). The free tier is sufficient to process all evaluation tickets.

> **Never commit `.env` to git.** It is already in `.gitignore`.

---

## Configuration

All tuneable settings are in `code/config.py`. You can adjust these without touching any other file:

| Setting | Default | Effect |
|---|---|---|
| `GROQ_MODEL` | `"llama-3.1-8b-instant"` | Change to `"llama-3.3-70b-versatile"` for higher accuracy (slower, may hit rate limits) |
| `TOP_K_RESULTS` | `5` | Number of corpus chunks retrieved per ticket |
| `MIN_SIMILARITY_SCORE` | `0.1` | Minimum TF-IDF cosine score for a result to count as "good match"; below this → auto-escalate |
| `CHUNK_SIZE` | `500` | Max characters per corpus chunk |
| `CHUNK_OVERLAP` | `100` | Characters of overlap between consecutive chunks |

---

## Running the Agent

All commands are run from the **`code/` directory**:

```bash
cd code
```

### Batch mode (default)

Processes all rows in `support_tickets/support_tickets.csv` and writes results to `support_tickets/output.csv`.

```bash
python main.py
```

Output on first run (corpus not yet cached):
```
[Corpus Loader] Loading corpus files...
  Loading hackerrank: 42 files found
    -> 1834 chunks indexed for hackerrank
  Loading claude: 31 files found
    -> 1102 chunks indexed for claude
  Loading visa: 27 files found
    -> 989 chunks indexed for visa

  Total chunks: 3925
[Corpus Loader] Building TF-IDF index...
[Corpus Loader] Index cached to .tfidf_cache.pkl

[Main] Processing 120 tickets from support_tickets.csv...

Triaging tickets: 100%|████████| 120/120 [02:14<00:00]

==================================================
✓ Done! Results written to: .../support_tickets/output.csv
  Total tickets : 120
  Replied       : 84
  Escalated     : 36
==================================================
```

Subsequent runs load the index from cache instantly.

### Sample / test mode

Uses `support_tickets/sample_support_tickets.csv` (smaller set with expected signals, useful for development):

```bash
python main.py --sample
```

### Interactive mode

Ask the agent about individual tickets directly in the terminal — no CSV needed:

```bash
python main.py --ask
```

Example session:
```
Company (hackerrank / claude / visa) [Enter to skip]: hackerrank
Subject: My test results are not showing
Issue: I completed a coding test 2 hours ago but I can't see my score anywhere in the dashboard.

Triaging...

  Status       : REPLIED
  Product Area : Assessment Platform
  Request Type : product_issue

  Response:
  After completing a test, results may take up to 24 hours to appear in your
  dashboard depending on the assessment type. You can check the status in
  My Tests > Completed. If results are not visible after 24 hours, please
  contact support with your test ID.

  Justification:
  Documentation covers result visibility timelines for completed assessments.
  Ticket is a standard product inquiry, no escalation required.
```

Type `exit` or `quit` at any prompt to stop.

### Custom input and output paths

```bash
python main.py --input path/to/my_tickets.csv
python main.py --output path/to/my_results.csv
python main.py --input path/to/my_tickets.csv --output path/to/my_results.csv
```

---

## Input Format

The agent reads any CSV with at least an `Issue` column. `Subject` and `Company` are optional but improve accuracy.

| Column | Required | Notes |
|---|---|---|
| `Issue` (or `issue`) | Yes | The body of the support ticket |
| `Subject` (or `subject`) | No | Short description / title of the issue |
| `Company` (or `company`) | No | `hackerrank`, `claude`, or `visa`. If absent or unrecognised, the agent searches all corpora. |

Column names are case-insensitive (`Issue` and `issue` both work).

---

## Output Format

Five columns are appended to (or merged into) the output CSV:

| Column | Possible values | Description |
|---|---|---|
| `status` | `replied`, `escalated` | `replied` = agent answered from docs. `escalated` = needs human review. |
| `product_area` | Any string | e.g. "Account Management", "Billing & Payments", "Assessment Platform", "Card Services", "Fraud & Security", "Technical Issue" |
| `response` | Any string | The user-facing message. Escalated responses always include a human handoff statement. |
| `justification` | Any string | 1–3 sentences explaining the routing decision. If an auto-escalation rule fired, this is noted in brackets. |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` | `invalid` for out-of-scope or malicious tickets. |

### When is a ticket `replied`?

Only when all three conditions hold:
1. No hard escalation keyword matched.
2. Retrieval found at least one chunk with similarity ≥ 0.1.
3. The LLM determined the documentation fully answers the issue.

### When is a ticket `escalated`?

Any of these alone is sufficient:
- A hard rule matched (fraud, security, legal, PII, prompt injection, etc.).
- No relevant documentation found in the corpus.
- The LLM determined it could not answer from the available docs.
- The Groq API call failed (safe fallback).

---

## Pipeline Walkthrough

### Step 1 — Out-of-scope check

Before any retrieval or LLM call, `escalation.check_out_of_scope()` scans the ticket for patterns that indicate the request has nothing to do with HackerRank, Claude, or Visa — for example:

- Questions about weather, recipes, sports scores, stock prices, horoscopes.
- Requests to write code, poems, stories, or essays.
- General knowledge questions ("What is the capital of France?").

If matched, the ticket returns immediately with `status=escalated`, `product_area="Out of Scope"`, `request_type="invalid"`. No LLM call is made.

### Step 2 — Corpus retrieval

`retriever.retrieve()` transforms the query `"<subject> <issue>"` into a TF-IDF vector and computes cosine similarity against every chunk in the index.

**Company filtering:** If a company is specified, only that company's chunks are searched first. If the best result scores below the threshold, the search is repeated across all companies.

**Result:** Top-5 chunks by similarity score, plus a `has_good_match` flag.

### Step 3 — Prompt construction

`prompts.build_user_prompt()` formats the ticket metadata and the top-5 retrieved chunks into a markdown string capped at 3000 characters of context.

### Step 4 — LLM call

`agent._call_groq()` sends the system prompt (rules + output schema) and the user prompt (ticket + docs) to Groq. Temperature is 0 for deterministic output.

### Step 5 — JSON parsing

`agent._parse_json_response()` extracts the JSON from the response using three fallback strategies:
1. Direct `json.loads()` on the full response.
2. Regex extraction from a ` ```json ... ``` ` fenced block.
3. Regex extraction of any `{ ... }` object in the response.

If all three fail, `FALLBACK_RESPONSE` is used and the ticket is escalated.

### Step 6 — Validation and escalation override

`agent._validate_and_fix()` applies post-LLM safety checks:
- Re-runs hard escalation rules against the original ticket text.
- If `has_good_match` is `False`, escalates regardless of LLM answer.
- Ensures all required fields are present and valid.
- Ensures escalated responses contain a human handoff statement.

---

## Corpus Structure

The agent uses three product corpora, each stored in `data/`:

| Directory | Product | Typical content |
|---|---|---|
| `data/hackerrank/` | HackerRank | Assessment platform help, account management, billing, technical FAQs |
| `data/claude/` | Claude (Anthropic) | Claude AI usage, API, subscriptions, safety policies |
| `data/visa/` | Visa | Card services, fraud disputes, payment processing, merchant support |

**Supported file types:** `.md`, `.txt`, `.html`, `.htm`, `.csv`

Files are recursively loaded from subdirectories. HTML tags are stripped during loading. Each file is split into overlapping chunks of ~500 characters.

---

## TF-IDF Index and Caching

### Why TF-IDF?

- Zero GPU dependency — runs on any machine, including Windows laptops.
- No ONNX or PyTorch install complexity.
- Fast to build (seconds) and fast to query (milliseconds).
- Bigrams (`ngram_range=(1,2)`) capture multi-word tech phrases naturally.
- `sublinear_tf=True` dampens the effect of very common words.

### Cache

The fitted vectorizer and document matrix are saved to `code/.tfidf_cache.pkl`. The cache stores an MD5 fingerprint of all corpus files (path + size + modification time). If any corpus file changes, the cache is automatically invalidated and rebuilt.

Cache size is approximately 17 MB with the default corpus. Do not commit it to git.

---

## Escalation Rules

Hard rules in `code/escalation.py` override LLM decisions in both directions — the LLM cannot "un-escalate" a ticket that triggered a rule.

### Always-escalate patterns

| Category | Example triggers |
|---|---|
| Account security | "hacked", "compromised", "unauthorized access", "account stolen" |
| Fraud & financial | "fraud", "chargeback", "unauthorized charge", "money stolen", "refund denied" |
| Legal & compliance | "lawsuit", "attorney", "court", "GDPR", "data breach", "suing" |
| Identity & PII | "identity theft", "SSN", "social security number", "bank account number" |
| Card security | "card stolen", "card cloned", "card skimmed", "PIN compromised" |
| Abuse | "threatening", "harassment", "discrimination" |
| Prompt injection | "ignore previous instructions", "jailbreak", "DAN mode", "you are now" |

### No-corpus-match escalation

If the retriever finds no chunks scoring ≥ 0.1 cosine similarity, the ticket is escalated with reason `"No relevant documentation found in corpus"`. The LLM is still called (to generate a polite response), but `_validate_and_fix` overrides the status to `escalated`.

---

## LLM Integration

### Provider: Groq

The agent uses the Groq API with the Llama 3.1 8B Instant model. Groq offers a free tier with generous rate limits sufficient for this evaluation set.

Get your API key: [console.groq.com](https://console.groq.com)

### Upgrading the model

For higher accuracy on ambiguous tickets, change one line in `code/config.py`:

```python
GROQ_MODEL = "llama-3.3-70b-versatile"
```

This increases latency per ticket and may require a paid Groq tier for large batches.

### Rate limiting

The batch runner sleeps 0.3 seconds between tickets. If you encounter HTTP 429 errors from Groq, increase the sleep in `main.py`:

```python
time.sleep(0.3)  # increase to 0.5 or 1.0 if hitting rate limits
```

---

## Module Reference

| File | Responsibility |
|---|---|
| `config.py` | All constants, paths, allowed values. Single source of configuration truth. |
| `corpus_loader.py` | Reads corpus files, chunks them, builds TF-IDF index, manages disk cache. |
| `retriever.py` | Cosine similarity search against TF-IDF matrix, context string formatter. |
| `escalation.py` | Regex-based escalation rules and out-of-scope detection. |
| `prompts.py` | System prompt definition and per-ticket user prompt builder. |
| `agent.py` | Full triage pipeline: retrieval → LLM → parse → validate → return. |
| `main.py` | CLI entry point for batch, sample, and interactive modes. |

---

## Tuning and Configuration

### Getting more `replied` decisions (less aggressive escalation)

Lower the similarity threshold (more corpus matches counted as good):
```python
MIN_SIMILARITY_SCORE = 0.05   # default is 0.1
```

Retrieve more context per ticket:
```python
TOP_K_RESULTS = 8             # default is 5
```

### Getting more `escalated` decisions (more conservative)

Raise the similarity threshold:
```python
MIN_SIMILARITY_SCORE = 0.2
```

Add new hard escalation patterns in `escalation.py`:
```python
ESCALATION_PATTERNS.append(
    (r"\b(your_pattern_here)\b", "your_category_label")
)
```

### Switching to a stronger model

```python
GROQ_MODEL = "llama-3.3-70b-versatile"
```

### Adding a new product corpus

1. Create `data/<product>/` and populate it with `.md`, `.txt`, or `.html` files.
2. Add to `CORPUS_DIRS` in `config.py`:
   ```python
   CORPUS_DIRS = {
       "hackerrank": DATA_DIR / "hackerrank",
       "claude":     DATA_DIR / "claude",
       "visa":       DATA_DIR / "visa",
       "myproduct":  DATA_DIR / "myproduct",   # ← add this
   }
   ```
3. Delete `code/.tfidf_cache.pkl` to force a rebuild.
4. Add the product name to the `SYSTEM_PROMPT` in `prompts.py`.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'groq'`

```bash
pip install -r code/requirements.txt
```

Make sure you activated the virtual environment first.

### `AuthenticationError` or `Invalid API Key`

Check that `code/.env` exists and contains a valid `GROQ_API_KEY`. The key should start with `gsk_`.

### `RuntimeError: No corpus chunks loaded`

The `data/` directories are empty or missing. Ensure `data/hackerrank/`, `data/claude/`, and `data/visa/` exist and contain `.md`, `.txt`, or `.html` files.

### All tickets are being escalated

Most likely cause: the corpus has no relevant content for your tickets, so `has_good_match` is always `False`. Check that `data/` directories contain appropriate support documentation. Run the agent in `--ask` mode on a simple ticket and observe the retrieval scores printed in debug output.

### Groq HTTP 429 (rate limit)

Increase the sleep interval in `main.py`:
```python
time.sleep(1.0)   # was 0.3
```

Or switch to a paid Groq tier.

### `[WARN] Could not parse Groq response`

The LLM returned non-JSON output. This is rare at `temperature=0` but can happen on very unusual tickets. The ticket will be processed with `FALLBACK_RESPONSE` (escalated). Check the raw response printed in the warning to debug.

### Cache is stale after editing corpus files

Delete the cache and rerun:
```bash
rm code/.tfidf_cache.pkl
python main.py
```
