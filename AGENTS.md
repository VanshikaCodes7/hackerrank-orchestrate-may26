# AGENTS.md — HackerRank Orchestrate Support Triage Agent
> **Single source of truth** for every AI coding agent or developer working in this repository.
> Read this file **in full** before taking any action. Obey it exactly.

---

## Table of Contents

1. [What This Repository Is](#1-what-this-repository-is)
2. [Repository Layout](#2-repository-layout)
3. [Architecture Overview](#3-architecture-overview)
4. [Module-by-Module Reference](#4-module-by-module-reference)
   - 4.1 [config.py](#41-configpy)
   - 4.2 [corpus_loader.py](#42-corpus_loaderpy)
   - 4.3 [retriever.py](#43-retrieverpy)
   - 4.4 [escalation.py](#44-escalationpy)
   - 4.5 [prompts.py](#45-promptspy)
   - 4.6 [agent.py](#46-agentpy)
   - 4.7 [main.py](#47-mainpy)
5. [Data Flow — End to End](#5-data-flow--end-to-end)
6. [Escalation Logic in Detail](#6-escalation-logic-in-detail)
7. [Retrieval System in Detail](#7-retrieval-system-in-detail)
8. [LLM Integration (Groq)](#8-llm-integration-groq)
9. [Configuration Reference](#9-configuration-reference)
10. [Output Schema](#10-output-schema)
11. [Environment Variables](#11-environment-variables)
12. [Caching Behaviour](#12-caching-behaviour)
13. [Constraints and Invariants](#13-constraints-and-invariants)
14. [Extension Points](#14-extension-points)
15. [What Agents Must Never Do](#15-what-agents-must-never-do)

---

## 1. What This Repository Is

This is the implementation of a **terminal-based AI support triage agent** built for the HackerRank Orchestrate hackathon (May 1–2, 2026).

The agent reads rows from `support_tickets/support_tickets.csv`, and for each ticket it:

1. Detects whether the ticket is out-of-scope or contains hard-escalation triggers (rule-based, before any LLM call).
2. Retrieves the most relevant documentation chunks from a local TF-IDF index built over three product corpora (HackerRank, Claude, Visa).
3. Calls the Groq LLM (Llama 3.1 8B by default) with a structured prompt containing the ticket and the retrieved context.
4. Parses and validates the LLM's JSON response.
5. Applies a second layer of rule-based escalation override post-LLM.
6. Writes the final structured decision to `support_tickets/output.csv`.

The agent is **entirely offline for ground-truth answers** — it never calls the live web to answer tickets. All knowledge comes from the files in `data/`.

---

## 2. Repository Layout

```
.
├── AGENTS.md                          ← This file
├── README.md                          ← Human-facing quickstart and setup guide
├── .gitignore
├── .env.example                       ← Template; copy to .env and fill GROQ_API_KEY
├── code/
│   ├── agent.py                       ← Core triage pipeline
│   ├── config.py                      ← All constants, paths, and allowed values
│   ├── corpus_loader.py               ← File reader, chunker, TF-IDF builder, disk cache
│   ├── escalation.py                  ← Rule-based escalation guard (pre- and post-LLM)
│   ├── main.py                        ← CLI entry point (batch, sample, interactive modes)
│   ├── prompts.py                     ← SYSTEM_PROMPT and build_user_prompt()
│   ├── retriever.py                   ← TF-IDF cosine-similarity search + context formatter
│   ├── requirements.txt               ← Python dependencies
│   └── .tfidf_cache.pkl               ← Auto-generated disk cache (do NOT commit)
├── data/
│   ├── hackerrank/                    ← HackerRank help-center articles (.md / .txt / .html)
│   ├── claude/                        ← Claude Help Center export
│   └── visa/                          ← Visa consumer and small-business support docs
└── support_tickets/
    ├── sample_support_tickets.csv     ← Dev/test tickets with expected signals
    ├── support_tickets.csv            ← Evaluation input (inputs only — no labels)
    └── output.csv                     ← Agent predictions written here
```

**Never commit:** `.env`, `.tfidf_cache.pkl`, `venv/`, `__pycache__/`, `output.csv`.

---

## 3. Architecture Overview

```
support_tickets.csv
        │
        ▼
   main.py  (CLI / batch loop)
        │
        ├─── initialize_corpus()  [corpus_loader.py]
        │         │
        │         ├─ Reads data/{hackerrank,claude,visa}/**
        │         ├─ Chunks text (500 chars, 100 overlap)
        │         ├─ Fits TF-IDF vectorizer (sklearn, bigrams, 20 k features)
        │         └─ Caches to .tfidf_cache.pkl (fingerprint-checked)
        │
        └─── triage()  [agent.py]  ← called once per ticket
                  │
                  ├─ 1. check_out_of_scope()    [escalation.py]
                  │       └─ regex patterns: weather, recipes, gibberish → early return
                  │
                  ├─ 2. retrieve()              [retriever.py]
                  │       └─ TF-IDF cosine similarity → top-5 chunks
                  │           (company-filtered first, falls back to all-corpus)
                  │
                  ├─ 3. build_user_prompt()     [prompts.py]
                  │       └─ Formats ticket + retrieved context as markdown
                  │
                  ├─ 4. _call_groq()            [agent.py → Groq API]
                  │       └─ llama-3.1-8b-instant, temp=0, max_tokens=1024
                  │
                  ├─ 5. _parse_json_response()  [agent.py]
                  │       └─ tries raw JSON → fenced block → bare object
                  │
                  └─ 6. _validate_and_fix()     [agent.py + escalation.py]
                          ├─ evaluate_escalation(): hard rules + no-match rule
                          ├─ Ensures "human" appears in escalated responses
                          ├─ Validates status and request_type against allowlists
                          └─ Fills missing fields from FALLBACK_RESPONSE
```

---

## 4. Module-by-Module Reference

### 4.1 `config.py`

Central configuration. **All tuneable constants live here — nowhere else.**

| Symbol | Type | Value | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | str | from `.env` | Groq authentication |
| `GROQ_MODEL` | str | `"llama-3.1-8b-instant"` | Swap to `"llama-3.3-70b-versatile"` for higher quality |
| `ROOT_DIR` | Path | repo root | Base path anchor |
| `DATA_DIR` | Path | `ROOT_DIR/data` | Root of all corpus files |
| `SUPPORT_ISSUES_DIR` | Path | `ROOT_DIR/support_tickets` | Input/output CSV location |
| `INPUT_CSV` | Path | `.../support_tickets.csv` | Default evaluation input |
| `SAMPLE_CSV` | Path | `.../sample_support_tickets.csv` | Dev/test input |
| `OUTPUT_CSV` | Path | `.../output.csv` | Default prediction output |
| `CORPUS_DIRS` | dict | `{hackerrank, claude, visa}` | Maps company name → corpus directory |
| `TOP_K_RESULTS` | int | `5` | Max chunks returned by retriever |
| `MIN_SIMILARITY_SCORE` | float | `0.1` | TF-IDF cosine threshold for "good match" |
| `CHUNK_SIZE` | int | `500` | Max characters per text chunk |
| `CHUNK_OVERLAP` | int | `100` | Overlap characters between consecutive chunks |
| `CHROMA_COLLECTION_NAME` | str | `"support_corpus"` | Unused legacy constant (ChromaDB was removed) |
| `ALLOWED_STATUS` | set | `{"replied", "escalated"}` | Valid `status` output values |
| `ALLOWED_REQUEST_TYPES` | set | `{"product_issue", "feature_request", "bug", "invalid"}` | Valid `request_type` values |

**Rules for agents editing this file:**
- Never hardcode paths in other modules — always import from `config`.
- Do not add secrets here; use `.env` only.
- If you change `ALLOWED_STATUS` or `ALLOWED_REQUEST_TYPES`, update `prompts.py` and `escalation.py` to match.

---

### 4.2 `corpus_loader.py`

Responsible for loading, chunking, indexing, and caching all support documentation.

#### Public API

```python
initialize_corpus() -> Tuple[None, TfidfVectorizer, scipy.sparse.csr_matrix, List[Dict]]
```

Returns `(None, vectorizer, matrix, chunks)`. The first element is always `None` (ChromaDB collection placeholder kept for API compatibility — do not remove it).

#### Internal pipeline

**Step 1 — Fingerprinting (`_corpus_fingerprint`)**
Computes an MD5 hash over every file's path, size, and mtime across all `CORPUS_DIRS`. Used to detect corpus changes and decide whether to rebuild the TF-IDF index.

**Step 2 — File reading (`_read_file`)**
- Reads `.md`, `.txt`, `.html`, `.htm`, `.csv` files.
- Strips HTML tags with a regex (`<[^>]+>`).
- Collapses whitespace.
- Silently skips unreadable files.

**Step 3 — Chunking (`_chunk_text`)**
- Splits on sentence boundaries (`(?<=[.!?])\s+`).
- Accumulates sentences into chunks up to `CHUNK_SIZE` characters.
- Carries over the last `CHUNK_OVERLAP` characters into the next chunk to preserve context across boundaries.
- Discards chunks shorter than 50 characters.

**Step 4 — Chunk metadata**
Each chunk dict contains:
```python
{
    "id":          str,   # MD5 of "company::filepath::index"
    "text":        str,   # chunk content
    "company":    str,   # "hackerrank" | "claude" | "visa"
    "source_file": str,  # filename only (not full path)
}
```

**Step 5 — TF-IDF index (`build_tfidf_index`)**
```python
TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),   # unigrams and bigrams
    min_df=1,
    sublinear_tf=True,    # log(1 + tf) — reduces impact of very common terms
)
```

**Step 6 — Disk cache**
- Cache path: `code/.tfidf_cache.pkl`
- Contains: `{"fingerprint": str, "vectorizer": ..., "matrix": ..., "chunks": [...]}`
- On load: fingerprint is compared; mismatch triggers full rebuild.
- Cache write failures are non-fatal (logged as warnings, not raised).

---

### 4.3 `retriever.py`

Performs TF-IDF cosine similarity search over the indexed corpus.

#### Public API

```python
retrieve(
    query: str,
    collection,          # always None — kept for signature compatibility
    vectorizer,
    matrix,
    chunks: List[Dict],
    company: Optional[str] = None,
    top_k: int = TOP_K_RESULTS,
) -> Dict
```

**Returns:**
```python
{
    "hits":          List[Dict],  # top-k result chunks with scores
    "best_score":    float,       # highest cosine similarity in the result set
    "has_good_match": bool,       # True if best_score >= MIN_SIMILARITY_SCORE
    "company_used":  str,         # "hackerrank"|"claude"|"visa"|"all"
}
```

Each hit dict:
```python
{
    "text":        str,
    "company":    str,
    "source_file": str,
    "similarity":  float,   # rounded to 4 decimal places
}
```

#### Two-pass retrieval strategy

1. **First pass:** Filter chunks to only those matching `company` (case-insensitive). Compute cosine similarity over filtered indices. Take top-k.
2. **Fallback:** If `best_score < MIN_SIMILARITY_SCORE` and a company filter was applied, repeat the search over the **entire corpus** (all three companies). This prevents a good cross-domain answer from being suppressed by an incorrect company label on a ticket.

#### Context formatting (`format_context`)

Assembles retrieved chunks into a single string:
- Each entry prefixed with `[Doc N | Source: <filename> | Score: <score>]`.
- Hard truncation at `MAX_CONTEXT_CHARS = 3000` to stay within Groq free-tier token limits.
- Chunks that would exceed the budget are either partially included (if ≥200 chars remain) or skipped entirely.

---

### 4.4 `escalation.py`

Pure rule-based escalation guard. **These rules override all LLM decisions.** They cannot be soft-configured — they are hard-coded for safety.

#### `ESCALATION_PATTERNS`

A list of `(regex_pattern, category_label)` tuples evaluated against the lowercased concatenation of `subject + " " + issue`. If any pattern matches, the ticket is force-escalated regardless of what the LLM returns.

| Category | What triggers it |
|---|---|
| `account_security` | "hacked", "compromised", "unauthorized access", "account stolen", "change password urgently" |
| `fraud_financial` | "fraud", "fraudulent", "chargeback", "unauthorized charge/transaction/payment", "money stolen/missing/charged", "refund not received/denied" |
| `legal_compliance` | "lawsuit", "legal action", "attorney", "lawyer", "court", "GDPR", "data breach/leak/stolen", "sue", "suing", "litigation", "right to erasure" |
| `identity_pii` | "identity theft", "personal data exposed/leaked/stolen", "SSN", "social security", "passport number", "bank account number" |
| `card_security` | "card stolen/lost/missing/cloned/skimmed", "PIN compromised/stolen/exposed" |
| `abuse` | "threatening", "harassment", "abuse", "discrimination" |
| `prompt_injection` | "ignore previous instructions", "you are now", "pretend you are", "act as a [bot]", "system prompt", "jailbreak", "DAN mode" |

#### `OUT_OF_SCOPE_PATTERNS`

Checked by `check_out_of_scope()` **before** any retrieval or LLM call. If matched, an early structured response is returned immediately.

Patterns cover: weather, recipes, sports, stocks, lottery, horoscopes, "write me a poem/story/essay/code", general knowledge questions.

#### `evaluate_escalation(issue, subject, has_good_match, llm_status) -> (str, str)`

Final decision combiner, called **after** the LLM responds:

1. Run `check_escalation()` — if hard rule fires → `"escalated"`.
2. Run `check_no_corpus_match()` — if `has_good_match is False` → `"escalated"`.
3. Otherwise → trust `llm_status`.

Returns `(final_status, escalation_reason)`.

---

### 4.5 `prompts.py`

Contains all prompt text used with the LLM. **Do not embed prompt logic in other modules.**

#### `SYSTEM_PROMPT`

Instructs the model to act as a professional support triage agent for three products (HackerRank, Claude, Visa). Key rules enforced in the prompt:

- **Corpus-only**: respond only from the provided documentation excerpts.
- **Escalate when unsure**: billing, security, fraud, legal, PII → always escalate.
- **No hallucination**: if the answer is not in the docs, say so and escalate.
- **Invalid tickets**: prompt injection, gibberish → `request_type: "invalid"`, `status: "escalated"`.
- **Multi-issue tickets**: address primary issue; note others in `justification`.

The system prompt specifies the exact JSON output schema the model must follow.

#### `build_user_prompt(issue, subject, company, context) -> str`

Formats a single ticket into the user turn of the conversation:

```
## Support Ticket
Company: <company or "Unknown / Cross-domain">
Subject: <subject or "(no subject)">
Issue:
<issue text>

---

## Relevant Documentation
<context from format_context()>

---
Analyze this ticket and respond with the JSON triage decision.
```

---

### 4.6 `agent.py`

Orchestrates the complete triage pipeline for a single ticket. This is the core module.

#### `triage(issue, subject, company, collection, vectorizer, matrix, chunks) -> Dict`

The main entry point called by `main.py` for each row.

**Full execution sequence:**

```
1. Strip whitespace from all inputs.
2. check_out_of_scope(issue, subject)
   → if True: return early with status="escalated", product_area="Out of Scope"

3. retrieve(...) → retrieval dict
   → extracts context string and has_good_match flag

4. build_user_prompt(...) → user_prompt string

5. _call_groq(SYSTEM_PROMPT, user_prompt) → raw_response string
   → temperature=0, max_tokens=1024
   → on exception: result = FALLBACK_RESPONSE

6. _parse_json_response(raw_response) → result dict or None
   → tries: json.loads(raw) → fenced code block → bare {...} regex
   → on failure: result = FALLBACK_RESPONSE

7. _validate_and_fix(result, llm_status, issue, subject, has_good_match)
   → runs evaluate_escalation()
   → appends "[Auto-escalated: <reason>]" to justification if overridden
   → ensures escalated responses contain "human"
   → validates status and request_type against allowlists
   → fills missing/empty fields from FALLBACK_RESPONSE

8. return result
```

#### `FALLBACK_RESPONSE`

Used whenever the LLM call fails or the response cannot be parsed:

```python
{
    "status": "escalated",
    "product_area": "Unknown",
    "response": "We've received your request and a human support agent will review it shortly.",
    "justification": "Agent encountered an error processing this ticket; escalated for safety.",
    "request_type": "product_issue",
}
```

#### `_call_groq(system, user) -> str`

Wraps the Groq SDK call. Returns `response.choices[0].message.content.strip()`. Raises on API errors (caught by `triage()`).

#### `_parse_json_response(raw) -> Optional[Dict]`

Three-stage JSON extraction:
1. `json.loads(raw)` — clean JSON.
2. Regex for ` ```json { ... } ``` ` fenced blocks.
3. Regex for bare `{ ... }` anywhere in the response.

Returns `None` if all three stages fail.

#### `_validate_and_fix(result, llm_status, issue, subject, has_good_match) -> Dict`

Post-LLM sanitization:
- Calls `evaluate_escalation()` to apply hard rules.
- If escalation reason differs from LLM decision, appends audit note to `justification`.
- If `status == "escalated"` and response does not contain "human", appends the human-handoff sentence.
- Replaces invalid `status` values with `"escalated"`.
- Replaces invalid `request_type` values with `"product_issue"`.
- Fills any empty/missing required fields from `FALLBACK_RESPONSE`.

---

### 4.7 `main.py`

CLI entry point. Three modes of operation:

#### Batch mode (default)
```bash
python main.py
python main.py --input path/to/custom.csv
python main.py --output path/to/custom_output.csv
```
- Reads `INPUT_CSV` (or custom path).
- Processes all rows with a `tqdm` progress bar.
- Adds a `0.3 s` sleep between rows to respect Groq rate limits.
- Writes merged results (original columns + output columns) to `OUTPUT_CSV`.
- Prints summary: total / replied / escalated / errors.

#### Sample mode
```bash
python main.py --sample
```
Uses `SAMPLE_CSV` instead of `INPUT_CSV`. Identical pipeline otherwise.

#### Interactive mode
```bash
python main.py --ask
```
- REPL loop: prompts for `company`, `subject`, `issue`.
- Calls `triage()` directly and pretty-prints the result to stdout.
- Type `exit` or `quit` at any prompt to stop.

#### Output CSV structure
The output CSV contains all original input columns plus these appended columns (not duplicated if already present):
- `status`
- `product_area`
- `response`
- `justification`
- `request_type`

---

## 5. Data Flow — End to End

```
CSV Row
  │
  │  issue="My HackerRank test won't load"
  │  subject="Assessment not working"
  │  company="hackerrank"
  │
  ▼
check_out_of_scope()
  → no match → continue
  │
  ▼
retrieve(query="Assessment not working My HackerRank test won't load", company="hackerrank")
  → First pass: filter to hackerrank chunks
  → TF-IDF cosine similarity computed
  → Top 5 hits returned, best_score = 0.34 → has_good_match = True
  │
  ▼
build_user_prompt()
  → Markdown string with ticket + 3000-char context window
  │
  ▼
_call_groq(SYSTEM_PROMPT, user_prompt)
  → Groq API → llama-3.1-8b-instant → raw JSON string
  │
  ▼
_parse_json_response()
  → {status: "replied", product_area: "Assessment Platform", ...}
  │
  ▼
evaluate_escalation()
  → No hard rule match
  → has_good_match = True → no no-match escalation
  → Trust LLM: status = "replied"
  │
  ▼
_validate_and_fix()
  → status ∈ ALLOWED_STATUS ✓
  → request_type ∈ ALLOWED_REQUEST_TYPES ✓
  → All fields populated ✓
  │
  ▼
Output Row
  status="replied"
  product_area="Assessment Platform"
  response="Based on our documentation, to resolve..."
  justification="Ticket clearly describes a loading issue..."
  request_type="product_issue"
```

---

## 6. Escalation Logic in Detail

Escalation is applied in **two separate phases**:

### Phase 1 — Pre-retrieval (check_out_of_scope)
Triggered before any expensive computation. Returns immediately if the ticket has no relation to any of the three supported products.

### Phase 2 — Post-LLM (evaluate_escalation)
Runs after the LLM responds. Three sub-checks in order:

```
check_escalation()       ← hard keyword patterns
        │
        ├── match found → ESCALATED (reason = pattern category)
        │
        └── no match → check_no_corpus_match()
                                │
                                ├── has_good_match=False → ESCALATED (reason = "no_corpus_match")
                                │
                                └── has_good_match=True → trust LLM decision
```

**The LLM can never un-escalate a ticket that triggered a hard rule or had no corpus match.**

### Justification audit trail
When an auto-escalation override occurs, the original LLM justification is preserved and the override reason is appended:
```
"Documentation clearly explains the process. [Auto-escalated: Hard rule triggered: fraud_financial]"
```

---

## 7. Retrieval System in Detail

### Why TF-IDF (not dense embeddings)?

- No GPU required, no ONNX/PyTorch install issues.
- Fast to build and search on any machine.
- Bigrams (`ngram_range=(1,2)`) capture common tech phrases like "test case", "account balance", "two factor".
- `sublinear_tf=True` prevents very frequent terms from dominating.

### Similarity threshold

`MIN_SIMILARITY_SCORE = 0.1`. TF-IDF cosine scores are naturally lower than dense embedding cosines (no vector normalisation to a semantic space). 0.1 was chosen as a conservative floor — scores below this reliably indicate the corpus has nothing relevant.

### Company filtering

The retriever first searches only within the specified company's chunks. This avoids Visa documentation appearing in HackerRank answers. The fallback to all-corpus search exists because:
- Some tickets have wrong or missing company labels.
- Some issues span products (e.g., a Visa question submitted via a HackerRank integration form).

### Context budget

`MAX_CONTEXT_CHARS = 3000`. Groq's free tier has token limits. 3000 characters is approximately 750–900 tokens of context, leaving ample room for the system prompt and the LLM's JSON response within a 4096-token window.

---

## 8. LLM Integration (Groq)

### Model

Default: `llama-3.1-8b-instant` — fast, low-latency, free-tier friendly.

Upgrade path: change `GROQ_MODEL` in `config.py` to `"llama-3.3-70b-versatile"` for significantly higher accuracy on ambiguous tickets.

### Parameters

```python
temperature=0     # fully deterministic
max_tokens=1024   # enough for the JSON response with long justifications
```

### Rate limiting

`main.py` sleeps `0.3 s` between tickets. For large batches, increase this if Groq returns 429 errors.

### Failure handling

Any exception from the Groq SDK is caught in `triage()`. The ticket is processed with `FALLBACK_RESPONSE` (status=escalated) so the batch never halts on a single API error.

---

## 9. Configuration Reference

To change any behaviour, edit **only `config.py`**. Do not scatter magic numbers across modules.

| What you want to change | Symbol to edit |
|---|---|
| Switch LLM model | `GROQ_MODEL` |
| More/fewer retrieved chunks | `TOP_K_RESULTS` |
| Similarity floor | `MIN_SIMILARITY_SCORE` |
| Chunk granularity | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Input CSV location | `INPUT_CSV` |
| Output CSV location | `OUTPUT_CSV` |
| Add a new product corpus | Add entry to `CORPUS_DIRS` |
| Add new valid status values | `ALLOWED_STATUS` (+ update prompts + escalation) |
| Add new request types | `ALLOWED_REQUEST_TYPES` (+ update prompts) |

---

## 10. Output Schema

Every row in `output.csv` must have these five columns populated:

| Column | Allowed values | Notes |
|---|---|---|
| `status` | `replied`, `escalated` | `replied` only if docs fully answer the issue |
| `product_area` | Any string | e.g. "Account Management", "Billing & Payments", "Card Services", "Assessment Platform", "Fraud & Security", "Technical Issue" |
| `response` | Any string | User-facing text. If escalated, must contain "human" |
| `justification` | Any string | 1–3 sentences. References docs or escalation rule |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` | `invalid` for out-of-scope or malicious tickets |

---

## 11. Environment Variables

Required in `code/.env` (copy from `.env.example`):

```
GROQ_API_KEY=gsk_...
```

- Loaded by `config.py` using `python-dotenv`.
- Never hardcode in any source file.
- Never log. If a user pastes an API key into a chat/prompt, write `[REDACTED]` in any log output.

---

## 12. Caching Behaviour

The TF-IDF index is cached to `code/.tfidf_cache.pkl`.

| Condition | Behaviour |
|---|---|
| Cache file absent | Full rebuild |
| Cache fingerprint matches corpus | Load from cache (instant) |
| Cache fingerprint differs | Full rebuild, overwrite cache |
| Cache file corrupted | Full rebuild, overwrite cache |
| Cache write fails | Non-fatal warning; continues without cache |

The cache stores the fitted `TfidfVectorizer`, the sparse `csr_matrix`, and the full `chunks` list. Rebuild time is proportional to total corpus size — typically a few seconds.

**Do not commit `.tfidf_cache.pkl`** — it is platform-specific (Python version, sklearn version) and large (~17 MB based on the current corpus).

---

## 13. Constraints and Invariants

These must hold at all times. Any agent editing code must preserve them:

1. **`triage()` never raises.** All exceptions are caught internally; the worst outcome is returning `FALLBACK_RESPONSE` (escalated).
2. **`status` is always in `ALLOWED_STATUS`.** `_validate_and_fix` enforces this.
3. **`request_type` is always in `ALLOWED_REQUEST_TYPES`.** `_validate_and_fix` enforces this.
4. **Escalated responses always mention "human".** `_validate_and_fix` appends the human-handoff sentence if absent.
5. **Hard escalation rules cannot be disabled by the LLM.** `evaluate_escalation` runs after the LLM.
6. **The corpus is the only ground-truth source.** The LLM prompt explicitly forbids inventing policies.
7. **Output CSV always includes all original input columns** plus the five output columns.
8. **`collection` parameter is always `None`** in the current implementation (ChromaDB was removed; the parameter is kept for API stability).
9. **`GROQ_API_KEY` is never logged or printed.**

---

## 14. Extension Points

If you want to improve the agent, here are the intended extension surfaces:

### Upgrade the retriever
Replace TF-IDF with dense embeddings (e.g., `sentence-transformers`). Keep the same `retrieve()` return signature so `agent.py` does not need changes.

### Upgrade the LLM
Change `GROQ_MODEL` in `config.py`. No other changes needed.

### Add a new product corpus
1. Create `data/<product_name>/` with support docs.
2. Add `"<product_name>": DATA_DIR / "<product_name>"` to `CORPUS_DIRS` in `config.py`.
3. Update `SYSTEM_PROMPT` in `prompts.py` to mention the new product.
4. Delete `.tfidf_cache.pkl` to force a rebuild.

### Add new escalation rules
Add a tuple to `ESCALATION_PATTERNS` in `escalation.py`. Pattern is a Python regex string; category is a short label string.

### Add a new output field
1. Add to `OUTPUT_FIELDS` in `main.py`.
2. Add to `FALLBACK_RESPONSE` in `agent.py`.
3. Add to the JSON schema in `SYSTEM_PROMPT` in `prompts.py`.
4. Add the fill-missing-fields loop in `_validate_and_fix` in `agent.py`.

---

## 15. What Agents Must Never Do

- **Do not remove the `collection` parameter** from `triage()` or `retrieve()` — it is `None` but maintained for API stability.
- **Do not move configuration constants** out of `config.py` into other modules.
- **Do not bypass `_validate_and_fix()`** — the output schema guarantee depends on it.
- **Do not add live web calls** to answer ticket questions — the agent is designed to be offline for ground-truth.
- **Do not commit `.env`, `.tfidf_cache.pkl`, `output.csv`, or `venv/`.**
- **Do not log `GROQ_API_KEY`** or any other secret anywhere.
- **Do not change the `OUTPUT_FIELDS` list order** without updating the CSV writer in `main.py`.
- **Do not raise exceptions from `triage()`** — wrap new code in try/except and fall back gracefully.
