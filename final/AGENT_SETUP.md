# MateriAl Agent — Setup and Run Guide

MateriAl is a conversational AI assistant for composite micromechanics
characterisation. You describe your measurements in plain English and the agent
runs the correct inverse solvers, checks results for physical plausibility, and
saves everything to a material card — no GUI operation or Python knowledge
required.

This guide covers everything from a fresh machine to a running chat session.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install the Python environment](#2-install-the-python-environment)
3. [Install packages](#3-install-packages)
4. [Set up the local AI model (Ollama)](#4-set-up-the-local-ai-model-ollama)
5. [API key setup (optional — Anthropic cloud models)](#5-api-key-setup-optional)
6. [One-time database setup](#6-one-time-database-setup)
7. [Build the knowledge base (optional)](#7-build-the-knowledge-base-optional)
8. [Run the agent](#8-run-the-agent)
9. [Switch between models](#9-switch-between-models)
10. [How to use the agent](#10-how-to-use-the-agent)
11. [Common errors and fixes](#11-common-errors-and-fixes)

---

## 1. Prerequisites

You will need:

- A computer running **macOS, Windows, or Linux**
- At least **16 GB of RAM** (the local AI model uses ~10 GB)
- **20 GB of free disk space** (model download + environment)
- An internet connection for the one-time download

If you only want to use cloud models (Claude Haiku / Sonnet) and do not want to
run locally, 8 GB of RAM is sufficient and you can skip Section 4.

---

## 2. Install the Python environment

The agent uses the same conda environment as the rest of the application.

If you have not set up the environment yet, follow the instructions in
`../SETUP_AND_RUN.md` for your operating system, then come back here.

The environment name is **`jax_trial`**.

To activate it:

```bash
conda activate jax_trial
```

Your terminal prompt will change to show `(jax_trial)`.

> **Every time you open a new terminal window, run `conda activate jax_trial`
> before any other command.**

---

## 3. Install packages and initialise the database

The fastest way is to run the setup script, which installs packages, creates
the uploads folder, and initialises the database in one step:

```bash
cd /path/to/DiffMicromehanics/final
conda activate jax_trial
bash setup.sh
```

The script will ask before overwriting an existing database and shows a
summary when done.

**Manual alternative** — if you prefer to run steps individually:

```bash
pip install -r requirements.txt
mkdir -p data/uploads
python db/init_db.py
python scripts/test_setup.py
```

---

## 4. Set up the local AI model (Ollama)

The agent runs a small language model **entirely on your machine**. Your
characterisation data never leaves your computer.

> **GPU is optional.** Ollama automatically uses your GPU if one is available
> (Apple Silicon, NVIDIA CUDA, AMD ROCm). If no compatible GPU is detected it
> falls back to CPU — the model still works, but responses will be slower
> (roughly 3-5x). For the 14B Qwen model, expect ~10 tokens/second on CPU vs
> ~40 tokens/second on Apple Silicon M2. If speed is a concern on CPU, use the
> smaller `qwen3:8b` model instead (see Section 9).

### Step 1 — Install Ollama

Download from **https://ollama.com** and install it:

- **Mac:** open the `.dmg` and drag Ollama to Applications, then launch it.
  A small icon appears in the menu bar.
- **Windows:** run the `.exe` installer and follow the prompts.
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

### Step 2 — Download the model

Open a terminal and run:

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

This downloads approximately **9 GB**. It only needs to be done once.
The model is stored in Ollama's local cache and reused on every run.

To confirm the download succeeded:

```bash
ollama list
```

You should see `qwen2.5:14b-instruct-q4_K_M` in the list.

### Step 3 — Make sure Ollama is running

Ollama must be running in the background when you use the agent.

- **Mac:** launch the Ollama app from Applications. The menu bar icon confirms
  it is running.
- **Windows:** Ollama starts automatically after installation.
- **Linux:** run `ollama serve` in a separate terminal window and leave it open.

---

## 5. API key setup (optional)

Skip this section if you are using the local Qwen model (the default).

To use Claude Haiku or Sonnet (Anthropic cloud models), you need an API key.

> **Important:** Cloud models send your inputs to Anthropic's servers.
> Use the local model if your material data is proprietary.

### Step 1 — Get an API key

Sign in at **https://console.anthropic.com**, go to **API Keys**, and create
a new key.

### Step 2 — Create the `.env` file

In the `final/` folder, create a file called `.env`:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Replace `sk-ant-your-key-here` with your actual key.

A template is provided at `final/.env.example` — you can copy and rename it:

```bash
cp .env.example .env
```

Then open `.env` in any text editor and replace the placeholder with your key.

> **The `.env` file is in `.gitignore` and will never be committed to GitHub.**
> Never paste your API key directly into any `.py` file or commit it to git.

---

## 6. One-time database setup

If you have not already done this as part of the main application setup, run:

```bash
cd /path/to/DiffMicromehanics/final
conda activate jax_trial
python db/init_db.py
```

Expected output:

```
Created tables.
Seeded 3 fibers.
Seeded 2 polymers.
Database ready at data/micromechanics.db
```

This only needs to be done once per machine. To reset the database (deletes all
saved cards and results), delete `data/micromechanics.db` and re-run the
command.

---

## 7. Build the knowledge base (optional)

The agent can answer composite mechanics theory questions using a searchable
library of reference PDFs. This step is optional — the agent works without it,
but will not be able to answer detailed theory questions.

### Step 1 — Pull the embedding model

The knowledge base uses a local embedding model. Download it once:

```bash
ollama pull nomic-embed-text
```

### Step 2 — Add your PDFs

Place any reference PDFs (textbooks, papers, datasheets) into:

```
final/agent/knowledge/
```

### Step 3 — Build the index

```bash
cd /path/to/DiffMicromehanics/final
conda activate jax_trial
python agent/ingest.py
```

The script processes each PDF page and builds a searchable vector index in
`agent/knowledge/.chroma/`. Re-running is safe — pages already indexed are
skipped.

You only need to re-run this when you add new PDFs.

---

## 8. Run the agent

Make sure:
- The `jax_trial` environment is active
- Ollama is running (if using the local model)
- You are in the `final/` directory

Then:

```bash
streamlit run app_chat.py
```

A browser tab opens automatically at `http://localhost:8501`.

If the browser does not open automatically, open it manually and go to that
address.

To stop the agent, press `Ctrl+C` in the terminal.

---

## 9. Switch between models

Open `final/app_chat.py` in any text editor. Find this block near the top:

```python
MODELS = {
    "haiku":  ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet": ("anthropic", "claude-sonnet-4-6"),
    "local":  ("ollama",    "qwen2.5:14b-instruct-q4_K_M"),
    "local2": ("ollama",    "qwen3:8b"),
    "local3": ("ollama",    "qwen3:14b"),
}
ACTIVE_MODEL = "local"   # ← change this
```

Change `ACTIVE_MODEL` to one of the keys:

| Value | Model | Requires |
|---|---|---|
| `"local"` | Qwen 2.5 14B (default) | Ollama running locally |
| `"local2"` | Qwen 3 8B (faster, smaller) | Ollama + `ollama pull qwen3:8b` |
| `"local3"` | Qwen 3 14B | Ollama + `ollama pull qwen3:14b` |
| `"haiku"` | Claude Haiku (fast, cheap) | `.env` with Anthropic API key |
| `"sonnet"` | Claude Sonnet (most capable) | `.env` with Anthropic API key |

Save the file and re-run `streamlit run app_chat.py`.

---

## 10. How to use the agent

### Keyword prefixes (optional)

You can start your message with a keyword to help the agent route correctly.
The keyword is optional — the agent infers intent from context — but it makes
responses more reliable, especially for short or ambiguous messages.

| Prefix | Use when |
|---|---|
| `INVERSE` | You have measured properties and want to infer microstructure or constituent properties |
| `PREDICT` | You want to predict properties for a saved card or a given microstructure |
| `SEARCH` | You want to look up material data or ask a theory question |

---

### Browsing your saved cards

The most reliable starting point is to check what you already have:

```
what cards do we have saved?
```

The agent returns a numbered list of all saved cards with fiber, polymer, printer,
and completed stages. Then you can drill into one:

```
tell me about the properties of card 1
what are the predicted elastic properties for card 1
PREDICT: elastic properties for card 4
```

---

### Running a parameter sweep

One of the most useful things the agent can do is answer "what if" questions
by sweeping a parameter and showing how it affects a target property.

```
what mass fraction would I need for E1 to be 18 GPa?
```

The agent runs a coarse sweep (e.g. mf = 0.15, 0.20, 0.25, 0.30) and
interpolates to the target. You can then ask for a finer result:

```
give me a more exact value
```

You can sweep any microstructure parameter — aspect ratio, mass fraction,
fiber alignment — against any composite property.

```
how much should I reduce the aspect ratio to get E1 to be 15 GPa?
how do all properties change at mass fraction 0.258?
```

---

### Comparing predictions to measurements

If a card has experimental measurements saved, you can ask the agent to
compare them directly:

```
PREDICT: elastic properties for card 4
compare these predictions to the experimental values on the database
```

The agent will retrieve the stored measurements and show a side-by-side
comparison of predicted vs measured E1, E2, E3, CTE11, etc.

---

### Running a new characterisation (elastic inverse)

To infer microstructure from measurements, give the agent everything in one
message. The more you provide upfront, the fewer clarifying questions it asks:

```
INVERSE  Fiber: AF, Polymer: AP, Printer: CAMRI
fiber mass fraction = 0.21, aspect ratio = 21
E1 = 15 GPa, E2 = 5 GPa, E3 = 4 GPa
```

> **Tip:** Always include the printer name. If you omit it, the agent will ask
> for it before running the solver, which adds a round-trip.

> **Tip:** If the quality check flags your mass fraction — e.g. "above typical
> range of 0.01 to 0.30" — that is a warning, not necessarily an error. High
> mass fractions (> 0.30) are physically possible in some systems. Review the
> predicted vs target differences and decide whether to proceed.

After the solver runs, the agent asks if you want to save to a card. Say
**yes** and provide a card name.

**Stage 2 — Thermoelastic inverse** (requires Stage 1 saved)

```
INVERSE thermoelastic for card 1.
CTE11 = 10 ppm/K, CTE22 = 55 ppm/K.
```

The agent loads microstructure and matrix modulus from the card automatically.

**Stage 3 — Thermal inverse** (requires Stage 1 saved)

Place your K vs T CSV file in `final/data/uploads/` then:

```
INVERSE thermal for card 1, file is k_data.csv
```

Expected CSV format:

```
temperature_C,K11_WmK,K22_WmK,K33_WmK
25,0.52,0.35,0.35
50,0.55,0.37,0.37
75,0.58,0.39,0.39
```

---

### Asking about materials and theory

```
what fibers do we have?
what polymers are available?
SEARCH: how does flame retardant affect polymer stiffness?
SEARCH: what is the CTE of carbon fiber?
```

The SEARCH prefix causes the agent to check the knowledge base (if you have
built it) as well as the material database.

---

### Units

The agent converts units automatically, but state what you are providing:

| Property | Tell the agent | Agent uses internally |
|---|---|---|
| Stiffness | GPa | MPa (× 1000) |
| CTE | ppm/K | 1/K (× 1e-6) |
| Conductivity | W/m·K | W/m·K (no change) |

---

## 11. Common errors and fixes

### "Connection refused" or agent cannot reach Ollama

Ollama is not running. Start it:
- **Mac:** open the Ollama app from Applications
- **Linux:** run `ollama serve` in a separate terminal

### "model not found" error

The model has not been downloaded. Run:

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

### "ANTHROPIC_API_KEY not found" or authentication error

The `.env` file is missing or the key is wrong. Check that:
- `final/.env` exists (not `.env.example`)
- The key starts with `sk-ant-`
- There are no extra spaces or quotes around the key

### Streamlit opens but the agent does not respond

Check the terminal for errors. Common causes:
- Ollama is not running
- Wrong conda environment (must be `jax_trial`)
- Missing packages — re-run `pip install -r requirements.txt`

### "No module named 'streamlit'" or similar

The environment is not active or packages are missing:

```bash
conda activate jax_trial
pip install -r requirements.txt
```

### The agent says a material is not found

The material name must match exactly what is in the database. Ask:

```
what materials do we have?
```

to see the exact names, then retry.

### Fit error is above 0.10

Your measurements may be inconsistent, or you may not have enough measurements
to constrain all unknowns. Common fixes:
- Add more measurement types (G12, nu12 help a lot for elastic inverse)
- Check units (stiffness must be in GPa, CTE in ppm/K)
- Ask the agent: *"why is my fit error high?"*

### Stage 2 or 3 fails with "Stage 1 not saved"

Stage 1 (elastic inverse) must be saved to a card before running Stage 2 or 3.
Go back, run Stage 1, save it, and note the card ID.
