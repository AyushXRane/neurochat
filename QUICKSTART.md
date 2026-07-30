# Quickstart

Five minutes, no neuroimaging background assumed. If a term is unfamiliar,
[GLOSSARY.md](GLOSSARY.md) explains all of them.

---

## 1. Install and start it

```bash
pip install .
neurochat demo
```

Opens `http://127.0.0.1:8000` with a real brain already loaded.

No internet? `neurochat demo --offline` uses bundled synthetic data instead.

---

## 2. What you're looking at

Four panels:

| Panel | What it's for |
|---|---|
| **Left** | Your scans, and the list of brain regions |
| **Middle** | The brain. Click anywhere to move the crosshair. |
| **Right** | Chat (only if you set an API key) |
| **Bottom** | What you've done, and the Python code it wrote for you |

The bottom-right panel is the point of the whole tool. Every action you take
writes real code there.

---

## 3. Your first measurement

**a. Get some scans.** Top-left, click **MRI cohort** (12 real people) or
**PET cohort** (8 real people). First click downloads; after that it's instant.

**b. Load region names.** In the **Atlas** dropdown, pick `harvard-oxford-sub`.
That gives you 21 real brain structures.

**c. Find a region.** Type `hippocampus` in the filter box. Click **Left
Hippocampus**. The crosshair jumps there.

**d. Measure everyone.** Click the button that now says *"Left Hippocampus
across 12 scans"*. You get a table — one row per person, with the average value
in that region.

**e. Take the code.** Click **Export .py** (bottom right). That file reproduces
your numbers on any computer with Python.

That's the whole loop. Everything else is variations on it.

---

## 4. Using your own scans

Put your `.nii.gz` files in a folder. Type the folder path in the **Library**
box, hit **Scan**. Click any scan to view it.

Your files need to be **already preprocessed** — this tool measures, it doesn't
clean up data. If your scans came straight off a scanner, run them through
fMRIPrep first.

---

## 5. Talking to it instead of clicking

Optional. Set an API key before starting:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
neurochat demo
```

Then type things like:

> what's the average value in the left hippocampus?
>
> show me the amygdala
>
> compare these two scans

Everything you can type, you can also click. The clicking is free and instant;
the chat costs API calls.

To use it inside Claude Desktop instead, see the MCP section in the
[README](README.md#use-it-from-claude-desktop-or-claude-code).

---

## 6. When it says no

**"Region names cannot be resolved"** — your scan's file header doesn't say
which coordinate system it's in, so the tool won't guess where the hippocampus
is. If you know the scan was normalised, click the **Treat as…** button that
appears. That records the decision as yours.

**"Did you mean…"** — you typed a region name that doesn't exist. It won't pick
one for you. Click one of the suggestions.

**It refuses to run a t-test** — on purpose. This tool measures; it doesn't do
statistics. Export the table and use R, or `nilearn.glm`.

---

## What it will never do

No statistics. No preprocessing. No medical or diagnostic conclusions. It won't
run code that an AI wrote.

It gives you honest numbers and the code behind them. What they mean is your
call. See [LIMITATIONS.md](LIMITATIONS.md) before trusting output for anything
that matters.
