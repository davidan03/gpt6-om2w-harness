# GPT-6 computer use on Online-Mind2Web

The harness behind our GPT-6 result on the 300 Online-Mind2Web tasks: **152/300 = 50.67 % successes** (GPT-4.1 judge, one run, September 2026). It combines [OpenWebRL](https://github.com/OpenWebRL/OpenWebRL)'s evaluator (tasks, browser environment, judge, scoring) with GPT-6's own agent loop on the OpenAI Responses API.

## How the agent works

`astra_eval.py` replaces OpenWebRL's per-task generation function and then runs the stock evaluator.

- **Model.** `gpt-6-astra`, reasoning effort `medium` with summaries, `max_output_tokens` 8192, one action batch per response (`parallel_tool_calls=False`).
- **Tools.** The native `computer` tool, plus strict functions: `done(response)` and OpenWebRL's `goto_url`, `go_back`, `new_tab`, `switch_tab`, `close_tab`.
- **Executing actions.** `native_actions.py` runs computer actions in Playwright inside each env server (click, double_click, type, keypress, move, scroll, drag, wait). Coordinates are not converted.
- **Prompt.** The `POLICY` string, then `Task: <intent>`, then JSON with the current URL and the open tabs (URL and title).
- **Observations.** Screenshots of the web content only (no browser chrome), 1280x1000 at DPR 1, sent at `detail: original`.
- **History.** Kept on the server via `previous_response_id`, so the model sees every earlier screenshot.
- **Steps and aborts.** A turn that only requests a screenshot is a handshake, not a step. More than 3 in a row aborts the task.
- **Run limits.** 30 steps per task, 4 tasks in parallel, a 9,000 s task timeout, and the step hang guard on.
- **Judge.** Only tasks that end with `done` are judged; all others score 0. GPT-4.1 sees the task, the `done` answer and the last 3 screenshots, and replies SUCCESS or NOT SUCCESS (`reward_browser.py`).

## Setup

You need Linux, Python 3.12 and an OpenAI key with access to `gpt-6-astra` and `gpt-4.1`. No GPU.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium
```

## Run

```bash
export OPENAI_API_KEY=...
export PYTHON=$PWD/.venv/bin/python   # run from the repo root
scripts/run_om2w.sh outputs/smoke --task-indices 0,1,2   # 3 tasks
scripts/run_om2w.sh outputs/run1                         # all 300
```

- **Output directory.** Use a fresh one for every run. Tasks that already have a result file are skipped, so a reused directory "finishes" instantly.
- **Scratch space.** Browser profiles go under `$TMPDIR`. On a cluster, point it at node-local disk.
- **Logs.** Each step prints an `ASTRA_STEP` line. Every API response is saved to `<run>/eval_*/api_traces/<task>.jsonl`.

## Score

```bash
grep -h '"reward": 1.0' outputs/run1/eval_*/results_task_*.jsonl | wc -l   # successes; divide by 300
python openwebrl/compute_eval_success_rate_woaborted_v2.py outputs/run1/eval_*
```

Report successes out of 300 first. The second command gives "v2", which also drops aborted tasks and start pages that block bots from the denominator. Our run scored v2 68.16 % (152 / 223). Always name the judge: o4-mini reads about 1.5x lower than GPT-4.1 on this harness.

`reference/gpt6_om2w_job39936221.jsonl` lists our per-task outcomes so you can compare at task level.

- **Aborts.** Our run had 43 aborted tasks, all scored as failures. 16 of them ran out of the 8,192-token output budget on reasoning before taking any action. `scoring_status` marks these FAILED, which is why v2 counts 27 aborted. Raising the budget would likely add a few successes.
- **Noise.** The same configuration run twice disagrees on about 22 % of tasks. Treat differences under about 5 pp between single runs as noise.

## What to change for experiments

Change one thing at a time, and rerun this baseline on the same day: some start pages block cluster IPs on some days.

- **Model, reasoning and prompt:** `MODEL`, `POLICY`, and the `responses.create(...)` call in `astra_eval.py`.
- **Tools:** `TOOLS` and `NAVIGATION` in `astra_eval.py`. Native action execution is in `native_actions.py`.
- **Observations and history:** `image_url()`, `observation_text()`, and the `payload` built after each step.
- **Harness settings:** the flags in `scripts/run_om2w.sh`.

## Layout

| path | what |
|---|---|
| `astra_eval.py` | agent loop |
| `native_actions.py` | runs the `computer` actions in the env server |
| `openwebrl/` | the parts of OpenWebRL's evaluator this harness runs (from 05f8ed4 plus our harness fixes): task loading, browser env server, judge, scorer |
| `scripts/` | launcher, and the env-server wrapper that installs the native actions |
| `reference/` | per-task outcomes of our run |

License: Apache-2.0, as for OpenWebRL, whose code is included here.
