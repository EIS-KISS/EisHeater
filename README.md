# llmeis

Identify the equivalent circuit model (ECM) of EIS spectra with a Pydantic AI agent.

For each spectrum in a kisstype dataset tar, llmeis renders Nyquist and Bode plots, resamples the
spectrum to 20 log-spaced points, and hands all three forms to a multimodal LLM. The LLM
reasons over several turns, calling tools to fit candidate circuits (`eisfit`) and to
simulate them on the resampled grid (`eisgenerator`), then returns a structured answer.
Each spectrum gets a JSON with the circuit string, a confidence, a short description of
what the spectrum is believed to measure, and the parameters of a final `eisfit`.

## Requirements

- Python >= 3.11
- `eisfit` on your `$PATH`
- `eisgenerator` (the `eisgenerator` python bindings)
- Any OpenAI compatible llm endpoint serving a multimodal model

## Setup

```sh
pip install -e .
```

## Usage

```sh
llmeis <dataset> --model <model-name> --endpoint http://<host>:8080/v1 -o results
```

options (see `--help` for all):

| flag | default | meaning |
| --- | --- | --- |
| `--model` | (required) | model name to request from the server |
| `--endpoint` | `http://localhost:8080/v1` | OpenAI-compatible base URL |
| `-o`, `--outdir` | `results` | output directory |
| `--limit N` | `0` | process only the first N spectra (0 = all) |
| `--max-tokens` | `65536` | max completion tokens per turn (reasoning models need headroom) |
| `--request-timeout` | `3600` | per-request read timeout in seconds |
| `--turns` | off | also write a per-spectrum `<name>.turns.json` with the agent's turn history |
| `--turns-stdout` | off | stream the agent's turns (reasoning, tool calls, returns) to stdout as it runs |

