"""llmeis - identify the equivalent circuit model of EIS spectra with an LLM agent.

A Pydantic AI agent loop that, for each spectrum in a dataset tar, renders Nyquist and Bode
plots, resamples the spectrum, and lets a multimodal LLM (served by llama-server over the OpenAI
API) identify the circuit, using eisfit and eisgenerator as tools. The result is a JSON per
spectrum with the model's circuit string, confidence, and a description, enriched with the
parameters of a final eisfit.
"""

__version__ = "0.1.0"
