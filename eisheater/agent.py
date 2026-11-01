from __future__ import annotations

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers._openai_compatible import OpenAICompatibleProvider
from pydantic_ai._http import create_async_httpx2_client
from pydantic_ai._http import httpx2

from .modelstr import KISS_ENCODING_DOC
from .tools import SpectrumDeps, register_tools

DEFAULT_REQUEST_TIMEOUT = 3600.0


class LlamaProvider(OpenAICompatibleProvider):
    @property
    def name(self) -> str:
        return "llama-server"

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # Pick the profile by model-name prefix, the same way pydantic-ai's Ollama provider does,
        # so reasoning models get their thinking field and JSON-schema handling right.
        prefix_to_profile = {
            "llama": meta_model_profile,
            "gemma": google_model_profile,
            "qwen": qwen_model_profile,
            "qwq": qwen_model_profile,
            "deepseek": deepseek_model_profile,
            "mistral": mistral_model_profile,
            "command": cohere_model_profile,
            "gpt-oss": harmony_model_profile,
        }
        lowered = model_name.lower()
        profile = None
        for prefix, func in prefix_to_profile.items():
            if lowered.startswith(prefix):
                profile = func(model_name)
                break
        base = OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)
        if profile is None:
            return base
        return merge_profile(base, profile)

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        http_client=None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        if http_client is None:
            http_client = create_async_httpx2_client(timeout=httpx2.Timeout(timeout, connect=15))
        self._client = self._create_openai_client(
            base_url=base_url, api_key=api_key or "llama", http_client=http_client
        )


class EcmResult(BaseModel):
    """The agent's structured answer for one spectrum."""

    model_str: str = Field(
        description=(
            "The KISS-encoded equivalent circuit WITH concrete parameter values for every element, "
            "e.g. 'r{50}-p{1e-4, 0.8}'. Single numbers, not ranges."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the model, 0.0 to 1.0.")
    type: str = Field(
        description=(
            "A short description of what this spectrum is believed to be a measurement of, "
            "e.g. 'Li-ion NMC battery cell' or 'corroding steel electrode'."
        )
    )


SYSTEM_PROMPT = f"""\
You are an expert electrochemist analysing electrochemical impedance spectroscopy (EIS) data.

TASK
Identify the equivalent circuit model (ECM) that best describes the impedance spectrum you are
given. You are shown the spectrum in three forms:
1. The full measured data: omega (angular frequency, rad/s) and Re(Z), Im(Z) in ohm.
2. A resampled version with 20 log-spaced points across the same frequency range.
3. Two plots: a Nyquist plot (Re(Z) vs -Im(Z)) and a Bode plot (|Z| and phase vs frequency).

MODEL STRING FORMAT (KISS encoding)
{KISS_ENCODING_DOC}

TOOLS
You have two tools:
- fit_spectrum(model_str): fits the circuit to the ORIGINAL full-resolution spectrum with a
  nonlinear optimizer (eisfit) and returns the fit report: the fitted parameters, the fit quality
  (residuals, chi^2), and a verdict on whether the circuit actually fits. Use it to check how well
  a candidate fits and to read off good parameter values.
- simulate_spectrum(model_str): generates a spectrum from the circuit on the 20-point grid and
  returns the values, as well as residuals between the simulation and the real spectrum.
  Use it to compare a circuit's predicted curve against the measured one.

WORKFLOW
1. Study the Nyquist and Bode plots and the data. Count the arcs/semicircles, look for a
   low-frequency 45-degree Warburg tail, a high-frequency inductive loop, and note the magnitude
   scale of each feature.
2. Consider the possiblity of overlapping features.
3. Propose a candidate circuit in KISS encoding with concrete parameter estimates.
4. Call fit_spectrum to check the fit; read the verdict and the fitted parameters.
5. Call simulate_spectrum to compare the generated curve with the measured one.
6. Iterate: adjust the topology and parameters until the fit is good and the curves agree.
7. When confident, produce the final answer.

OUTPUT
Return the structured output with:
- model_str: the KISS-encoded circuit WITH concrete parameter values (single numbers, not ranges),
  e.g. "r{50}-p{1e-4, 0.8}". These values are used as starting points for a final fit, so make
  them physically sensible estimates consistent with the data.
- confidence: your confidence in the model, a number between 0.0 and 1.0.
- type: a short description of what you believe this spectrum is a measurement of.

CONSTRAINTS
- The final model_str must be parseable by eisgenerator and must carry a concrete value for every
  element.
- Prefer the simplest circuit that fits the data well; do not add elements that do not improve the
  fit but don't oversimplify either.
"""


def make_model(
    base_url: str,
    model_name: str,
    api_key: str | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name, provider=LlamaProvider(base_url, api_key=api_key, timeout=timeout)
    )


def make_agent(model: OpenAIChatModel) -> Agent:
    agent = Agent(
        model,
        output_type=EcmResult,
        deps_type=SpectrumDeps,
        instructions=SYSTEM_PROMPT,
    )
    register_tools(agent)
    return agent
