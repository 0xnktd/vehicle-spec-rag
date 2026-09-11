"""LangChain structured-output adapters and OpenAI-compatible construction."""

from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from vehicle_specs.retrieval import RetrievalHit

from .models import CandidateExtraction, ExtractionDraft
from .prompt import EXTRACTION_PROMPT, PROMPT_VERSION, format_retrieval_context


class StructuredExtractor(Protocol):
    """Extraction boundary used by the application pipeline."""

    model_name: str
    prompt_version: str

    def extract(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
    ) -> CandidateExtraction: ...


class LangChainStructuredExtractor:
    """Invoke a chat model through LangChain's native structured-output contract."""

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        *,
        model_name: str,
        structured_output_method: str | None = "json_schema",
        structured_output_strict: bool | None = None,
        max_attempts: int = 2,
        _chain: Runnable[Any, Any] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")

        self.model_name = model_name.strip()
        self.prompt_version = PROMPT_VERSION

        if _chain is None:
            if chat_model is None:
                raise ValueError("chat_model is required when no chain is supplied")
            kwargs: dict[str, Any] = {}
            if structured_output_method is not None:
                kwargs["method"] = structured_output_method
            if structured_output_strict is not None:
                kwargs["strict"] = structured_output_strict
            structured_model = chat_model.with_structured_output(
                ExtractionDraft,
                **kwargs,
            )
            _chain = EXTRACTION_PROMPT | structured_model
        self._chain = _chain.with_retry(stop_after_attempt=max_attempts)

    def extract(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
    ) -> CandidateExtraction:
        """Return a schema-valid candidate grounded in the supplied chunks."""
        query = query.strip()
        if not hits:
            return CandidateExtraction(status="not_found")

        result = self._chain.invoke(
            {
                "query": query,
                "context": format_retrieval_context(hits),
            }
        )
        if isinstance(result, CandidateExtraction):
            return result
        if isinstance(result, ExtractionDraft):
            draft = result
        elif isinstance(result, dict):
            draft = ExtractionDraft.model_validate(result)
        else:
            raise TypeError(
                "structured model returned an unsupported value: "
                f"{type(result).__name__}"
            )

        result_count = len(draft.results)
        if result_count == 0:
            return CandidateExtraction(status="not_found")
        if result_count == 1:
            return CandidateExtraction(status="found", results=draft.results)
        clarification = draft.clarification or (
            "Which of these specifications do you mean?"
        )
        return CandidateExtraction(
            status="ambiguous",
            results=draft.results,
            clarification=clarification,
        )


def create_openai_compatible_extractor(
    model_name: str,
    *,
    base_url: str,
    api_key: str | SecretStr,
    max_output_tokens: int = 2048,
    request_timeout_seconds: float = 180,
) -> LangChainStructuredExtractor:
    """Create a deterministic extractor backed by an OpenAI-compatible server."""
    secret_key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=secret_key,
        temperature=0,
        seed=42,
        max_completion_tokens=max_output_tokens,
        timeout=request_timeout_seconds,
        max_retries=2,
    )
    return LangChainStructuredExtractor(
        model,
        model_name=model_name,
        structured_output_method="json_schema",
        structured_output_strict=True,
    )
