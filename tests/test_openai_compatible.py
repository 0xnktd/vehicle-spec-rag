from typing import Any

import pytest
from pydantic import SecretStr

import vehicle_specs.extraction.extractor as extractor_module


def test_openai_compatible_factory_configures_schema_constrained_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_client: dict[str, Any] = {}
    captured_extractor: dict[str, Any] = {}
    fake_client = object()
    fake_extractor = object()

    def build_client(**kwargs: Any) -> object:
        captured_client.update(kwargs)
        return fake_client

    def build_extractor(*args: Any, **kwargs: Any) -> object:
        captured_extractor["args"] = args
        captured_extractor.update(kwargs)
        return fake_extractor

    monkeypatch.setattr(extractor_module, "ChatOpenAI", build_client)
    monkeypatch.setattr(
        extractor_module,
        "LangChainStructuredExtractor",
        build_extractor,
    )

    result = extractor_module.create_openai_compatible_extractor(
        "local-model",
        base_url="http://vllm:8000/v1",
        api_key=SecretStr("test-key"),
        max_output_tokens=1024,
        request_timeout_seconds=60,
    )

    assert result is fake_extractor
    assert captured_client["model"] == "local-model"
    assert captured_client["base_url"] == "http://vllm:8000/v1"
    assert captured_client["temperature"] == 0
    assert captured_client["max_completion_tokens"] == 1024
    assert captured_extractor["args"] == (fake_client,)
    assert captured_extractor["structured_output_method"] == "json_schema"
    assert captured_extractor["structured_output_strict"] is True
