"""Schema-constrained extraction and deterministic evidence validation."""

from .evidence import assemble_table_evidence
from .extractor import (
    LangChainStructuredExtractor,
    StructuredExtractor,
    create_openai_compatible_extractor,
)
from .models import (
    AlternateValue,
    CandidateExtraction,
    ExtractionResponse,
    ExtractionStatus,
    SourceEvidence,
    SpecificationResult,
    SpecType,
)
from .normalization import normalize_candidate, normalize_table_fields
from .prompt import PROMPT_VERSION, format_retrieval_context
from .table_completion import (
    complete_capacity_table_results,
    complete_dimension_table_results,
    complete_torque_table_results,
)
from .validation import (
    EvidenceValidationError,
    ValidationCode,
    ValidationIssue,
    validate_candidate,
)

__all__ = [
    "PROMPT_VERSION",
    "AlternateValue",
    "CandidateExtraction",
    "EvidenceValidationError",
    "ExtractionResponse",
    "ExtractionStatus",
    "LangChainStructuredExtractor",
    "SourceEvidence",
    "SpecType",
    "SpecificationResult",
    "StructuredExtractor",
    "ValidationCode",
    "ValidationIssue",
    "assemble_table_evidence",
    "complete_capacity_table_results",
    "complete_dimension_table_results",
    "complete_torque_table_results",
    "create_openai_compatible_extractor",
    "format_retrieval_context",
    "normalize_candidate",
    "normalize_table_fields",
    "validate_candidate",
]
