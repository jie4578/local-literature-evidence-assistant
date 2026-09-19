from __future__ import annotations

import json

import pytest

from grounded_qa import build_evidence_pack
from providers import ProviderResponse
from retrieval.models import HybridSearchResponse, RetrievalResult
from tests.fixtures.grounded_qa_provider_outputs import (
    COMPATIBILITY_EXPECTED,
    COMPATIBILITY_MATRIX,
    REPLAY_FIXTURES,
)
from scripts.validate_grounded_qa_live import (
    FULL_DOCUMENT_SENTINEL,
    LIVE_MAX_OUTPUT_TOKENS,
    LiveRequestState,
    _provider_error_category,
    response_shape_diagnostic,
    _safe_validation_trace,
    UNSELECTED_SENTINEL,
    audit_provider_payload,
    local_provider_preflight,
    local_abstention_without_provider,
    main,
    run_live_request,
)


def _pack():
    result = RetrievalResult(
        passage_id="P1",
        source_file="synthetic.pdf",
        document_id="doc",
        section="Results",
        pdf_page_start=2,
        pdf_page_end=2,
        text="Prolonged thermal exposure increased aggregation.",
        rank=1,
        retrieval_sources=["lexical"],
    )
    response = HybridSearchResponse([result], "lexical", "disabled", [])
    return build_evidence_pack("What happened?", response)


def _pack_two():
    first = RetrievalResult(
        passage_id="P1",
        source_file="synthetic.pdf",
        document_id="doc",
        section="Results",
        pdf_page_start=2,
        pdf_page_end=2,
        text="Prolonged thermal exposure increased aggregation.",
        rank=1,
        retrieval_sources=["lexical"],
    )
    second = RetrievalResult(
        passage_id="P2",
        source_file="synthetic.pdf",
        document_id="doc",
        section="Discussion",
        pdf_page_start=3,
        pdf_page_end=3,
        text="Extended heat exposure reduced stability.",
        rank=2,
        retrieval_sources=["lexical"],
    )
    return build_evidence_pack("What happened?", HybridSearchResponse([first, second], "lexical", "disabled", []))


class FakeProvider:
    def __init__(self):
        self.calls = []

    def generate_json(self, messages, schema=None, **kwargs):
        self.calls.append((messages, schema, kwargs))
        return ProviderResponse(
            json.dumps(
                {
                    "status": "supported",
                    "claims": [{"text": "Aggregation increased.", "evidence_ids": ["E1"]}],
                    "limitations": [],
                }
            ),
            input_tokens=10,
            output_tokens=12,
            total_tokens=22,
        )


def test_live_script_requires_synthetic_only_and_dry_run_makes_no_provider_call(capsys):
    assert main([]) == 2
    assert main(["--synthetic-only", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Provider instance created: 0" in output
    assert "Network calls: 0" in output
    assert "API key present:" in output
    assert "DEEPSEEK_API_KEY=" not in output


def test_live_authorized_flag_is_required_for_non_dry_run(capsys):
    assert main(["--synthetic-only"]) == 2
    assert "--live-authorized" in capsys.readouterr().out


def test_hard_request_limit_rejects_values_above_three(capsys):
    assert main(["--synthetic-only", "--live-authorized", "--max-live-requests", "4"]) == 2
    assert "between 1 and 3" in capsys.readouterr().out


def test_local_provider_preflight_does_not_create_client_or_network():
    result = local_provider_preflight(api_key_present=True)
    assert result["provider_config_exists"] is True
    assert result["model_non_empty"] is True
    assert result["provider_instance_created"] is False
    assert result["network_calls"] == 0


def test_payload_audit_rejects_unselected_sentinel_and_absolute_path():
    result = audit_provider_payload(
        [
            {"role": "system", "content": "grounding"},
            {"role": "user", "content": f"{UNSELECTED_SENTINEL} C:\\Users\\secret"},
        ],
        api_key="secret-key",
    )
    assert result["selected_sentinels_absent"] is False
    assert result["absolute_path_absent"] is False
    assert result["api_key_absent"] is True


def test_payload_audit_rejects_full_document_sentinel_only_in_memory():
    result = audit_provider_payload(
        [{"role": "user", "content": FULL_DOCUMENT_SENTINEL}],
        api_key="different-key",
    )
    assert result["selected_sentinels_absent"] is False


def test_live_request_uses_eight_hundred_token_cap_and_no_retry():
    provider = FakeProvider()
    state = LiveRequestState(max_requests=1)
    answer, record = run_live_request(provider, _pack(), state, scenario="supported", api_key="not-in-payload")
    assert answer.status == "supported"
    assert record["max_output_tokens"] == LIVE_MAX_OUTPUT_TOKENS == 800
    assert record["input_tokens"] == 10
    assert len(provider.calls) == 1
    assert state.generation_attempts == 1
    assert state.responses_received == 1
    assert state.validated_answers == 1
    assert record["diagnostic"]["validation_status"] == "PASS"
    with pytest.raises(RuntimeError, match="LIVE_REQUEST_LIMIT_EXCEEDED"):
        run_live_request(provider, _pack(), state, scenario="second", api_key="not-in-payload")


def test_unsafe_payload_is_rejected_before_counting_as_provider_request():
    provider = FakeProvider()
    state = LiveRequestState(max_requests=1)
    unsafe = build_evidence_pack(
        "What happened?",
        HybridSearchResponse(
            [
                RetrievalResult(
                    passage_id="P-unsafe",
                    source_file="synthetic.pdf",
                    document_id="doc",
                    section="Results",
                    pdf_page_start=2,
                    pdf_page_end=2,
                    text=f"{UNSELECTED_SENTINEL} should never be sent.",
                    rank=1,
                    retrieval_sources=["lexical"],
                )
            ],
            "lexical",
            "disabled",
            [],
        ),
    )
    answer, record = run_live_request(provider, unsafe, state, scenario="unsafe", api_key="not-in-payload")
    assert answer is None
    assert record["category"] == "PAYLOAD_BOUNDARY_FAIL"
    assert record["provider_called"] is False
    assert state.count == 0
    assert not provider.calls


class ErrorProvider:
    def __init__(self, error):
        self.error = error

    def generate_json(self, messages, schema=None, **kwargs):
        raise self.error


def test_connection_failure_counts_attempt_without_response():
    state = LiveRequestState(max_requests=1)
    answer, record = run_live_request(
        ErrorProvider(ConnectionError("connection reset")),
        _pack(),
        state,
        scenario="supported",
        api_key="secret",
    )
    assert answer is None
    assert record["category"] == "CONNECTION_ERROR"
    assert record["provider_called"] is True
    assert record["response_received"] is False
    assert state.generation_attempts == 1
    assert state.responses_received == 0


def test_local_abstention_does_not_count_generation_request():
    empty = build_evidence_pack(
        "What happened?",
        HybridSearchResponse([], "lexical", "disabled", []),
    )
    answer, record = local_abstention_without_provider(empty)
    assert answer is None
    assert record["provider_called"] is False
    assert record["status"] == "insufficient_evidence"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConnectionError("connection reset"), "CONNECTION_ERROR"),
        (TimeoutError("timed out"), "TIMEOUT"),
        (type("AuthError", (Exception,), {"code": "auth_error"})("unauthorized"), "AUTH_ERROR"),
        (type("RateError", (Exception,), {"code": "rate_limit"})("limited"), "RATE_LIMIT"),
        (type("ServerError", (Exception,), {"status_code": 500})("server"), "PROVIDER_5XX"),
        (type("ClientError", (Exception,), {"status_code": 403})("client"), "AUTH_ERROR"),
        (ValueError("invalid json body"), "MALFORMED_PROVIDER_RESPONSE"),
        (RuntimeError("sdk exploded"), "UNKNOWN_PROVIDER_ERROR"),
    ],
)
def test_provider_transport_errors_have_deterministic_categories(error, expected):
    assert _provider_error_category(error) == expected


class StaticProvider:
    def __init__(self, content):
        self.content = content

    def generate_json(self, messages, schema=None, **kwargs):
        return ProviderResponse(self.content)


def _run_static(content, pack=None):
    state = LiveRequestState(max_requests=1)
    answer, record = run_live_request(StaticProvider(content), pack or _pack(), state, scenario="fixture", api_key="secret")
    return answer, record, state


def test_malformed_json_is_diagnosed_without_raw_response():
    answer, record, state = _run_static('{"status": "supported"')
    assert answer.status == "validation_failed"
    assert record["diagnostic"]["validation_reasons"] == ["JSON_PARSE_ERROR"]
    assert record["diagnostic"]["parse_status"] == "FAIL"
    assert state.generation_attempts == 1
    assert state.responses_received == 1
    assert state.validated_answers == 0
    assert "raw_response" not in record["diagnostic"]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ('{"status":"supported","claims":[],"limitations":[]}', "ALL_CLAIMS_REJECTED"),
        ('{"status":"supported","claims":[{"text":"x"}],"limitations":[]}', "MISSING_EVIDENCE_ID"),
        ('{"status":"supported","claims":[{"text":"x","evidence_ids":["E999"]}],"limitations":[]}', "UNKNOWN_EVIDENCE_ID"),
    ],
)
def test_claim_validation_failures_have_specific_reasons(content, reason):
    answer, record, state = _run_static(content)
    assert answer.status == "validation_failed"
    assert reason in record["diagnostic"]["validation_reasons"]
    assert state.generation_attempts == 1
    assert state.responses_received == 1
    assert state.validated_answers == 0


def test_mixed_valid_and_invalid_claims_is_diagnosed_and_valid_claim_survives():
    content = (
        '{"status":"supported","claims":['
        '{"text":"valid","evidence_ids":["E1"]},'
        '{"text":"invalid","evidence_ids":["E999"]}],"limitations":[]}'
    )
    answer, record, state = _run_static(content)
    assert answer.status == "supported"
    assert len(answer.claims) == 1
    assert record["diagnostic"]["validation_status"] == "FAIL"
    assert "UNKNOWN_EVIDENCE_ID" in record["diagnostic"]["validation_reasons"]
    assert state.validated_answers == 1


def test_page_metadata_is_ignored_and_does_not_change_program_bound_citation():
    content = '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"],"page":99}],"limitations":[]}'
    answer, record, _ = _run_static(content)
    assert answer.status == "supported"
    assert record["category"] == "VALIDATION_FAIL"
    assert "UNSAFE_CLAIM_METADATA" in record["diagnostic"]["validation_reasons"]
    assert record["evidence_ids"] == ["E1"]


@pytest.mark.parametrize("fixture_name", ["A_valid_json", "B_markdown_fenced_json"])
def test_valid_replay_fixtures_are_supported(fixture_name):
    answer, record, _ = _run_static(REPLAY_FIXTURES[fixture_name])
    assert answer.status == "supported"
    assert record["diagnostic"]["validation_status"] == "PASS"


def test_insufficient_evidence_replay_is_a_valid_abstention():
    answer, record, state = _run_static(REPLAY_FIXTURES["H_insufficient_evidence"])
    assert answer.status == "insufficient_evidence"
    assert record["diagnostic"]["validation_status"] == "PASS"
    assert state.validated_answers == 1


def test_prompt_injection_style_output_is_not_a_semantic_truth_test():
    answer, record, _ = _run_static(REPLAY_FIXTURES["J_prompt_injection_style"])
    assert answer.status == "supported"
    assert record["diagnostic"]["validation_status"] == "PASS"
    assert record["rendered_contains_injection_claim"] is True


@pytest.mark.parametrize("fixture_name", sorted(COMPATIBILITY_MATRIX))
def test_schema_compatibility_matrix_has_explicit_strict_expectation(fixture_name):
    answer, record, _ = _run_static(COMPATIBILITY_MATRIX[fixture_name], _pack_two() if fixture_name == "G_evidence_ids_two" else None)
    expected = COMPATIBILITY_EXPECTED[fixture_name]
    accepted = record["diagnostic"]["validation_status"] == "PASS"
    assert accepted is (expected == "STRICT_ACCEPT")
    if expected == "STRICT_REJECT":
        assert record["diagnostic"]["validation_reasons"]


def test_response_shape_diagnostic_contains_structure_only():
    payload = {
        "status": "supported",
        "claims": [
            {
                "text": "SECRET_CLAIM_SENTINEL",
                "evidence_ids": ["E1"],
            }
        ],
        "limitations": [],
        "raw": "SECRET_RAW_RESPONSE_SENTINEL",
        "path": "C:\\SyntheticPrivateUser\\fixture.pdf",
        "key": "sk-secret-test",
    }
    diagnostic = response_shape_diagnostic(payload)
    rendered = json.dumps(diagnostic.as_dict(), ensure_ascii=False)
    assert "SECRET_CLAIM_SENTINEL" not in rendered
    assert "SECRET_RAW_RESPONSE_SENTINEL" not in rendered
    assert "PrivateUser" not in rendered
    assert "sk-secret-test" not in rendered
    assert diagnostic.claim_shapes[0]["text_type"] == "str"
    assert diagnostic.claim_shapes[0]["evidence_ids_count"] == 1


def test_validation_stages_separate_citation_from_schema():
    answer, record, _ = _run_static(
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E999"]}],"limitations":[]}'
    )
    trace = record["validation_trace"]
    assert answer.status == "validation_failed"
    assert trace["parse_status"] == "PASS"
    assert trace["top_level_schema_status"] == "PASS"
    assert trace["claim_schema_status"] == "PASS"
    assert trace["citation_status"] == "FAIL"
    assert "UNKNOWN_EVIDENCE_ID" in trace["validation_reasons"]


def test_validation_stages_report_status_consistency():
    _, record, _ = _run_static(
        '{"status":"insufficient_evidence","claims":[{"text":"x","evidence_ids":["E1"]}],"limitations":[]}'
    )
    trace = record["validation_trace"]
    assert trace["status_consistency_status"] == "FAIL"
    assert "STATUS_CLAIM_INCONSISTENCY" in trace["validation_reasons"]


def test_diagnostics_never_include_secret_path_or_raw_response():
    raw = '{"status":"supported","claims":[{"text":"secret-key LOCAL_TEST_PATH","evidence_ids":["E1"]}],"limitations":[]}'
    _, record, _ = _run_static(raw)
    rendered = json.dumps(record["diagnostic"], ensure_ascii=False)
    assert "secret-key" not in rendered
    assert "LOCAL_TEST_PATH" not in rendered
    assert "valid" not in record["diagnostic"]
