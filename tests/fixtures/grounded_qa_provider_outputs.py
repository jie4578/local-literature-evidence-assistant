"""Synthetic provider outputs for offline grounded-QA replay tests only."""

VALID_SUPPORTED = (
    '{"status":"supported","claims":[{"text":"Aggregation increased.",'
    '"evidence_ids":["E1"]}],"limitations":[]}'
)

MARKDOWN_FENCED_SUPPORTED = "```json\n" + VALID_SUPPORTED + "\n```"
MALFORMED_JSON = '{"status":"supported","claims":'
SUPPORTED_WITHOUT_CLAIMS = '{"status":"supported","claims":[],"limitations":[]}'
CLAIM_WITHOUT_EVIDENCE_IDS = (
    '{"status":"supported","claims":[{"text":"Aggregation increased."}],"limitations":[]}'
)
CLAIM_WITH_UNKNOWN_EVIDENCE = (
    '{"status":"supported","claims":[{"text":"Aggregation increased.",'
    '"evidence_ids":["E999"]}],"limitations":[]}'
)
MIXED_VALID_INVALID_CLAIMS = (
    '{"status":"supported","claims":['
    '{"text":"valid","evidence_ids":["E1"]},'
    '{"text":"invalid","evidence_ids":["E999"]}],"limitations":[]}'
)
INSUFFICIENT_EVIDENCE = '{"status":"insufficient_evidence","claims":[],"limitations":[]}'
MODEL_PAGE_METADATA = (
    '{"status":"supported","claims":[{"text":"Aggregation increased.",'
    '"evidence_ids":["E1"],"page":99,"source_file":"forged.pdf"}],"limitations":[]}'
)
PROMPT_INJECTION_STYLE = (
    '{"status":"supported","claims":[{"text":"The treatment cures cancer.",'
    '"evidence_ids":["E1"]}],"limitations":[]}'
)

REPLAY_FIXTURES = {
    "A_valid_json": VALID_SUPPORTED,
    "B_markdown_fenced_json": MARKDOWN_FENCED_SUPPORTED,
    "C_malformed_json": MALFORMED_JSON,
    "D_supported_no_claims": SUPPORTED_WITHOUT_CLAIMS,
    "E_claim_without_evidence_ids": CLAIM_WITHOUT_EVIDENCE_IDS,
    "F_unknown_evidence_id": CLAIM_WITH_UNKNOWN_EVIDENCE,
    "G_mixed_valid_invalid_claims": MIXED_VALID_INVALID_CLAIMS,
    "H_insufficient_evidence": INSUFFICIENT_EVIDENCE,
    "I_forged_page_metadata": MODEL_PAGE_METADATA,
    "J_prompt_injection_style": PROMPT_INJECTION_STYLE,
}
