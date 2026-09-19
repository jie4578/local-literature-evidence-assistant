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

COMPATIBILITY_MATRIX = {
    "A_canonical_valid": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"]}],'
        '"limitations":[]}'
    ),
    "B_limitations_missing": '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"]}]}',
    "C_limitations_string": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"]}],'
        '"limitations":"none"}'
    ),
    "D_limitations_null": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"]}],'
        '"limitations":null}'
    ),
    "E_evidence_ids_string": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":"E1"}],'
        '"limitations":[]}'
    ),
    "F_evidence_ids_one": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"]}],'
        '"limitations":[]}'
    ),
    "G_evidence_ids_two": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1","E2"]}],'
        '"limitations":[]}'
    ),
    "H_claim_extra_confidence": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"],"confidence":0.9}],'
        '"limitations":[]}'
    ),
    "I_top_level_reasoning": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"]}],'
        '"limitations":[],"reasoning":"omitted"}'
    ),
    "J_status_uppercase": (
        '{"status":"SUPPORTED","claims":[{"text":"x","evidence_ids":["E1"]}],'
        '"limitations":[]}'
    ),
    "K_status_lowercase": (
        '{"status":"supported","claims":[{"text":"x","evidence_ids":["E1"]}],'
        '"limitations":[]}'
    ),
    "L_claim_key_instead_of_text": (
        '{"status":"supported","claims":[{"claim":"x","evidence_ids":["E1"]}],'
        '"limitations":[]}'
    ),
    "M_citations_key_instead_of_evidence_ids": (
        '{"status":"supported","claims":[{"text":"x","citations":["E1"]}],'
        '"limitations":[]}'
    ),
    "N_supported_empty_claims": '{"status":"supported","claims":[],"limitations":[]}',
    "O_insufficient_empty_claims": '{"status":"insufficient_evidence","claims":[],"limitations":[]}',
    "P_insufficient_with_claims": (
        '{"status":"insufficient_evidence","claims":[{"text":"x","evidence_ids":["E1"]}],'
        '"limitations":[]}'
    ),
}

COMPATIBILITY_EXPECTED = {
    "A_canonical_valid": "STRICT_ACCEPT",
    "B_limitations_missing": "STRICT_REJECT",
    "C_limitations_string": "STRICT_REJECT",
    "D_limitations_null": "STRICT_REJECT",
    "E_evidence_ids_string": "STRICT_REJECT",
    "F_evidence_ids_one": "STRICT_ACCEPT",
    "G_evidence_ids_two": "STRICT_ACCEPT",
    "H_claim_extra_confidence": "STRICT_REJECT",
    "I_top_level_reasoning": "STRICT_REJECT",
    "J_status_uppercase": "STRICT_REJECT",
    "K_status_lowercase": "STRICT_ACCEPT",
    "L_claim_key_instead_of_text": "STRICT_REJECT",
    "M_citations_key_instead_of_evidence_ids": "STRICT_REJECT",
    "N_supported_empty_claims": "STRICT_REJECT",
    "O_insufficient_empty_claims": "STRICT_ACCEPT",
    "P_insufficient_with_claims": "STRICT_REJECT",
}
