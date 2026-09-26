# Failure Behavior

## Model unavailable (connection refused)
- LLMAdapter catches URLError
- Returns LLMResult(decision=None, error="SEMANTIC_REASONER_UNAVAILABLE: ollama connection failed")
- SemanticDirectorReasoner propagates error
- SemanticDirectorService returns status="SEMANTIC_REASONER_UNAVAILABLE"
- NO silent fallback to heuristic or keyword parser

## Invalid JSON from model
- LLMAdapter catches JSONDecodeError
- Returns error with raw_response for debugging
- Fail closed

## Schema validation failure
- DirectorDecision.model_validate_json() raises
- LLMAdapter catches, returns error
- extra=forbid ensures unknown fields are rejected
- Fail closed

## Empty response
- LLMAdapter checks content.strip()
- Returns "SEMANTIC_REASONER_UNAVAILABLE: model returned empty content"
- Fail closed

## Timeout
- urllib.request.urlopen with timeout=300s
- URLError on timeout → fail closed
