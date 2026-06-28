# Enforce strict versioned structured model outputs

All model-driven stages, including graph extraction, query classification,
answer claim generation, and claim verification, will use versioned Pydantic
schemas and prompts with JSON-schema-constrained Ollama output. Responses must
validate before use; one retry may include validation errors, after which the
stage fails explicitly rather than guessing or silently repairing malformed
output.

