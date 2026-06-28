# Verify generated claims in an isolated pass using the main model

The initial system will reuse its main locally hosted language model for answer
generation and claim verification rather than loading a separate entailment
model. Each atomic claim will be verified in a fresh context containing only the
claim and its cited passages; the verifier must return `SUPPORTED`,
`CONTRADICTED`, or `NOT_ESTABLISHED` with exact supporting spans, and
deterministic validation will reject missing spans and every result other than
`SUPPORTED`. This reduces local resource requirements, while accepting that
generation and verification failures may remain correlated.

