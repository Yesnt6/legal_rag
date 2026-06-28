# Require objective provenance for extracted graph facts

Every extracted graph node and relationship must link to a supporting passage,
exact supporting text span, page number, and extractor model and prompt version.
Facts without a valid source span are rejected before entering Neo4j, and
answers cite source passages rather than graph relationships. The system will
not store model-reported extraction confidence because local language-model
confidence is not reliably calibrated.

