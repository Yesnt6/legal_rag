# Ingest documents asynchronously with resumable checkpoints

When the full ingestion workflow is implemented, it will run as an asynchronous
staged process targeting roughly one minute per ten pages on the target device.
The pipeline will expose progress, persist checkpoints for resumption after
interruption, and make a document searchable only after passage indexing and
graph construction complete. Individual passage failures may be isolated
without discarding the entire document, but all omissions must be reported. Its
implementation is deferred until after the first research-query vertical slice.
