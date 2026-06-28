# Backlog

## Durable ingestion jobs

Add resumable background document ingestion after the initial chatbot query
pipeline is working. The likely implementation is a single in-process worker
with SQLite-backed job state and stage checkpoints, startup recovery, and
one-document-at-a-time processing to respect the local memory budget. Redis,
Celery, and parallel ingestion remain deferred.

## Deferred product capabilities

- Parse and validate native-text PDFs.
- Build the Streamlit document and chatbot interface.
- Add compliance assistance and cited checklists.
- Add a reviewed evaluation suite and formal release thresholds.
- Reconsider document approval and quarantine if untrusted uploads become a
  practical concern.
