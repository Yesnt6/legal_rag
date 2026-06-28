# Immediately index supplied documents

The initial system will parse every supplied document and immediately add its
passages, extracted graph entities, and relationships to the active retrieval
corpus. This avoids building a document-approval workflow during the initial
GraphRAG implementation, at the cost of treating every supplied document as
trusted input.

