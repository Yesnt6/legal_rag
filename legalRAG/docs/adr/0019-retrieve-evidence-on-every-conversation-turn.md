# Retrieve evidence on every conversation turn

Conversation history may be used to rewrite follow-up questions and prior
verified passages may receive a retrieval boost, but every turn must retrieve
and rerank source evidence again. Earlier chatbot answers never become legal
evidence; cached embeddings, graph neighborhoods, retrieval candidates, and
verified claim-passage pairs may be reused to remain within the answer latency
budget.

