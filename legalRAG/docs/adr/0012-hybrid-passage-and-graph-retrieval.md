# Combine vector, keyword, and graph retrieval

Question answering will combine semantic vector retrieval, exact keyword and
full-text retrieval, and bounded graph expansion from entities identified in the
question. A locally hosted reranker will score the unified candidate set, and
only source passages selected after reranking may enter answer generation.

