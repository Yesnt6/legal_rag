# Start with configurable EmbeddingGemma and BGE reranking

The initial retrieval pipeline will use `embeddinggemma` through Ollama for
passage and query embeddings and `BAAI/bge-reranker-v2-m3` locally for candidate
reranking. Both model identifiers and providers remain configurable because this
selection is a practical baseline for the target device and multilingual corpus,
not a permanent quality conclusion.

