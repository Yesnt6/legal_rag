# Target an Apple M4 with 16 GB unified memory

The initial system will be designed for a single Apple M4 device with 16 GB of
unified memory. The language-model runtime, Neo4j, Chroma, embedding and
reranking models, and application must coexist within that constraint, so the
main model will initially be limited to a quantized 7B-8B class model served
locally through Ollama.

