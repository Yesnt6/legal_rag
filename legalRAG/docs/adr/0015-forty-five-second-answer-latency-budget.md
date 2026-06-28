# Set a forty-five-second answer latency budget

The initial local assistant will target a maximum end-to-end answer latency of
forty-five seconds on the target Apple M4 device. This budget includes query
analysis, hybrid retrieval, graph expansion, reranking, answer generation,
separate-pass claim verification, and final response assembly; the interface
should expose progress while these stages run.

