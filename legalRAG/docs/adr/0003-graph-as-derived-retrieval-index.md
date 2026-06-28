# Treat the graph as a derived retrieval index

The assistant will construct graph nodes and relationships from supplied source
documents using locally hosted models. Extracted graph facts may guide retrieval
and connect relevant passages, but they are not themselves legal evidence;
answers and compliance conclusions must be supported by citations to the exact
source-document passages from which they were derived. The initial system will
operate without web search or external tool calls.

