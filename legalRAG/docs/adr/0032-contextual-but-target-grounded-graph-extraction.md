# Extract graph facts with context but ground them in the target passage

Each graph-extraction call will receive one target retrieval passage, its
complete parent provision, ancestor heading path, immediately adjacent sibling
passages, and document metadata. Context may help the local model interpret the
target, but only nodes and relationships supported by exact spans in the target
passage may be persisted during that call.

