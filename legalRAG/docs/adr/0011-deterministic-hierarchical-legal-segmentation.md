# Segment legal documents hierarchically and deterministically

Native-text PDFs will be segmented into a stable hierarchy of document,
heading, section, clause, subclause, paragraph, and retrieval passage while
preserving page and character offsets. Deterministic layout and numbering rules
own passage boundaries; a local model may classify ambiguous headings as a
fallback but cannot directly invent unstable chunks. Oversized provisions may
be split into child passages that retain links to their complete parent context.

