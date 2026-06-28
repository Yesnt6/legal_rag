# Preserve mentions and canonicalize entities deterministically

Every extracted entity mention will be preserved with its source span and linked
to a canonical entity through deterministic normalization and an explicit alias
table. The extraction model may propose aliases but cannot silently merge
entities; ambiguous mentions remain distinct until deterministic rules establish
their equivalence.

