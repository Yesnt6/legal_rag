# Disclose unresolved source conflicts

When retrieved source passages support incompatible claims, the system may
prefer a source only when indexed metadata deterministically establishes that it
is applicable, newer, and higher in the authority hierarchy. The model cannot
infer precedence from pretrained knowledge; unresolved conflicts must be
presented with citations as `CONFLICTING_SOURCES`, and the assistant must not
produce a compliance recommendation from them.

