# Use one query endpoint with assistance modes

The chatbot will expose one query endpoint that classifies requests as
`RESEARCH`, `COMPLIANCE`, or `OUT_OF_SCOPE` and applies the corresponding cited
response contract. Users may explicitly select research or compliance mode to
override automatic classification, while out-of-scope and unsupported requests
still abstain.

