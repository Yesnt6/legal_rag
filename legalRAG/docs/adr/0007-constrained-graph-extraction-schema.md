# Constrain graph extraction to an allowed schema

The local extraction model will emit validated structured data using the allowed
node types `Document`, `Passage`, `Authority`, `Provision`, `Actor`, `Action`,
`Requirement`, `Evidence`, `Deadline`, `Condition`, `Exception`, `Consequence`,
and `LegalConcept`. Relationships are restricted to `CONTAINS`, `MENTIONS`,
`REQUIRES`, `PERFORMED_BY`, `SUPPORTED_BY`, `TRIGGERED_BY`, `HAS_DEADLINE`,
`NEEDS_EVIDENCE`, `HAS_EXCEPTION`, `NONCOMPLIANCE_CAUSES`, `AMENDS`,
`SUPERSEDES`, `CITES`, and `RELATES_TO`; this reduces ontology drift from small
local models while retaining a controlled generic relationship.

