# Store source documents as immutable versions

Every supplied PDF will receive a SHA-256 checksum, and exact duplicate files
will be rejected. Changed files will create immutable document versions with
independent passages and graph provenance; the latest version is searched by
default, while historical versions remain available for questions with an
applicable date. Existing graph facts will never be mutated to represent a newer
document version.

