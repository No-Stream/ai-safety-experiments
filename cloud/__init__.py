"""AWS Batch machinery for the games training arms: image, entrypoint, and job submission.

A package rather than loose scripts so the request-shaping in `submit_job` is importable and
testable without a real AWS account. See `docs/scratch/games-cloud-notes-2026-08-17.md` for the
operator-facing notes: image tags, the driver floor, and the fast-kernel decision.
"""
