"""Run provenance helpers."""

import subprocess


def git_commit() -> str:
    """HEAD commit, suffixed with '-dirty' when the working tree has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    except OSError:
        return "unknown"
    commit = head.stdout.strip() or "unknown"
    return f"{commit}-dirty" if dirty.stdout.strip() else commit
