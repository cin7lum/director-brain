"""Control signals shared by long-running analysis units."""


class AnalysisCancelledError(RuntimeError):
    """Raised at a safe unit boundary after a durable job cancel request."""
