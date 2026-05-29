"""Domain exceptions for the vector search API."""


class VectorServiceError(Exception):
    """Base exception mapped to HTTP responses in route handlers."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class IndexEmptyError(VectorServiceError):
    def __init__(self, message: str = "Vector index is empty. Index documents first.") -> None:
        super().__init__(message, status_code=404)


class DimensionMismatchError(VectorServiceError):
    def __init__(
        self,
        expected: int,
        actual: int,
        message: str | None = None,
    ) -> None:
        detail = message or (
            f"Embedding dimension mismatch: expected {expected}, got {actual}"
        )
        super().__init__(detail, status_code=400)


class InvalidFiltersError(VectorServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class InvalidSearchTypeError(VectorServiceError):
    def __init__(self, search_type: str) -> None:
        super().__init__(
            f"Invalid search_type '{search_type}'. "
            "Use semantic, filtered, or hybrid.",
            status_code=400,
        )
