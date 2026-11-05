"""
Custom exception definitions and error formatting utilities.
"""

import types

__all__: list[str] = [
    "ConfigurationError",
    "DocumentParsingError",
    "EmbeddingError",
    "MyException",
    "ValidationError",
    "error_message_detail",
]


def error_message_detail(error: Exception | str, error_detail: types.ModuleType) -> str:
    """Extract detailed error message including file name, line number, and the error.

    Args:
        error (Exception | str): The exception that was raised.
        error_detail (types.ModuleType): The sys module to extract traceback details.

    Returns:
        str: Formatted error message containing the location and description.
    """
    _, _, exception_traceback = error_detail.exc_info()
    source_filename: str = "unknown"
    line_number: int = -1
    if exception_traceback is not None:
        source_filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno

    return f"Error in {source_filename} at line {line_number}: {error}"


class MyException(Exception):
    """Base custom application exception with traceback and file context."""

    def __init__(
        self, error_message: Exception | str, error_detail: types.ModuleType
    ) -> None:
        """Initialize the custom exception with detailed error information.

        Args:
            error_message (Exception | str): The original exception raised.
            error_detail (types.ModuleType): The sys module to extract traceback details.
        """
        super().__init__(str(error_message))
        self.error_message: str = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        """Return the string representation of the custom exception.

        Returns:
            str: The detailed error message string.
        """
        return self.error_message


class ConfigurationError(MyException):
    """Raised when configuration loading, parsing, or validation fails."""


class DocumentParsingError(MyException):
    """Raised when parsing or extracting documents (PDF, Markdown, TXT) fails."""


class EmbeddingError(MyException):
    """Raised when embedding model initialization or vector generation fails."""


class ValidationError(MyException):
    """Raised when input parameters, thresholds, or dimensions fail validation."""
