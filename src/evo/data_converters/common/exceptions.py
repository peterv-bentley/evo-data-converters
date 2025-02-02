from evo.common.exceptions import EvoClientException


class ConflictingConnectionDetailsError(EvoClientException):
    """Raised when there are conflicting connection details."""


class MissingConnectionDetailsError(EvoClientException):
    """Raised when no connection details can be derived from inputs."""


class ResponseError(EvoClientException):
    """Raised when an API request fails."""
