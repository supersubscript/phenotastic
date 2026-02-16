"""Custom exceptions for the Phenotastic package."""


class PhenotasticError(Exception):
    """Base exception for all Phenotastic errors."""


class PipelineError(PhenotasticError):
    """Error during pipeline execution.

    Raised when:
    - An unknown operation is requested
    - A pipeline step fails to execute
    - Pipeline context is in an invalid state
    """


class ConfigurationError(PhenotasticError):
    """Error in pipeline or operation configuration.

    Raised when:
    - YAML configuration is invalid
    - Required parameters are missing
    - Parameter values are out of valid range
    """


class InvalidMeshError(PhenotasticError):
    """Error related to invalid mesh data.

    Raised when:
    - Mesh has no points or faces
    - Mesh is non-manifold when manifold is required
    - Mesh data is corrupted or inconsistent
    """


class InvalidImageError(PhenotasticError):
    """Error related to invalid image data.

    Raised when:
    - Image has wrong dimensions
    - Image data type is incompatible
    - Image file cannot be read
    """
