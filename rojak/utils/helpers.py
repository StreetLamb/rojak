from datetime import datetime, timedelta
import inspect
from rojak.types import RetryPolicy
from temporalio.common import RetryPolicy as TRetryPolicy


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def create_retry_policy(retry_policy: RetryPolicy | None) -> TRetryPolicy | None:
    """Convert serialisable retry policy to Temporal RetryPolicy."""
    if retry_policy is None:
        return None

    initial_interval = (
        timedelta(seconds=retry_policy.initial_interval_in_seconds)
        if retry_policy.initial_interval_in_seconds
        else timedelta(seconds=1)
    )

    maximum_interval = (
        timedelta(seconds=retry_policy.maximum_interval_in_seconds)
        if retry_policy.maximum_interval_in_seconds
        else None
    )

    return TRetryPolicy(
        initial_interval=initial_interval,
        backoff_coefficient=retry_policy.backoff_coefficient,
        maximum_interval=maximum_interval,
        maximum_attempts=retry_policy.maximum_attempts,
        non_retryable_error_types=retry_policy.non_retryable_error_types,
    )


def debug_print(debug: bool, timestamp: datetime, *args: str) -> None:
    if not debug:
        return
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")
