from langchain_daytona import DaytonaSandbox

from .daytona import create_sandbox


default_sandbox_name = "assistant_bot_sandbox"

_sandbox_backend: DaytonaSandbox | None = None    


def _get_sandbox_backend() -> DaytonaSandbox:
    global _sandbox_backend
    if _sandbox_backend is not None:
        return _sandbox_backend

    sandbox = create_sandbox(default_sandbox_name) 
    _sandbox_backend = DaytonaSandbox(sandbox=sandbox)

    return _sandbox_backend


class _LazySandboxBackend:
    """Defer Daytona connection until the backend is actually used."""

    def __getattr__(self, name: str):
        return getattr(_get_sandbox_backend(), name)


sandbox_backend = _LazySandboxBackend()
