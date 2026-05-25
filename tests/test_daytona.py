from agent.daytona import create_sandbox,SANDBOX_SKILLS_PATH
from daytona import SandboxState

# pytest test_daytona.py::test_create_sandbox -s
def test_create_sandbox():
    sandbox = create_sandbox("test_sandbox")
    assert sandbox is not None
    assert sandbox.state == SandboxState.STARTED

    skills_files = sandbox.fs.list_files(SANDBOX_SKILLS_PATH)
    print(skills_files)
