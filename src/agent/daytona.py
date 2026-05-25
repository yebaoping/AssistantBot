import os

from daytona import Daytona, Sandbox
from daytona import CreateSandboxFromSnapshotParams
from daytona.common.errors import DaytonaNotFoundError
from dotenv import load_dotenv


load_dotenv()

SANDBOX_HOME_PATH = "/home/daytona"

SANDBOX_SKILLS_PATH = os.path.join(SANDBOX_HOME_PATH, "skills")


daytona_client = Daytona()


def upload_skills(sandbox: Sandbox, skills_path: str) -> None:
    if not skills_path:
        return

    for skill in os.listdir(skills_path):
        _upload_skill_file(sandbox, skill, os.path.join(skills_path, skill))


def _upload_skill_file(sandbox: Sandbox, prefix: str, skill_path: str) -> None:
    for skill_file in os.listdir(skill_path):
        path = os.path.join(skill_path, skill_file)
        if os.path.isdir(path):
            _upload_skill_file(sandbox, os.path.join(prefix, skill_file), path)
        else:
            with open(path, "r") as file:
                sandbox.fs.upload_file(
                    file.read().encode("utf-8"),
                    os.path.join(SANDBOX_SKILLS_PATH, prefix, skill_file),
                )


def create_sandbox(name: str) -> Sandbox:
    try:
        sandbox = daytona_client.get(name)
    except DaytonaNotFoundError:
        sandbox = daytona_client.create(
            CreateSandboxFromSnapshotParams(
                snapshot=os.getenv("DAYTONA_SANDBOX_SNAPSHOT"),
                name=name,
                auto_delete_interval=1,
            )
        )

    sandbox.start()
    upload_skills(sandbox, os.getenv("SKILLS_PATH"))

    return sandbox
