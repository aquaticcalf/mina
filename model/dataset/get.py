import os
from roboflow import Roboflow


def download_dataset(
    api_key: str,
    workspace: str,
    project_name: str,
    version_number: int,
    format: str = "yolov8",
):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(format)
    return dataset


if __name__ == "__main__":
    API_KEY = os.getenv("ROBOFLOW_API_KEY")
    if not API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is not set in the environment")

    WORKSPACE = "mina-orfdd"
    PROJECT = "mina-u7bag"
    VERSION = 2

    download_dataset(
        api_key=API_KEY,
        workspace=WORKSPACE,
        project_name=PROJECT,
        version_number=VERSION,
    )

