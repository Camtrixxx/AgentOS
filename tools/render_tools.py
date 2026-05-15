from __future__ import annotations

from pathlib import Path
from typing import Any

from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.environment_io import load_environment_document
from runtime.workspace import initialize_workspace
from envs.ppm_writer import write_ppm
from tools.response import ToolResponse


class RenderFakeEnvTool:
    name = "render_fake_env"
    description = "Render the current workspace ENVIRONMENT.md state to a PPM image."

    def __init__(self, workspace: str | Path = "workspace", output: str | Path = "outputs/fake_env_tool.ppm"):
        self.workspace = Path(workspace)
        self.output = Path(output)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        workspace = Path(parameters.get("workspace") or self.workspace)
        output = Path(parameters.get("output") or self.output)
        paths = initialize_workspace(workspace)
        driver = FakeManipulationDriver(seed=int(parameters.get("seed", 0)))
        driver.load_environment(load_environment_document(paths.environment))
        image = driver.env.render_rgb()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_ppm(output, image)
        return ToolResponse.success(
            "rendered fake environment",
            data={"path": str(output), "shape": list(image.shape), "dtype": str(image.dtype)},
        )

