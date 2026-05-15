from __future__ import annotations

from pathlib import Path
from typing import Any

from envs.ppm_writer import write_ppm
from hal.fake_manipulation_driver import FakeManipulationDriver
from runtime.repository import WorkspaceRepository, resolve_repo
from tools.response import ToolResponse


class RenderFakeEnvTool:
    name = "render_fake_env"
    description = "Render the current workspace ENVIRONMENT.md state to a PPM image."

    def __init__(
        self,
        workspace: str | Path | WorkspaceRepository = "workspace",
        output: str | Path = "outputs/fake_env_tool.ppm",
    ):
        self._repo_params: str | Path | WorkspaceRepository = workspace
        self.output = Path(output)

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        output = Path(parameters.get("output") or self.output)
        repo.initialize()
        driver = FakeManipulationDriver(seed=int(parameters.get("seed", 0)))
        driver.load_environment(repo.get_environment())
        image = driver.env.render_rgb()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_ppm(output, image)
        return ToolResponse.success(
            "rendered fake environment",
            data={"path": str(output), "shape": list(image.shape), "dtype": str(image.dtype)},
        )
