import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


class BetaUpdate(BaseModel):
    beta: Annotated[list[float], Field(min_length=3, max_length=3)]

    @field_validator("beta")
    @classmethod
    def non_negative(cls, v: list[float]) -> list[float]:
        if any(b < 0 for b in v):
            raise ValueError("beta values must be >= 0")
        return v


class SimTunerEndpointPlugin:
    name = "sim_tuner"
    required_tasks = None

    def attach_router(self, app: FastAPI) -> None:
        if not os.environ.get("VLLM_SIM_TUNER"):
            return

        router = APIRouter()

        @router.post("/sim/config")
        def update_config(update: BetaUpdate) -> dict:
            config_path = os.environ.get("VLLM_SIM_CONFIG_PATH")
            if not config_path:
                raise HTTPException(
                    status_code=503,
                    detail="VLLM_SIM_CONFIG_PATH is not set",
                )
            p = Path(config_path)
            if not p.exists():
                raise HTTPException(
                    status_code=503,
                    detail="Config file not yet available; sim may still be starting",
                )
            data = json.loads(p.read_text())
            data["latency"]["beta"] = update.beta
            # Atomic write: write to tmp in same dir, then os.replace
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=p.parent,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(data, tmp)
                tmp_path = tmp.name
            os.replace(tmp_path, config_path)
            return {"status": "ok", "beta": update.beta}

        app.include_router(router)

    async def init_state(self, engine_client, state, args) -> None:
        pass  # no engine interaction needed


def create_plugin() -> SimTunerEndpointPlugin:
    return SimTunerEndpointPlugin()
