import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProfilerCfg:
    enable: bool = False
    wait: int = 1  # steps to skip before profiling
    warmup: int = 1  # warmup steps
    active: int = 3  # active profiling steps
    repeat: int = 1  # number of profiling cycles
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True


class Profiler:
    """Wrapper around torch.profiler.profile that works as a context manager."""

    def __init__(self, cfg: ProfilerCfg, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir / "profile"
        self._profiler: Optional[torch.profiler.profile] = None

    @property
    def enabled(self) -> bool:
        return self.cfg.enable

    def __enter__(self) -> "Profiler":
        if not self.cfg.enable:
            return self

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Only include CUDA activity if available
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.cfg.wait,
                warmup=self.cfg.warmup,
                active=self.cfg.active,
                repeat=self.cfg.repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=self.cfg.record_shapes,
            profile_memory=self.cfg.profile_memory,
            with_stack=self.cfg.with_stack,
        )
        self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._profiler is not None:
            self._profiler.__exit__(exc_type, exc_val, exc_tb)
            self._profiler = None
        return False

    def step(self):
        """Advance the profiler to the next step."""
        if self._profiler is not None:
            self._profiler.step()
