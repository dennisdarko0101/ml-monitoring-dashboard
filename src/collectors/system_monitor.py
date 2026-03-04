"""System resource monitoring (CPU, memory, GPU, disk)."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SystemSnapshot:
    """Point-in-time system resource snapshot."""

    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    disk_percent: float
    timestamp: float


class SystemMonitor:
    """Monitors system resources. Falls back to estimation if psutil unavailable."""

    def __init__(self) -> None:
        self._has_psutil = False
        try:
            import psutil  # noqa: F401

            self._has_psutil = True
        except ImportError:
            logger.info("psutil_not_available", msg="Using mock system metrics")

    def get_cpu_usage(self) -> float:
        """Get CPU usage as a percentage (0-100)."""
        if self._has_psutil:
            import psutil

            return psutil.cpu_percent(interval=0.1)
        return self._mock_metric(15.0, 85.0)

    def get_memory_usage(self) -> float:
        """Get memory usage as a percentage (0-100)."""
        if self._has_psutil:
            import psutil

            return psutil.virtual_memory().percent
        return self._mock_metric(30.0, 80.0)

    def get_gpu_usage(self) -> float:
        """Get GPU usage. Returns mock data if no GPU or nvidia-smi unavailable."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return self._mock_metric(0.0, 50.0)

    def get_disk_usage(self) -> float:
        """Get disk usage for the root partition as a percentage."""
        if self._has_psutil:
            import psutil

            root = "/" if platform.system() != "Windows" else "C:\\"
            return psutil.disk_usage(root).percent
        return self._mock_metric(20.0, 70.0)

    def snapshot(self) -> SystemSnapshot:
        """Capture a full system resource snapshot."""
        return SystemSnapshot(
            cpu_percent=self.get_cpu_usage(),
            memory_percent=self.get_memory_usage(),
            gpu_percent=self.get_gpu_usage(),
            disk_percent=self.get_disk_usage(),
            timestamp=time.time(),
        )

    @staticmethod
    def _mock_metric(low: float, high: float) -> float:
        """Generate a deterministic mock metric based on time."""
        import math

        t = time.time()
        normalized = (math.sin(t / 60) + 1) / 2
        return round(low + normalized * (high - low), 1)
