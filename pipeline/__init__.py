"""
Orchestration core for the Stego-Latent Phase Disruptor (SLPD) package.

This module acts as the main package entry point, exposing the pipeline runtime
orchestrator and strategic steps. It manages sequential execution state variables,
tracks node benchmarking performance metrics, and handles graceful step failure recovery.

Compliance: PEP 8, PEP 257, PEP 484 Type Hinting.
Author: e1ephntbrd
"""

import time
from typing import List, Dict, Any
from .base import PipelineStep
from .steps import GeometryWarpStep, LatentDiffusionStep, QuantizationAndLanczosStep

# Expose internal components clean and explicit to the package boundary
__all__ = [
    "SynthIDBypassPipeline",
    "GeometryWarpStep",
    "LatentDiffusionStep",
    "QuantizationAndLanczosStep",
]


class SynthIDBypassPipeline:
    """
    Main orchestration engine managing the execution lifecycle, execution context
    state variables, and step execution metrics.
    """

    def __init__(self) -> None:
        """
        Initializes an empty pipeline state context container and tracking arrays.
        """
        self.steps: List[PipelineStep] = []
        self.context: Dict[str, Any] = {}

    def add_step(self, step: PipelineStep) -> "SynthIDBypassPipeline":
        """
        Appends an execution step strategy node to the workflow matrix sequence.

        Args:
            step (PipelineStep): An instance of a processing step class
                                 inheriting from the base PipelineStep interface.

        Returns:
            SynthIDBypassPipeline: The orchestrator engine instance itself
                                   to enable clean fluid Method Chaining.
        """
        self.steps.append(step)
        return self

    def run(self, input_path: str, output_path: str) -> bool:
        """
        Triggers a sequential execution workflow matrix loop across all registered nodes.

        Dynamically initializes the multi-modal dictionary context container, manages
        real-time logging output streams, and performs high-precision execution timing benchmarks.

        Args:
            input_path (str): System filesystem string path to the raw original image source.
            output_path (str): System filesystem path specifying where the final verified
                               clean image will be written on the disk.

        Returns:
            bool: True if every strategy node executed successfully,
                  False immediately if any block breaks or returns a critical exception.
        """
        self.context = {
            "input_path": input_path,
            "output_path": output_path,
            "image": None
        }

        print("\n=======================================================", flush=True)
        print("=== INITIATING FRACTAL DE-SYNCHRONIZATION PIPELINE ===", flush=True)
        print("=======================================================\n", flush=True)

        global_start_time: float = time.perf_counter()

        for step in self.steps:
            step_name: str = step.__class__.__name__
            step_start_time: float = time.perf_counter()

            # Execute step and perform strict Boolean response filtering
            success: bool = step.execute(self.context)

            if not success:
                print(
                    f"\n[CRITICAL FAILURE] Pipeline execution matrix aborted at node: {step_name}\n",
                    flush=True
                )
                return False

            step_elapsed: float = time.perf_counter() - step_start_time
            print(f"-> Node [{step_name}] finished processing in {step_elapsed:.3f} seconds.\n", flush=True)

        global_elapsed: float = time.perf_counter() - global_start_time
        print("-------------------------------------------------------", flush=True)
        print(f"🎉 PIPELINE SUCCESS: Processing block complete in {global_elapsed:.3f} seconds.", flush=True)
        print(f"Target successfully saved to storage container -> {output_path}", flush=True)
        print("=======================================================\n", flush=True)
        return True

