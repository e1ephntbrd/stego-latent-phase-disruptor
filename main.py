"""
Operational system entry point for the Stego-Latent Phase Disruptor (SLPD).

This module provides a Command Line Interface (CLI) configuration framework
to parse user arguments, handle filesystem checks, initialize the execution
context state, and trigger the strategic sequential de-synchronization pipeline.

Compliance: PEP 8, PEP 257, PEP 484 Type Hinting.
Author: e1ephntbrd
"""

import os
import sys
import argparse
from typing import NoReturn
from pipeline import (
    SynthIDBypassPipeline,
    GeometryWarpStep,
    LatentDiffusionStep,
    QuantizationAndLanczosStep,
)


def main() -> None:
    """
    Main operational execution block that initializes the CLI parser configuration.

    Validates filesystem paths dynamically, configures the modular strategy steps
    via method chaining, and securely invokes the core pipeline orchestration engine.

    Returns:
        None

    Raises:
        SystemExit: If the target input path is missing or invalid on the system disk.
    """
    # 1. Instantiate the arguments parser framework
    parser = argparse.ArgumentParser(
        description="Stego-Latent Phase Disruptor (SLPD): Advanced Watermark Attenuation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 2. Map system interface parameters and arguments
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="input_banana.jpg",
        help="System filesystem string path to the raw input image containing embedded SynthID tokens."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="bypass_output.png",
        help="System filesystem path specifying where the final clean high-fidelity PNG container will be saved."
    )

    args: argparse.Namespace = parser.parse_args()

    # 3. Defensive path parsing and runtime verification
    if not os.path.exists(args.input):
        print(
            f"\n[CRITICAL ERROR] File Initialization Failed: Source path '{args.input}' does not exist on disk.",
            file=sys.stderr,
            flush=True
        )
        print("Please check your input string reference or verify file layout in your directory.\n", file=sys.stderr,
              flush=True)
        sys.exit(1)

    try:
        # 4. Initialize the orchestrator engine context
        pipeline: SynthIDBypassPipeline = SynthIDBypassPipeline()

        # 5. Build the pipeline workflow execution array via Strategy Pattern Method Chaining
        pipeline.add_step(GeometryWarpStep(target_size=768, angle=0.15, scale=1.001)) \
            .add_step(LatentDiffusionStep(strength=0.12, steps=25, guidance_scale=7.5)) \
            .add_step(QuantizationAndLanczosStep(max_colors=256, shift_amplitude=0.5))

        # 6. Trigger sequential execution matrix flow
        pipeline.run(input_path=args.input, output_path=args.output)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Process interrupted by user workspace action. Aborting pipeline...", flush=True)
        sys.exit(130)

    except Exception as error:
        print(f"\n[UNHANDLED ENGINE EXCEPTION] Failure details: {error}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

