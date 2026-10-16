"""
Abstract architectural foundations for the Stego-Latent Phase Disruptor (SLPD).

This module defines the core abstract interfaces and common logging wrappers
shared across all pipeline processing stages. It establishes the uniform runtime
contracts required to execute decoupled algorithmic attacks via the Strategy Pattern.

Compliance: PEP 8, PEP 257, PEP 484 Type Hinting.
Author: e1ephntbrd
"""

from typing import Dict, Any


class PipelineStep:
    """
    Abstract base class representing a single modular node within the processing pipeline.

    Acts as the operational strategy contract interface. Every concrete attack vector
    (spatial warping, generative relaxation, color space disruption) must inherit
    from this class and implement its core abstract runtime execution matrix.
    """

    def execute(self, context: Dict[str, Any]) -> bool:
        """
        Executes the specific algorithmic signal processing or machine learning logic.

        This method must be explicitly overridden by every concrete subclass node
        to execute modifications on the shared pipeline state array container.

        Args:
            context (Dict[str, Any]): A shared dictionary data container acting
                                      as the thread-safe state repository across
                                      the entire pipeline execution context lifecycle.

        Returns:
            bool: True if the specific step logic executed successfully without
                  critical errors, False immediately if the operation breaks.

        Raises:
            NotImplementedError: If the inherited subclass does not explicitly
                                 override this core execution matrix interface.
        """
        raise NotImplementedError(
            f"Execution matrix node '{self.__class__.__name__}' must implement the abstract execute() method."
        )

    def log(self, message: str) -> None:
        """
        Emits real-time execution buffers, diagnostic metrics, and state progress to stdout.

        Provides consistent console tracking structure for each distinct block
        module during sequential pipeline processing.

        Args:
            message (str): The specific configuration, metrics, or diagnostic progress
                           string buffer to display in the standard output stream.

        Returns:
            None
        """
        print(f"[{self.__class__.__name__}] {message}", flush=True)

