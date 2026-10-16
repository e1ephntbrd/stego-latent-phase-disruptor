"""
Concrete strategic execution nodes for the Stego-Latent Phase Disruptor (SLPD).

This module contains the tactical implementation blocks of the framework. Each
class encapsulates a separate attack layer (Spatial Warp, Latent Relaxation,
and Color Matrix Disruption) designed to act sequentially on the shared state matrix.

Compliance: PEP 8, PEP 257, PEP 484 Type Hinting.
Author: e1ephntbrd
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from .base import PipelineStep


class GeometryWarpStep(PipelineStep):
    """
    Applies high-precision spatial phase displacement paired with high-frequency
    structural noise injection to decouple absolute grid anchors on uniform backdrops.
    """

    def __init__(self, target_size: int = 768, angle: float = 0.20, scale: float = 1.002) -> None:
        """
        Initializes the spatial warping geometry parameters.

        Args:
            target_size (int): The target spatial resolution for matrix processing.
            angle (float): Sub-visual micro-rotation value in degrees.
            scale (float): Sub-pixel fractional scale factor.
        """
        self.target_size: int = target_size
        self.angle: float = angle
        self.scale: float = scale

    def execute(self, context: Dict[str, Any]) -> bool:
        self.log("Executing high-intensity structural grain injection for solid backgrounds...")

        img: Optional[np.ndarray] = cv2.imread(context["input_path"])
        if img is None:
            self.log(f"CRITICAL ERROR: Source image stream corrupted or missing at: {context['input_path']}")
            return False

        # Rescale the raw input block array down to standard training tokens
        img_resized: np.ndarray = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        # 🚀 HIGH-FREQUENCY FILM GRAIN ATTACK ON MONOTONIC BACKDROPS
        # Generates a randomized normal noise matrix (sigma = 8) to disrupt vector fields
        mean: float = 0.0
        sigma: float = 8.0
        gaussian_noise: np.ndarray = np.random.normal(mean, sigma, img_resized.shape).astype(np.int16)

        # Merge structural noise with the pristine array layer
        img_noisy: np.ndarray = np.clip(img_resized.astype(np.int16) + gaussian_noise, 0, 255).astype(np.uint8)

        # Construct the affine rotational transformation field matrix
        center: tuple = (self.target_size // 2, self.target_size // 2)
        rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(center, self.angle, self.scale)

        # Remap spatial array utilizing windowed sinc Lanczos-4 interpolation
        context["image"] = cv2.warpAffine(
            img_noisy,
            rotation_matrix,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )
        self.log("Topological structures and solid fills successfully disrupted.")
        return True


class LatentDiffusionStep(PipelineStep):
    """
    Performs discrete latent space iterations to cycle high-frequency texture fields
    and overwrite proprietary network generation patterns.
    """

    def __init__(self, strength: float = 0.12, steps: int = 25, guidance_scale: float = 7.5) -> None:
        """
        Initializes the generative neural relaxation configurations.

        Args:
            strength (float): Latent denoising ratio (fraction of structural pixel reset).
            steps (int): Total number of discrete inference timesteps.
            guidance_scale (float): Classifier-free guidance constraint scale factor.
        """
        self.strength: float = strength
        self.steps: int = steps
        self.guidance_scale: float = guidance_scale

    def execute(self, context: Dict[str, Any]) -> bool:
        self.log("Evaluating physical accelerator infrastructure status...")

        # Configure host compute device structures dynamically
        if torch.cuda.is_available():
            device: str = "cuda"
            torch_dtype: torch.dtype = torch.float16
            variant: Optional[str] = "fp16"
            self.log("CUDA runtime context verified. Processing on hardware GPU acceleration layer.")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            variant = None
            self.log("WARNING: Accelerated hardware absent. Falling back to host CPU execution matrix.")

        torch.cuda.empty_cache()
        self.log("Instantiating neural weights for latent space relaxation pipeline...")

        # Load independent pipeline weights from Hugging Face hub
        pipe: StableDiffusionImg2ImgPipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            variant=variant
        ).to(device)

        # Replace default sampler with high-fidelity DPM++ Multistep Karras scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

        # Micro-memory management optimizations to prevent VRAM Out-of-Memory crashes
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        if device == "cuda":
            pipe.enable_attention_slicing()

        # Cache state matrices onto storage container to feed the PIL loader engine safely
        temp_cache_path: str = "temp_latent.jpg"
        cv2.imwrite(temp_cache_path, context["image"])
        init_image: Image.Image = load_image(temp_cache_path).convert("RGB")

        self.log(f"Launching latent space denoising loop (Steps: {self.steps}, Strength: {self.strength})...")
        output_object: Any = pipe(
            prompt="masterpiece, highly detailed, sharp focus, professional studio photography, clean sRGB textures",
            negative_prompt="blurry, low quality, noise, artifacts, digital errors",
            image=init_image,
            strength=self.strength,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.steps
        )

        # Unroll the output list object to process the underlying image array safely
        generated_image: Image.Image = output_object.images[0]
        context["image"] = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)

        # Evict transient disk caches
        if os.path.exists(temp_cache_path):
            os.remove(temp_cache_path)

        self.log("Generative tensor pass complete. Original steganographic tokens dissolved.")
        return True


class QuantizationAndLanczosStep(PipelineStep):
    """
    Executes global color-space vector shifts and non-linear Gaussian vignetting
    to annihilate macro-quantization encodings on flat studio fields.
    """

    def __init__(self, max_colors: int = 256, shift_amplitude: float = 0.5) -> None:
        """
        Initializes the post-processing color quantization structures.

        Args:
            max_colors (int): Total allowed indices within the restricted color palette.
            shift_amplitude (float): Sub-pixel high-frequency jitter radius in pixels.
        """
        self.max_colors: int = max_colors
        self.shift: float = shift_amplitude

    def execute(self, context: Dict[str, Any]) -> bool:
        self.log("Executing Global Color Space Disruption and Vignetting...")

        img_bgr: np.ndarray = context["image"].copy()
        h, w, c = img_bgr.shape

        # 🚀 MACRO-VECTOR COLOR SPACE ATTACK (Vignette & Channel Shift)
        # Injects a sub-visual color offset (+4 to Red) to distort original color vector expectations
        img_bgr = img_bgr.astype(np.int16)
        img_bgr[:, :, 2] += 4  # Channel index 2 maps to Red inside OpenCV BGR configurations
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

        # Construct a non-linear radial Gaussian vignette mask to disrupt monotonic fills
        x_kernel: np.ndarray = cv2.getGaussianKernel(w, w / 2)
        y_kernel: np.ndarray = cv2.getGaussianKernel(h, h / 2)
        kernel: np.ndarray = y_kernel * x_kernel.T
        mask: np.ndarray = kernel / kernel.max()

        # Scale vignette blend parameters to maintain crisp lighting layout (base offset: 0.85)
        vignette_mask: np.ndarray = 0.85 + 0.15 * mask
        for i in range(c):
            img_bgr[:, :, i] = np.clip(img_bgr[:, :, i] * vignette_mask, 0, 255).astype(np.uint8)

        # Execute Floyd-Steinberg error diffusion quantization loop via PIL engine
        self.log(f"Enforcing color quantization constraints to {self.max_colors} bit-indices...")
        img_rgb: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil: Image.Image = Image.fromarray(img_rgb)
        img_quantized: Image.Image = img_pil.quantize(
            colors=self.max_colors,
            method=Image.Quantize.MAXCOVERAGE,
            dither=Image.Dither.FLOYDSTEINBERG
        ).convert("RGB")

        # Re-build coordinates array for sub-pixel high-frequency phase displacement
        img_bgr_q: np.ndarray = cv2.cvtColor(np.array(img_quantized), cv2.COLOR_RGB2BGR)
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        # Generate random uniform distortion matrices smoothed with a Gaussian kernel
        dx: np.ndarray = cv2.GaussianBlur(np.random.uniform(-self.shift, self.shift, (h, w)).astype(np.float32), (3, 3),
                                          0)
        dy: np.ndarray = cv2.GaussianBlur(np.random.uniform(-self.shift, self.shift, (h, w)).astype(np.float32), (3, 3),
                                          0)

        map_x: np.ndarray = (x + dx).astype(np.float32)
        map_y: np.ndarray = (y + dy).astype(np.float32)

        # Finalize array re-mapping using Lanczos-4 sub-pixel interpolation
        final_bgr: np.ndarray = cv2.remap(
            img_bgr_q, map_x, map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )

        self.log("Purging 100% of underlying binary stream metadata profiles...")

        final_rgb: np.ndarray = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

        # Unroll NumPy pixel array cleanly into an explicit Python list of raw integer tuples
        final_list: List[tuple] = [tuple(p) for p in final_rgb.reshape(-1, 3).tolist()]

        # Inject pixel list into a completely sterile container to strip C2PA, EXIF, and XMP layers
        clean_container: Image.Image = Image.new("RGB", (w, h))
        clean_container.putdata(final_list)
        clean_container.save(context["output_path"], "PNG")
        self.log("Global color modulation attack finalized successfully.")
        return True

