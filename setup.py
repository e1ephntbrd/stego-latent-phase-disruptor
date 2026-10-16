from setuptools import setup, find_packages

setup(
    name="stego-latent-phase-disruptor",
    version="1.0.0",
    author="Your Name",
    description="A modular framework for structural steganography analysis and latent phase de-synchronization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "diffusers>=0.30.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "opencv-python>=4.9.0",
        "numpy>=1.24.0",
        "scipy>=1.12.0",
        "Pillow>=10.2.0",
    ],
    entry_points={
        "console_scripts": [
            "stego-disrupt=main:main",  # Дозволить запускати конвеєр прямо з терміналу командою stego-disrupt
        ],
    },
)
