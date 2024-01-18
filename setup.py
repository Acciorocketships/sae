from setuptools import setup
from setuptools import find_packages

setup(
    name="sae",
    version="1.1.0",
    packages=find_packages(),
    install_requires=["torch", "scipy", "wandb", "matplotlib", "torchvision", "tqdm"],
    author="Ryan Kortvelesy",
    author_email="rk627@cam.ac.uk",
    description="A Set Autoencoder. Defines a bijective mapping between variable-sized inputs and a fixed-sized embedding.",
)
