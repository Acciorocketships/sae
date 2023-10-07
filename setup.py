from setuptools import setup
from setuptools import find_packages

setup(
    name="sae",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["torch", "scipy", "wandb", "plotly"],
    author="Ryan Kortvelesy",
    author_email="rk627@cam.ac.uk",
    description="A Set Autoencoder. Functions as a neural network that can produce a variable number of outputs.",
)
