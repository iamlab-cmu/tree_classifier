from setuptools import setup, find_packages

setup(
    name="learning",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "hydra-core",
        "numpy",
        "librosa"
    ]
) 