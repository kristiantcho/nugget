from setuptools import setup, find_packages

setup(
    name="nugget",
    version="2.0",
    description="NeUtrino experiement Geometry optimization and General Evaluation Tool",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "tqdm>=4.45.0",
        "conflictfree>=0.1.8",
        "imageio>=2.37.0",
    ],
    python_requires=">=3.8",
)
