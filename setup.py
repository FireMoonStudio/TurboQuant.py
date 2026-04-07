from setuptools import setup, find_packages
import os


long_description = ""



if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="trb-quant",
    version="0.1.3",
    author="FireMoon Studio",
    description="A library for Near-optimal Vector Quantization based on arXiv:2504.19874",
    long_description=long_description,
    long_description_content_type="text/markdown", 
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0", 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
