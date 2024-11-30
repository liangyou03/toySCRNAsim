# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="toySCRNAsim",
    version="0.1.0",
    author="Liang You",
    author_email="liangyou03@gmail.com",
    description="A toy example of scRNA-seq data simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liangyou03/toySCRNAsim",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
