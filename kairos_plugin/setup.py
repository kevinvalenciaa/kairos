#!/usr/bin/env python
"""
Kairos - AI/ML Cost Intelligence Layer

Setup script for backwards compatibility with older pip versions.
For modern pip (>=21.0), the pyproject.toml is preferred.
"""

from setuptools import setup, find_packages
import os

# Read the README for long description
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.md")

long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="kairos",
    version="0.1.0",
    author="Kairos Team",
    author_email="team@usekairos.ai",
    description="AI/ML Cost Intelligence Layer - Real-time GPU cost tracking for Jupyter notebooks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://usekairos.ai",
    project_urls={
        "Documentation": "https://docs.usekairos.ai",
        "Repository": "https://github.com/usekairos/kairos",
        "Issues": "https://github.com/usekairos/kairos/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.8",
    install_requires=[
        "ipython>=7.0.0",
    ],
    extras_require={
        "gpu": [
            "pynvml>=11.0.0,<13.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "pynvml>=11.0.0,<13.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Framework :: Jupyter",
    ],
    keywords=[
        "machine-learning",
        "deep-learning",
        "gpu",
        "cost-tracking",
        "jupyter",
        "mlops",
        "finops",
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
