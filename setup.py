"""
Setup script for Polytope Discovery & Hierarchical SAE Integration project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="polytope-hsae",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Polytope Discovery & Hierarchical SAE Integration Research Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/polytope-hsae",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "bitsandbytes>=0.41.0",
            "flash-attn>=2.0.0",
        ],
        "viz": [
            "umap-learn>=0.5.3",
            "bokeh>=3.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "polytope-experiment=polytope_hsae.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "polytope_hsae": ["configs/*.yaml"],
    },
)