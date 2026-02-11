"""
Setup script for kidney-tumor-classification package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kidney-tumor-classification",
    version="1.0.0",
    author="Wen-Yi Li, Jia-Yang Peng, Bor-Wen Cheng, Wei-Shiung Yang, Feng-Jung Yang",
    author_email="fongrong@ntu.edu.tw",
    description="Deep Learning-Based Kidney Tumor Subtype Classification from CT Imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/kidney-tumor-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
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
            "pytest>=7.3.1",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kidney-preprocess=scripts.preprocess_data:main",
            "kidney-train-stage1=scripts.train_stage1:main",
            "kidney-inference=scripts.inference:main",
        ],
    },
)
