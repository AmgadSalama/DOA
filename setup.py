"""
Setup script for DOA Methods Tutorial Repository
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="doa-methods",
    version="1.0.0",
    author="DOA Methods Tutorial",
    author_email="",
    description="Educational Python implementation of Direction of Arrival estimation methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/doa-methods",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=1.0.0",
        "cvxpy>=1.2.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "ipywidgets>=7.6.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "examples": [
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "doa-demo=examples.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)