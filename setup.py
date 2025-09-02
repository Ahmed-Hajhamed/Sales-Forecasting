"""
Setup script for Sales Forecasting package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sales-forecasting",
    version="1.0.0",
    author="Sales Forecasting Team",
    author_email="sales-forecasting@example.com",
    description="A comprehensive sales forecasting solution using time series analysis and machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ahmed-Hajhamed/Sales-Forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sales-forecast=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
