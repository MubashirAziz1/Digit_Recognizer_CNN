"""
Setup script for the digit recognizer project.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="digit-recognizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A deep learning project for handwritten digit recognition using CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/digit-recognizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "digit-train=scripts.train:main",
            "digit-predict=scripts.predict:main",
            "digit-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
    },
    keywords="machine-learning deep-learning cnn pytorch mnist digit-recognition",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/digit-recognizer/issues",
        "Source": "https://github.com/yourusername/digit-recognizer",
        "Documentation": "https://github.com/yourusername/digit-recognizer#readme",
    },
) 