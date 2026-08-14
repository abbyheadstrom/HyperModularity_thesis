from setuptools import setup, find_packages

setup(
    name="comparisons",
    version="0.1.0",
    description="A comparison code project",
    author="Your Name",
    author_email="your.email@dartmouth.edu",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
)
