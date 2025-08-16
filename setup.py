from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except Exception:
    long_description = "Limestone Layer Analysis CLI"

setup(
    name="limestone-layer-analysis",
    version="0.1.0",
    description="CLI for analyzing and filling limestone layer properties via IDW + trend.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Limestone Tools",
    license="MIT",
    packages=find_packages(exclude=("notebook", "image", "output", "input")),
    python_requires=">=3.8",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "limestone=script.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
