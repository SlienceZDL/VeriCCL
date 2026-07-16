# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="vericcl",
    version="0.1.0",
    packages=find_packages(include=["vericcl", "vericcl.*"]),
    entry_points={
        "console_scripts": [
            "vericcl=vericcl.cli.main:console_main",
        ],
    },
    install_requires=[
        "argcomplete",
        "gurobipy",
        "lxml",
        "numpy",
        "ply",
        "z3-solver",
    ],
    python_requires=">=3.9",
)
