# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="vericcl",
    version="0.1.0",
    packages=find_packages(include=["vericcl", "vericcl.*"]),
    package_data={
        "vericcl": [
            "examples/atom/*.json",
            "examples/legacy/*.xml",
            "examples/legacy/sketch/*.json",
            "examples/legacy/topo/*.json",
            "examples/sketch/*.json",
            "examples/templates/*/*.json",
            "examples/topo/*.json",
        ]
    },
    data_files=[
        (
            "share/vericcl/runtime/msccl-trace",
            ["runtime/msccl-trace/README.md"],
        ),
        (
            "share/vericcl/runtime/msccl-trace/include",
            ["runtime/msccl-trace/include/vericcl_trace_format.h"],
        ),
        (
            "share/vericcl/runtime/msccl-trace/patches",
            [
                "runtime/msccl-trace/patches/"
                "0001-vericcl-fixed-step-trace.patch"
            ],
        ),
        (
            "share/vericcl/runtime/msccl-trace/tools",
            [
                "runtime/msccl-trace/tools/vericcl_clock_sync.cu",
                "runtime/msccl-trace/tools/verify_patch.py",
            ],
        ),
    ],
    entry_points={
        "console_scripts": [
            "vericcl=vericcl.cli.main:console_main",
        ],
    },
    install_requires=[
        "argcomplete",
        "dd>=0.5.7,<0.6",
        "gurobipy",
        "lxml",
        "numpy",
        "ply",
        "z3-solver",
    ],
    python_requires=">=3.9",
)
