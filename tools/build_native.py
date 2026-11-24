"""Build the optional VeriCCL native core into vericcl/native/libvericcl_core.so.

Needs a C++20 compiler, CMake and GMP headers. Without a system libgmp-dev,
pass --gmp-root to an extracted copy (for example from
``apt-get download libgmp-dev`` and ``dpkg -x``); the default is
.deps/gmp-dev/root/usr when it exists.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_gmp = os.environ.get("VERICCL_GMP_ROOT") or (
        str(ROOT / ".deps/gmp-dev/root/usr") if (ROOT / ".deps/gmp-dev/root/usr").is_dir() else "")
    parser.add_argument("--gmp-root", default=default_gmp, help="prefix that holds include/gmp.h")
    parser.add_argument("--build-dir", default=str(ROOT / "native/build"))
    parser.add_argument("--build-type", default="Release", choices=("Release", "Debug", "RelWithDebInfo"))
    args = parser.parse_args(argv)
    build = Path(args.build_dir)
    configure = ["cmake", "-S", str(ROOT / "native"), "-B", str(build), "-DCMAKE_BUILD_TYPE=" + args.build_type]
    if shutil.which("ninja"):
        configure += ["-G", "Ninja"]
    if args.gmp_root:
        configure.append("-DVERICCL_GMP_ROOT=" + args.gmp_root)
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel"], check=True)
    built = build / "libvericcl_core.so"
    target = ROOT / "vericcl/native/libvericcl_core.so"
    temporary = target.with_suffix(".so.tmp")
    shutil.copy2(built, temporary)
    os.replace(temporary, target)
    print("native core:", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
