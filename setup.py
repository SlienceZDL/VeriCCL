# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.sdist import sdist


class PublicSourceDistribution(sdist):
    """Use neutral archive ownership metadata by default."""
    def initialize_options(self):
        super().initialize_options()
        self.owner = 'root'
        self.group = 'root'


def public_data_files():
    """Package only files declared in the public source inventory."""
    root = Path(__file__).resolve().parent
    entries = json.loads((root / 'public-files.json').read_text())
    if not isinstance(entries, list) or not all(isinstance(name, str) for name in entries):
        raise ValueError('public file inventory must be a list of paths')
    if len(entries) != len(set(entries)):
        raise ValueError('public file inventory contains duplicate paths')
    groups = {}
    roots = {'runtime', 'third_party', 'tools', 'requirements-lock', 'docs', 'tests', 'examples'}
    for name in entries:
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or not path.parts
                or set(path.parts) & {'output', 'tmp', '.deps', '.git', '__pycache__', 'rebuildable-cache'}
                or not (root / path).resolve().is_relative_to(root)
                or not (root / path).is_file()):
            raise ValueError('invalid public source path: ' + name)
        if name == 'README.md' or path.parts[0] in roots:
            groups.setdefault(str(Path('share/vericcl') / path.parent), []).append(name)
    return [(target, sorted(paths)) for target, paths in sorted(groups.items())]


setup(
    name='vericcl',
    version='0.5.0.dev0',
    packages=find_packages(include=['vericcl', 'vericcl.*']),
    include_package_data=False,
    data_files=public_data_files(),
    cmdclass={"sdist": PublicSourceDistribution},
    entry_points={'console_scripts': ['vericcl=vericcl.cli.main:console_main']},
    install_requires=['argcomplete', 'dd>=0.6,<0.7', 'gmpy2>=2.3.2,<2.4',
        'pyscipopt>=6.3,<7', 'lxml', 'numpy', 'networkx>=3.7,<4', 'ply', 'z3-solver'],
    python_requires='>=3.12,<3.14',
)
