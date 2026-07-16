# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import taccl.collectives as collectives
from taccl.serialization import *
from pathlib import Path
import sys

class KnownCollectives:
    def __init__(self, parser):
        self.parser = parser
        self.constructors = {
            'Broadcast': self._rooted_coll(collectives.broadcast),
            'Reduce': self._rooted_coll(collectives.reduce),
            'Scatter': self._rooted_coll(collectives.scatter),
            'Gather': self._rooted_coll(collectives.gather),
            'Allgather': self._coll(collectives.allgather),
            'Allreduce': self._coll(collectives.allreduce),
            'Alltoall': self._coll(collectives.alltoall),
            'ReduceScatter': self._coll(collectives.reduce_scatter),
            'Scan': self._coll(collectives.scan),
            'MultirootBroadcast': self._multiroot_coll(collectives.multiroot_broadcast),
            'MultirootScatter': self._multiroot_coll(collectives.multiroot_scatter),
            'MultirootGather': self._multiroot_coll(collectives.multiroot_gather),
            'custom': self._custom_coll(),
            'Sub_Allgather': self._sub_coll(collectives.sub_allgather),
            'Sub_Allreduce': self._sub_coll(collectives.sub_allreduce),
            'Sub_Reduce': self._sub_rooted_coll(collectives.sub_reduce),
            'Sub_Broadcast': self._sub_rooted_coll(collectives.sub_broadcast),
        }

        self.constructors_sub = {
            'Sub_Allgather': self._subcoll(collectives.sub_allgather),
            'Sub_Allreduce': self._subcoll(collectives.sub_allreduce),
            'Sub_Reduce': self._subrooted_coll(collectives.sub_reduce),
            'Sub_Broadcast': self._subrooted_coll(collectives.sub_broadcast)
        }

        self.parser.add_argument('collective', type=str, choices=self.constructors.keys(), help='collective')
        self.parser.add_argument('--collective-file', type=Path, default=None, help='a serialized collective', metavar='FILE')
        self.parser.add_argument('--root', type=int, default=0, help='used by rooted collectives', metavar='N')
        self.parser.add_argument('--group', type=int, nargs='+', default=[0], help='used by sub_collective group', metavar='N')
        self.parser.add_argument('--roots', type=int, nargs='+', default=[0], help='used by multi-rooted collectives', metavar='N')

    def create(self, args, num_nodes):
        return self.constructors[args.collective](num_nodes, args)
    
    def create_sub_coll(self, args, num_nodes, sub_coll, group):
        return self.constructors_sub[sub_coll](num_nodes, group, args)

    def _custom_coll(self):
        def make(size, args):
            input_file = args.collective_file
            if input_file is None:
                self.parser.error('--collective-file is required for custom collectives')
                exit(1)

            if not input_file.exists():
                print(f'error: input file not found: {input_file}', file=sys.stderr)
                exit(1)

            return load_sccl_object(input_file)
        return make

    def _rooted_coll(self, fun):
        def make(size, args):
            root = args.root
            return fun(size, root)
        return make

    def _coll(self, fun):
        def make(size, args):
            return fun(size)
        return make

    def _multiroot_coll(self, fun):
        def make(size, args):
            roots = args.roots
            return fun(size, roots)
        return make

    def _sub_coll(self, fun):
        def make(size, args):
            group = args.group
            return fun(size, group)
        return make
    
    def _sub_rooted_coll(self, fun):
        def make(size, args):
            group = args.group
            root = args.root
            return fun(size, group, root)
        return make
    

    def _subcoll(self, fun):
        def make(size, group, args):
            return fun(size, group)
        return make
    
    def _subrooted_coll(self, fun):
        def make(size, group, args):
            root = args.root
            return fun(size, group, root)
        return make