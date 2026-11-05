"""Compare dispatch decisions against exhaustive small legal resource choices."""
from itertools import product
from random import Random

import pytest

from vericcl.solver.global_scheduler import assign_global_resources
from vericcl.solver.scheduling import fixed_topology_transfer_duration_us
from vericcl.topology.model import DirectedLink, LinkKey, SharedResource, Topology
from tests.unit.solver.test_global_scheduler import _schedule, _transfer
from tests.unit.verification.simulator_helpers import curve


def exhaustive_dispatch(schedule, topology, channel_count, hierarchy):
    """Slow oracle: enumerate every ready transfer, lane, and slot product."""
    transfers = {t.transfer_id: t for t in schedule.transfers}
    durations = {key: fixed_topology_transfer_duration_us(topology,
        LinkKey(t.src_rank, t.dst_rank), t.physical_bytes, channel_count)
        for key, t in transfers.items()}
    predecessors = {key: set(t.predecessor_ids) for key, t in transfers.items()}
    successors = {key: set() for key in transfers}
    for key, previous in predecessors.items():
        for p in previous:
            successors[p].add(key)

    def critical(key):
        return durations[key] + max((critical(s) for s in successors[key]), default=0.)

    priorities = schedule.metadata['route_priorities']
    result, occupancy, last = {}, {}, {}
    while len(result) != len(transfers):
        choices = []
        for key, transfer in transfers.items():
            if key in result or not predecessors[key] <= result.keys():
                continue
            release = max((result[p][1] for p in predecessors[key]), default=0.)
            link = topology.links[LinkKey(transfer.src_rank, transfer.dst_rank)]
            resources = sorted(link.resource_ids)
            for channel in range(min(channel_count, link.max_channels)):
                lane = ('lane', transfer.src_rank, transfer.dst_rank, channel)
                assignments = product(*(range(min(channel_count,
                    topology.shared_resources[r].max_channels)) for r in resources))
                for values in assignments:
                    slots = tuple(zip(resources, values))
                    keys = (lane,) + tuple(('slot', r, s) for r, s in slots)
                    start = max([release] + [occupancy.get(k, 0.) for k in keys])
                    end = start + durations[key]
                    position = next(iter(transfer.member_slice_ids)) % schedule.slice_count
                    order = ((start, priorities[key], -critical(key), release,
                              transfer.stage_id, position, key, channel, slots) if hierarchy else
                             (end, release, priorities[key], transfer.stage_id, position,
                              transfer.src_rank, transfer.dst_rank, key, channel, slots))
                    choices.append((order, key, channel, start, end, slots, keys))
        _, key, channel, start, end, slots, keys = min(choices)
        dependencies = predecessors[key] | {last[k] for k in keys if k in last}
        result[key] = (start, end, channel, slots, frozenset(dependencies))
        for resource_key in keys:
            occupancy[resource_key], last[resource_key] = end, key
    return result


@pytest.mark.parametrize('hierarchy', (False, True))
@pytest.mark.parametrize('seed', range(20))
def test_dispatch_matches_exhaustive_resource_choices(hierarchy, seed):
    random = Random(seed)
    links = tuple(LinkKey(*pair) for pair in ((0, 1), (1, 2), (2, 0), (0, 2)))
    shared = {'nic': links, 'bus': links[:2]}

    def random_curve():
        alpha = random.choice((0., 1., 2.))
        return curve(alpha, alpha + random.choice((0., .25, .5)))

    topology = Topology(rank_count=3,
        links={link: DirectedLink(link, random.choice((1, 2, 3)),
            random_curve(),
            tuple(name for name, members in shared.items() if link in members)) for link in links},
        shared_resources={name: SharedResource(name, members, random.choice((1, 2, 3)),
            curve(0., 0.)) for name, members in shared.items()},
        node_membership={0: 0, 1: 0, 2: 0}, gateways=frozenset(), warnings=())
    transfers = []
    for index in range(18):
        link = random.choice(links)
        dependencies = tuple(t.transfer_id for t in transfers if random.random() < .12)
        transfer = _transfer('op-%03d' % index, slice_id=link.src_rank * 18 + index, slice_count=18,
            path=((random.randrange(3), 'SEND', ((link.src_rank, link.dst_rank, 0.),)),),
            predecessors=dependencies)
        transfers.append(transfer)
    random.shuffle(transfers)
    schedule = _schedule(transfers, rank_count=3, slice_count=18, metadata={
        'hierarchy_event_policy': hierarchy,
        'route_priorities': {t.transfer_id: random.randrange(3) for t in transfers}})
    expected = exhaustive_dispatch(schedule, topology, 3, hierarchy)
    actual = assign_global_resources(schedule, topology, 3)
    assert {t.transfer_id: (t.st_time, t.ed_time, t.channel,
        tuple(sorted(actual.metadata['resource_slots'][t.transfer_id].items())), t.predecessor_ids)
        for t in actual.transfers} == expected
