"""Generate an illustrative two-server topology with eight V100-style GPUs each.

Ranks 0-7 belong to host-a and ranks 8-15 to host-b. Adapt NV, NIC_BW,
and the shared resources to the target hardware before GPU execution.
All costs are initial estimates, with alpha in us and beta in us/byte:
- NV2 and NV1 links use 45 and 22 GB/s per direction.
- GPU pairs without NVLink use 9 GB/s PCIe paths and share 12 GB/s uplinks.
- GPUs 0-3 use NIC0; GPUs 4-7 use NIC2. The example includes one slower NIC
  to illustrate asymmetric inter-host capacity.
- Per-step alpha is 7.0 us for NVLink, 7.7 us for PCIe, and 8.9 us between
  hosts. Online calibration can replace these estimates.
- A 14 GB/s host-staging resource limits combined incoming and outgoing
  inter-host traffic, representing a configuration without GPUDirect RDMA.
"""
import json
import sys

NV = {  # illustrative layout, identical on both hosts
    (0, 1): 2, (0, 2): 2, (0, 3): 1, (0, 4): 1,
    (1, 2): 1, (1, 3): 1, (1, 5): 2,
    (2, 3): 2, (2, 6): 1,
    (3, 7): 2,
    (4, 5): 2, (4, 6): 2, (4, 7): 1,
    (5, 6): 1, (5, 7): 1,
    (6, 7): 2,
}
NV.update({(b, a): v for (a, b), v in list(NV.items())})

GBPS = 1000.0  # bytes per microsecond per GB/s
NVLINK_BW = {2: 45 * GBPS, 1: 22 * GBPS}
PCIE_P2P_BW, PCIE_UPLINK_BW = 9 * GBPS, 12 * GBPS
NIC_BW = {  # (host, nic) -> bytes/us
    (0, 0): 11.5 * GBPS, (0, 2): 11.5 * GBPS,
    (1, 0): 6.0 * GBPS, (1, 2): 11.5 * GBPS,
}
HOST_NAMES = {0: 'host-a', 1: 'host-b'}
HOST_STAGING_BW = 14 * GBPS  # tx + rx inter-host bytes per host (initial estimate)


def switch(local):
    return 0 if local < 4 else 1


def nic(local):
    return 0 if local < 4 else 2


def rank(host, local):
    return host * 8 + local


def build():
    links, resources = [], {}

    def resource(name, kind, capacity):
        resources[name] = {'id': name, 'kind': kind, 'capacity_bytes_per_us': capacity}
        return name

    def staging(host):
        return resource('h%d-staging' % host, 'other', HOST_STAGING_BW)

    for host in (0, 1):
        for sw in (0, 1):
            resource('h%d-sw%d-up' % (host, sw), 'pcie', PCIE_UPLINK_BW)
            resource('h%d-sw%d-down' % (host, sw), 'pcie', PCIE_UPLINK_BW)
        for n in (0, 2):
            resource('h%d-nic%d-tx' % (host, n), 'nic', NIC_BW[host, n])
            resource('h%d-nic%d-rx' % (host, n), 'nic', NIC_BW[host, n])
        for a in range(8):
            for b in range(8):
                if a == b:
                    continue
                gpu_a, gpu_b = '%s-gpu%d' % (HOST_NAMES[host], a), '%s-gpu%d' % (HOST_NAMES[host], b)
                if (a, b) in NV:
                    links.append({'src': rank(host, a), 'dst': rank(host, b), 'alpha_us': 7.0,
                                  'beta_us_per_byte': 1.0 / NVLINK_BW[NV[a, b]], 'max_channels': 16,
                                  'resources': [], 'physical_path': [gpu_a, 'nv%d' % NV[a, b], gpu_b]})
                else:
                    links.append({'src': rank(host, a), 'dst': rank(host, b), 'alpha_us': 7.7,
                                  'beta_us_per_byte': 1.0 / PCIE_P2P_BW, 'max_channels': 16,
                                  'resources': ['h%d-sw%d-up' % (host, switch(a)), 'h%d-sw%d-down' % (host, switch(b))],
                                  'physical_path': [gpu_a, 'pcie-sw%d' % switch(a), 'host-bridge',
                                                    'pcie-sw%d' % switch(b), gpu_b]})
    for src_host, dst_host in ((0, 1), (1, 0)):
        for a in range(8):
            for b in range(8):
                tx, rx = (src_host, nic(a)), (dst_host, nic(b))
                links.append({'src': rank(src_host, a), 'dst': rank(dst_host, b), 'alpha_us': 8.9,
                              'beta_us_per_byte': 1.0 / min(NIC_BW[tx], NIC_BW[rx]), 'max_channels': 16,
                              'resources': ['h%d-nic%d-tx' % tx, 'h%d-nic%d-rx' % rx,
                                            staging(src_host), staging(dst_host)],
                              'physical_path': ['%s-gpu%d' % (HOST_NAMES[src_host], a), '%s-nic%d' % (HOST_NAMES[src_host], tx[1]),
                                                'ib-fabric', '%s-nic%d' % (HOST_NAMES[dst_host], rx[1]),
                                                '%s-gpu%d' % (HOST_NAMES[dst_host], b)]})
    return {
        'name': 'Illustrative 2x8 V100-style topology: hybrid cube-mesh NVLink, PCIe host-bridge P2P, NIC0/NIC2 per host',
        'ranks': 16,
        'nodes': [{'id': 0, 'ranks': list(range(8)), 'gateways': list(range(8))},
                  {'id': 1, 'ranks': list(range(8, 16)), 'gateways': list(range(8, 16))}],
        'directed_links': links,
        'shared_resources': sorted(resources.values(), key=lambda item: item['id']),
        'local_costs': [{'rank': r, 'copy': {'alpha_us': 0.3, 'beta_us_per_byte': 1.0 / 350000},
                         'reduce': {'alpha_us': 0.5, 'beta_us_per_byte': 1.0 / 200000}} for r in range(16)],
    }


if __name__ == '__main__':
    json.dump(build(), open(sys.argv[1], 'w') if len(sys.argv) > 1 else sys.stdout, indent=1)
