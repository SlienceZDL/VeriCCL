from types import MappingProxyType
from typing import Mapping


LEGACY_TACCL_TOPOLOGY_FORMAT = "taccl_topology_v2"

ALLOWED_TACCL_REFERENCES: Mapping[str, str] = MappingProxyType(
    {
        "vericcl/provenance.py": (
            "Retains the exact external legacy topology format identifier."
        ),
    }
)
