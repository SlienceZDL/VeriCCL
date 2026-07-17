from pathlib import Path

import pytest

from vericcl.semantics.collective import CollectiveKind
from vericcl.topology import load_topology
from vericcl.xml.lower import lower_to_xml
from vericcl.xml.parser import normalize_xml

from tests.unit.xml.helpers import (
    resolved,
    two_rank_allgather_schedule,
    two_rank_allreduce_schedule,
)


pytestmark = pytest.mark.phase04

GOLDEN_ROOT = Path(__file__).parent / "xml"


@pytest.mark.parametrize(
    "filename,schedule,inputs",
    [
        (
            "two_rank_allreduce_out_of_place.xml",
            two_rank_allreduce_schedule(),
            resolved(
                CollectiveKind.ALL_REDUCE,
                ranks=2,
                slices=1,
                inplace=False,
            ),
        ),
        (
            "two_rank_allgather_in_place.xml",
            two_rank_allgather_schedule(),
            resolved(
                CollectiveKind.ALL_GATHER,
                ranks=2,
                slices=1,
                inplace=True,
            ),
        ),
    ],
)
def test_xml_matches_golden_without_reordering(filename, schedule, inputs):
    artifact = lower_to_xml(schedule, inputs, load_topology(inputs))
    expected = (GOLDEN_ROOT / filename).read_text(encoding="utf-8")

    assert normalize_xml(artifact.xml_text) == normalize_xml(expected)
    assert artifact.sha256
    assert artifact.runtime_compatible is True
