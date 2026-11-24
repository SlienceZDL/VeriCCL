import pytest

from vericcl.semantics.collective import CollectiveKind
from vericcl.verification.online.nccl_tests import parse_nccl_tests_output

from tests.hardware._support import (
    hardware_config,
    nccl_request,
    parse_xml_contract,
    runner,
    runtime_environment,
    xml_directory,
)


pytestmark = [pytest.mark.phase06, pytest.mark.hardware]


@pytest.mark.parametrize(
    "kind",
    (
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.ALL_TO_ALL,
        CollectiveKind.REDUCE_SCATTER,
    ),
)
def test_release_smoke_for_each_executable_collective(kind):
    config = hardware_config(minimum_gpu_count=2, require_hostfile=False)
    directory = xml_directory("VERICCL_SIX_COLLECTIVE_XML_DIR")
    xml_path = directory / "{}.xml".format(kind.value)
    contract = parse_xml_contract(xml_path)
    assert contract["coll"] == kind.value
    message_size = int(contract["minBytes"])
    assert int(contract["maxBytes"]) == message_size + 1
    inplace = contract["inplace"] == "1"
    environment = runtime_environment(config, xml_path)
    request = nccl_request(
        config,
        kind,
        message_size,
        inplace=inplace,
    )
    process = runner(config, environment, process_count=2).diagnostic(
        request,
        environment,
    )
    rows = parse_nccl_tests_output(process.stdout, message_size)
    assert len(rows) == 1
    rows[0].selected_time_us(inplace=inplace)
