import pytest

from vericcl.semantics.collective import CollectiveKind

from tests.hardware._support import (
    CALIBRATION_BYTES,
    hardware_config,
    nccl_request,
    parse_xml_contract,
    runner,
    runtime_environment,
    xml_directory,
)


pytestmark = [pytest.mark.phase06, pytest.mark.hardware]


def test_intra_node_128_mib_calibration_xmls():
    config = hardware_config(minimum_gpu_count=2, require_hostfile=False)
    directory = xml_directory("VERICCL_INTRA_CALIBRATION_XML_DIR")
    xmls = tuple(sorted(directory.glob("*.xml")))
    if not xmls:
        pytest.skip("VERICCL_INTRA_CALIBRATION_XML_DIR contains no XML files")
    artifacts = tuple(
        sorted(
            (
                (int(contract["nchannels"]), xml_path, contract)
                for xml_path in xmls
                for contract in (parse_xml_contract(xml_path),)
            ),
            key=lambda item: item[0],
        )
    )
    assert tuple(item[0] for item in artifacts) == tuple(
        range(1, len(artifacts) + 1)
    )
    for _, xml_path, contract in artifacts:
        assert contract["coll"] == CollectiveKind.BROADCAST.value
        assert int(contract["minBytes"]) == CALIBRATION_BYTES
        assert int(contract["maxBytes"]) == CALIBRATION_BYTES + 1
        environment = runtime_environment(config, xml_path)
        history = runner(config, environment, process_count=2).measure(
            nccl_request(
                config,
                CollectiveKind.BROADCAST,
                CALIBRATION_BYTES,
                inplace=False,
            )
        )
        assert history.rounds
