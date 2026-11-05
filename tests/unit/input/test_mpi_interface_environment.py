from vericcl.execution.security import Redactor
from vericcl.input.loader import load_request
from tests.unit.input.v5_support import request_documents, write_request


def test_explicit_mpi_data_and_control_interfaces_survive_input_and_evidence(tmp_path):
    documents = request_documents()
    interfaces = {'OMPI_MCA_btl_tcp_if_include': '192.0.2.0/24',
                  'OMPI_MCA_oob_tcp_if_include': '192.0.2.0/24'}
    documents['runtime']['environment'].update(interfaces)
    inputs = load_request(write_request(tmp_path, documents))
    recorded = Redactor().value({'environment': inputs.request_config['runtime']['environment']})['environment']
    for key, value in interfaces.items():
        assert inputs.request_config['runtime']['environment'][key] == value
        assert recorded[key] == value
