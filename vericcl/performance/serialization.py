"""Lossless public cost-model load, shared by checkpoints and worker inputs."""
from dataclasses import fields
from pathlib import Path

from vericcl.errors import InputValidationError, SemanticError
from vericcl.performance.model import CostModel, ServiceCost
from vericcl.performance.profiles import RateProfile


def _fields(value, type_, *, required=()):
    if not isinstance(value, dict) or set(value) - {item.name for item in fields(type_)} or set(required) - value.keys():
        raise InputValidationError('invalid serialized ' + type_.__name__ + ' fields')
    return value


def model_from_json(document):
    try:
        _fields(document, CostModel, required=('version', 'services'))
        if not isinstance(document['services'], dict):
            raise InputValidationError('serialized model services must be an object')
        services = {}
        for key, values in document['services'].items():
            if not isinstance(values, (list, tuple)):
                raise InputValidationError('serialized service pieces must be a list')
            services[key] = tuple(ServiceCost(**_fields(value, ServiceCost, required=('alpha_us', 'beta_us_per_byte'))) for value in values)
        profiles = tuple(RateProfile(**_fields(value, RateProfile, required=('profile_id', 'mode', 'rates')))
                         for value in document.get('profiles', ()))
        return CostModel(document['version'], services, document.get('resources', {}), profiles,
                         document.get('context', {}), document.get('metadata', {}))
    except InputValidationError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, SemanticError) as error:
        raise InputValidationError('invalid serialized cost model: ' + str(error)) from error


def load_model(path):
    from vericcl.input.loader import _load_json_object
    return model_from_json(_load_json_object(Path(path), 'cost model'))
