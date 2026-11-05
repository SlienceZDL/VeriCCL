"""Generated candidates and independently discharged search branches are distinct."""
from dataclasses import asdict
import hashlib
import json

from vericcl.errors import SemanticError


REPAIR_FAMILIES = ('path', 'allocation', 'order', 'channel', 'joint', 'internal_root', 'structure', 'partition')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def model_digest(model):
    return digest({'version': model.version,
        'services': {key: [asdict(cost) for cost in costs] for key, costs in model.services.items()},
        'resources': dict(model.resources), 'profiles': [profile.as_json() for profile in model.profiles],
        'context': dict(model.context)})


class SearchCoverage:
    def __init__(self, scope_id, model, families=REPAIR_FAMILIES):
        if not scope_id or not families or len(set(families)) != len(families):
            raise SemanticError('search coverage needs an identity and unique nonempty branches')
        self.scope_id, self.model_version, self.model_digest = scope_id, model.version, model_digest(model)
        self.branches = {name: {'generated': 0, 'exhausted': False, 'proof': None, 'scope_id': None} for name in families}
        self.filtered = self.duplicates = 0

    def declare_scope(self, family, scope_id):
        branch = self.branches[family]
        if branch['scope_id'] not in {None, scope_id}:
            raise SemanticError('a declared proof scope is immutable')
        branch['scope_id'] = scope_id

    def generated(self, family):
        self.branches[family]['generated'] += 1

    def exhausted(self, family):
        self.branches[family]['exhausted'] = True

    def close_branch(self, family, proof):
        branch = self.branches[family]
        if (proof.status != 'proven_no_improvement' or proof.model_version != self.model_version
                or proof.model_digest != self.model_digest or family not in proof.families
                or branch['scope_id'] is None or proof.scope_id != branch['scope_id']):
            raise SemanticError('proof does not discharge the declared current-model branch')
        branch['proof'] = proof.certificate_digest

    def snapshot(self):
        unresolved = [key for key, value in self.branches.items() if value['proof'] is None]
        return {'scope_id': self.scope_id, 'model_version': self.model_version, 'model_digest': self.model_digest,
            'branches': {key: dict(value) for key, value in self.branches.items()},
            'unresolved_branches': unresolved, 'generation_complete': all(v['exhausted'] for v in self.branches.values()),
            'proven_no_improvement': not unresolved, 'filtered': self.filtered, 'duplicates': self.duplicates}
