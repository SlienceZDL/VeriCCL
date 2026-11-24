"""Fixed-membership version and request decisions shared by live controllers."""
from dataclasses import dataclass
from threading import RLock

from vericcl.verification.search_coverage import digest


class ProtocolError(ValueError):
    pass


def _identity(value):
    if not isinstance(value, str) or not value or len(value) > 256 or '\x00' in value:
        raise ProtocolError('invalid protocol identity')
    return value


@dataclass(frozen=True)
class Version:
    version_id: str
    xml_sha256: str
    contract_digest: str


class GroupProtocol:
    def __init__(self, group_id, members, contract_digest):
        self.group_id, self.contract = _identity(group_id), _identity(contract_digest)
        self.members = tuple(members)
        if not self.members or len(set(self.members)) != len(self.members) or any(type(rank) is not int or rank < 0 for rank in self.members):
            raise ProtocolError('membership must contain distinct nonnegative ranks')
        self.versions, self.ready_ranks = {}, {}
        self.decision = None
        self.commit_acks = set()
        self.request = None
        self.completions = set()
        self.last_completed = None
        self.last_successful_request = 0
        self.logical_successes = 0
        self.state, self.reason = 'idle', None
        self.lock = RLock()

    def _available(self):
        if self.state in {'uncertain', 'stopped'}:
            raise ProtocolError('group is stopped pending explicit reconciliation: ' + str(self.reason))

    def _rank(self, rank):
        if type(rank) is not int or rank not in self.members:
            raise ProtocolError('message rank is outside fixed membership')

    def prepare(self, version_id, xml_sha256, contract_digest):
        with self.lock:
            self._available()
            version = Version(_identity(version_id), _identity(xml_sha256), _identity(contract_digest))
            if version.contract_digest != self.contract:
                raise ProtocolError('prepared version changes the external collective contract')
            if version_id in self.versions and self.versions[version_id] != version:
                raise ProtocolError('version identity is immutable')
            self.versions[version_id] = version
            self.ready_ranks.setdefault(version_id, set())
            return {'version': version_id, 'xml_sha256': xml_sha256, 'contract': self.contract}

    def ready(self, rank, version_id, xml_sha256):
        with self.lock:
            self._available(); self._rank(rank)
            version = self.versions.get(version_id)
            if version is None or version.xml_sha256 != xml_sha256:
                raise ProtocolError('READY does not identify the prepared immutable version')
            self.ready_ranks[version_id].add(rank)
            return {'all_ready': self.ready_ranks[version_id] == set(self.members)}

    def commit(self, version_id, *, request_id):
        with self.lock:
            self._available()
            if type(request_id) is not int or request_id < 1:
                raise ProtocolError('invalid request boundary')
            if self.decision is not None and (self.decision['version'], self.decision['request_id']) == (version_id, request_id):
                return dict(self.decision)
            if self.request is not None or request_id != self.last_successful_request + 1:
                raise ProtocolError('COMMIT must occur at the next resolved request boundary')
            if version_id not in self.versions or self.ready_ranks[version_id] != set(self.members):
                raise ProtocolError('COMMIT requires READY from every fixed member')
            value = {'group_id': self.group_id, 'version': version_id, 'request_id': request_id,
                     'xml_sha256': self.versions[version_id].xml_sha256, 'contract': self.contract}
            self.decision = value | {'decision_id': digest(value)}
            self.commit_acks.clear()
            self.state = 'committing'
            return dict(self.decision)

    def acknowledge_commit(self, rank, decision_id):
        with self.lock:
            self._available(); self._rank(rank)
            if self.decision is None or self.decision['decision_id'] != decision_id:
                raise ProtocolError('COMMIT acknowledgement belongs to another decision')
            self.commit_acks.add(rank)
            if self.commit_acks == set(self.members):
                self.state = 'committed'
            return {'all_acknowledged': self.commit_acks == set(self.members)}

    def start(self, request_id, *, attempt):
        with self.lock:
            self._available()
            if type(request_id) is not int or type(attempt) is not int or request_id < 1 or attempt != 0:
                raise ProtocolError('a new logical request starts at attempt zero; recovery requires its own protocol')
            if self.request is not None:
                if (self.request['request_id'], self.request['attempt']) == (request_id, attempt):
                    return dict(self.request)
                raise ProtocolError('one unfinished request is already owned by this group')
            if self.decision is None or self.commit_acks != set(self.members):
                raise ProtocolError('START requires all COMMIT acknowledgements')
            if request_id != self.last_successful_request + 1 or request_id < self.decision['request_id']:
                raise ProtocolError('START is stale or skips an unresolved request boundary')
            self.request = dict(self.decision) | {'request_id': request_id, 'attempt': attempt}
            self.completions.clear()
            self.state = 'running'
            return dict(self.request)

    def complete(self, rank, request_id, attempt, *, gpu_completed, output_valid):
        with self.lock:
            self._available(); self._rank(rank)
            if gpu_completed is not True or output_valid is not True:
                raise ProtocolError('success requires actual GPU completion and a valid output')
            if self.request is None:
                if self.last_completed == (request_id, attempt):
                    return self.status()
                raise ProtocolError('completion is not for an owned request')
            if (self.request['request_id'], self.request['attempt']) != (request_id, attempt):
                raise ProtocolError('completion identifies another request or retry')
            self.completions.add(rank)
            if self.completions == set(self.members):
                self.last_successful_request = request_id
                self.last_completed = (request_id, attempt)
                self.logical_successes += 1
                self.request = None
                self.state = 'idle'
            return self.status()

    def uncertain(self, reason):
        with self.lock:
            self.reason, self.state = _identity(reason), 'uncertain'
            return self.status()

    def status(self):
        with self.lock:
            return {'group_id': self.group_id, 'state': self.state, 'reason': self.reason,
                'committed_version': None if self.decision is None else self.decision['version'],
                'decision_id': None if self.decision is None else self.decision['decision_id'],
                'commit_acknowledgements': sorted(self.commit_acks),
                'unfinished_request': None if self.request is None else self.request['request_id'],
                'last_successful_request': self.last_successful_request, 'logical_successes': self.logical_successes}
