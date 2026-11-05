"""Tiny independent XML interpreter, using sets of original contributions."""
import xml.etree.ElementTree as ET


def expected_outputs(operator, ranks, slices, root=0):
    outputs = {}
    for rank in range(ranks):
        if operator == 'reduce' and rank != root:
            continue
        count = slices * ranks if operator == 'allgather' else slices // ranks if operator == 'reduce_scatter' else slices
        for offset in range(count):
            if operator == 'broadcast':
                contributors = {root * slices + offset}
            elif operator in {'reduce', 'allreduce', 'reduce_scatter'}:
                logical = offset + (rank * (slices // ranks) if operator == 'reduce_scatter' else 0)
                contributors = {source * slices + logical for source in range(ranks)}
            elif operator == 'allgather':
                contributors = {offset}
            elif operator == 'alltoall':
                part = slices // ranks
                contributors = {(offset // part) * slices + rank * part + offset % part}
            else:
                raise AssertionError(operator)
            outputs[rank, offset] = frozenset(contributors)
    return outputs


def replay_xml_contributions(path, operator, *, ranks, slices, root=0):
    xml = ET.parse(path).getroot()
    assert int(xml.get('ngpus')) == ranks and xml.get('coll') == operator
    inplace = xml.get('inplace') == '1'
    def address(rank, buffer, offset):
        assert offset >= 0
        if inplace and buffer in {'i', 'o'}:
            if operator == 'allgather' and buffer == 'i':
                offset += rank * slices
            if operator == 'reduce_scatter' and buffer == 'o':
                offset += rank * (slices // ranks)
            buffer = 'alias'
        return rank, buffer, offset
    storage = {address(rank, 'i', offset): frozenset({rank * slices + offset})
               for rank in range(ranks) for offset in range(slices)}
    blocks = {}; positions = {}; completed = set()
    for gpu in xml.findall('gpu'):
        rank = int(gpu.get('id'))
        for block in gpu.findall('tb'):
            key = rank, int(block.get('id'))
            assert key not in blocks
            blocks[key] = block; positions[key] = 0
    def head(key):
        steps = blocks[key].findall('step'); index = positions[key]
        if index == len(steps):
            return None
        step = steps[index]; dep = int(step.get('depid')); dep_step = int(step.get('deps'))
        assert int(step.get('s')) == index and int(step.get('cnt')) == 1
        return step if dep < 0 or (key[0], dep, dep_step) in completed else None
    def mark(key):
        completed.add((*key, positions[key])); positions[key] += 1
    def read(rank, step):
        return storage[address(rank, step.get('srcbuf'), int(step.get('srcoff')))]
    def write(rank, step, value):
        storage[address(rank, step.get('dstbuf'), int(step.get('dstoff')))] = value
    total = sum(len(block.findall('step')) for block in blocks.values())
    while len(completed) < total:
        previous = len(completed)
        for key, block in blocks.items():
            step = head(key)
            if step is None:
                continue
            kind = step.get('type')
            if kind == 'nop':
                mark(key)
            elif kind == 'cpy':
                write(key[0], step, read(key[0], step)); mark(key)
            elif kind == 's':
                matches = [peer for peer, candidate in blocks.items()
                           if peer[0] == int(block.get('send')) and int(candidate.get('recv')) == key[0]
                           and candidate.get('chan') == block.get('chan')]
                assert len(matches) == 1
                peer = matches[0]; receive = head(peer)
                if receive is None or receive.get('type') not in {'r', 'rrc'}:
                    continue
                value = read(key[0], step)
                if receive.get('type') == 'rrc':
                    old = read(peer[0], receive)
                    assert old.isdisjoint(value), 'a contribution was reduced twice'
                    assert len({item % slices for item in old | value}) == 1
                    value = old | value
                write(peer[0], receive, value); mark(key); mark(peer)
            else:
                assert kind in {'r', 'rrc'}, kind
        assert len(completed) > previous, 'XML endpoint/dependency deadlock'
    expected = expected_outputs(operator, ranks, slices, root)
    actual = {(rank, offset): storage[address(rank, 'o', offset)] for rank, offset in expected}
    assert actual == expected, (actual, expected)
    return actual
