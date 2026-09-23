"""Small content-checked cache for the immutable retained orientation chain.

Every hit rehashes all files used by the projection validator. Live image and
scan proofs are never cached. Editing any source forces full revalidation.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

_CACHE = OrderedDict()


def _dependencies(path):
    projection = json.loads(path.read_bytes())
    axis = Path(projection['source_axis_observation']['path'])
    paths = {path, axis}
    for key in ('source_candidate_frame_projection', 'target_candidate_frame_projection'):
        source = Path(projection[key]['path'])
        data = json.loads(source.read_bytes())
        paths.update((source, Path(data['source_candidate_snapshot_path']),
                      Path(data['projected_candidate_snapshot_path'])))
    receipt = json.loads(axis.read_bytes())
    if receipt.get('head_position_evidence') is not None:
        paths.add(Path(receipt['head_position_evidence']['model_path']))
    proof = receipt.get('target_reconciliation')
    if proof is not None:
        paths.add(Path(proof['snapshot_path']))
        # Backside sources cannot originate from the opposite identity branch.
        if proof.get('retained_orientation') is not None:
            raise ValueError('backside source cannot recursively retain another orientation')
    return tuple(sorted(paths))


def _fingerprints(paths):
    return tuple((str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths)


def cached_orientation_record(path, loader):
    path = Path(path).resolve()
    previous = _CACHE.get(path)
    if previous is not None:
        paths, fingerprints, value = previous
        if _fingerprints(paths) == fingerprints:
            _CACHE.move_to_end(path)
            return deepcopy(value)
        del _CACHE[path]
    paths = _dependencies(path)
    before = _fingerprints(paths)
    value = loader(path)
    if before != _fingerprints(paths):
        raise ValueError('retained orientation source changed during validation')
    _CACHE[path] = paths, before, deepcopy(value)
    while len(_CACHE) > 8:
        _CACHE.popitem(last=False)
    return value
