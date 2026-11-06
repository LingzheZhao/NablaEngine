"""Compare static-convex AABB dispatch on the current source reference packets.

Run --aabb 0 and --aabb 1 in separate provisioned GPU processes, then use the
compare subcommand. Both conditions load the selected_states schema written by
screen_terrain_references.py. All outputs require fresh external directories.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

PROTOCOL = dict(
    format='nablaengine-source-convex-aabb-check-v1', environments=128,
    iterations=10, substeps=4, warmup=3, repeats=10, cotangent_seed=20261106,
    packets='All selected states, sorted by motion then selected frame, cycled to the batch size',
    comparison='Bitwise state, warm start, infos, gradients and valid tape fields; integer tape fields use uint32 bits',
    tape_support='Declared fields only; contacts and zones restricted to ncon and nefc; padding excluded',
    normal=0, clamp_kv=1,
)


def load_packets(scene, reference, screen):
    """Reuse the current source model/reference boundary and check selected rows."""
    from screen_terrain_references import load_inputs

    model, contract, _, _, sources, _ = load_inputs(scene, reference)
    report = json.loads((screen/'report.json').read_text())
    if report['protocol']['format'] != 'nablaengine-terrain-reference-screen-v1':
        raise ValueError('Current reference-state screen schema required')
    if report['state'] != 'completed' or report['failed_frames'] or report['nonfinite_frames']:
        raise ValueError('Selected reference-state screen must be complete with its failures visible')
    if report['scene_artifact_id'] != contract['artifact_id']:
        raise ValueError('Selected screen and constructed scene artifact identities differ')
    screening = json.loads((reference/'screening.json').read_text())
    if report['reference_artifact_id'] != screening['artifact_id']:
        raise ValueError('Selected screen and reference artifact identities differ')
    if set(report['motions']) != set(sources):
        raise ValueError('Selected screen must cover the declared reference motions')
    parts = {key: [] for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'friction', 'com')}
    selection = []
    columns = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    for motion in sorted(report['motions']):
        declared = report['motions'][motion]['selected']
        with np.load(screen/motion/'selected_states.npz', allow_pickle=False) as saved:
            frames = saved['frames']
            if frames.tolist() != [row['frame'] for row in declared]:
                raise ValueError(f'{motion} saved frames differ from the screen selection')
            source = sources[motion]
            if not np.array_equal(saved['qpos'], source['qpos'][frames]):
                raise ValueError(f'{motion} selected qpos differs from current reference')
            if not np.array_equal(saved['qvel'], source['qvel'][frames]):
                raise ValueError(f'{motion} selected qvel differs from current reference')
            if not np.array_equal(saved['ctrl'], source['qpos'][frames][:, columns]):
                raise ValueError(f'{motion} selected controls differ from current-frame joint targets')
            count = len(frames)
            shapes = dict(qpos=(count, model.nq), qvel=(count, model.nv), ctrl=(count, model.nu),
                          qacc_warmstart=(count, model.nv), friction=(count,), com=(count, 3))
            for key, shape in shapes.items():
                values = saved[key]
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f'{motion} {key} must have finite values and shape {shape}')
                parts[key].append(values.astype(np.float32))
            if np.any(saved['qacc_warmstart']) or np.any(saved['com']) or not np.all(saved['friction'] == 1):
                raise ValueError('Selected reference packets require zero warm/COM and unit friction multiplier')
        selection.extend(dict(motion=motion, frame=row['frame'], reasons=row['reasons']) for row in declared)
    return model, contract, report, {key: np.concatenate(value) for key, value in parts.items()}, selection


def valid_tape_arrays(raw, model):
    """Retain raw field words while excluding unwritten record padding."""
    from nablaengine.g1_engine import tape

    records = np.ascontiguousarray(raw, np.float32).reshape(raw.shape[0], PROTOCOL['substeps'], tape.rec_words(model))
    decoded = tape.decode(records, model)
    words = records.view(np.uint32)
    result = {}
    for name, (offset, count) in tape.layout(model).items():
        field = words[..., offset:offset+count].copy()
        if name == 'con':
            active = np.arange(count)[None, None, :] < decoded['ncon'][..., None]
            field[~active] = 0
        elif name == 'zone':
            for word in range(count):
                bits = np.clip(decoded['nefc']-32*word, 0, 32).astype(np.uint64)
                mask = ((np.uint64(1) << bits)-np.uint64(1)).astype(np.uint32)
                field[..., word] &= mask
        elif name == 'clamp':
            for word in range(count):
                bits = min(32, max(0, model.nv-32*word))
                field[..., word] &= np.uint32((1 << bits)-1)
        result[f'tape_{name}_bits'] = field
    return result, decoded


def run(args):
    import torch
    from nablaengine import jax_ffi, torch_ffi as engine
    from nablaengine.g1_engine import tape

    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(PROTOCOL, code_revision=args.code_revision, aabb=args.aabb,
                    scene=str(args.scene), reference=str(args.reference), screen=str(args.screen),
                    resolved_macro=dict(name='DK_CONVEX_AABB', value=args.aabb,
                                        selection='Explicit source prefix in every generated translation unit'))
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    os.environ['DK_CACHE_DIR'] = str(args.out/'build')
    os.environ.pop('DK_TORCH_ARTIFACT', None)
    model, contract, screen, packets, selection = load_packets(args.scene, args.reference, args.screen)
    batch_indices = np.arange(PROTOCOL['environments']) % len(selection)
    inputs = {key: value[batch_indices] for key, value in packets.items()}
    rng = np.random.default_rng(PROTOCOL['cotangent_seed'])
    inputs['qpos_cotangent'] = rng.normal(size=(PROTOCOL['environments'], model.nq)).astype(np.float32)
    inputs['qvel_cotangent'] = rng.normal(size=(PROTOCOL['environments'], model.nv)).astype(np.float32)
    inputs['qpos_cotangent'] /= np.linalg.norm(inputs['qpos_cotangent'], axis=1, keepdims=True)
    inputs['qvel_cotangent'] /= np.linalg.norm(inputs['qvel_cotangent'], axis=1, keepdims=True)
    np.savez_compressed(args.out/'inputs.npz', **inputs)
    (args.out/'selection.json').write_text(json.dumps(dict(selected=selection, batch_indices=batch_indices.tolist(),
        scene_artifact_id=contract['artifact_id'], reference_artifact_id=screen['reference_artifact_id']), indent=2)+'\n')
    generate = jax_ffi.library_sources
    resolved_units = []
    def selected_sources(*values, **options):
        units, constants, defines = generate(*values, **options)
        units = {name: f'#define DK_CONVEX_AABB {args.aabb}\n'+source for name, source in units.items()}
        resolved_units.extend(units)
        return units, constants, defines
    jax_ffi.library_sources = selected_sources
    capability = torch.cuda.get_device_capability()
    artifact = engine.build(f'sm_{capability[0]}{capability[1]}', model=model,
                            foot_pairs=contract['foot_pairs'], set_current=False)
    engine.init(artifact=artifact)
    gpu = {key: torch.tensor(value, device='cuda').contiguous() for key, value in inputs.items()}
    def forward():
        return engine.forward(gpu['qpos'], gpu['qvel'], gpu['qacc_warmstart'], gpu['ctrl'],
                              gpu['friction'], gpu['com'], iterations=PROTOCOL['iterations'],
                              substeps=PROTOCOL['substeps'], tape=True)
    def backward(result):
        return engine.backward(result[4], gpu['friction'], gpu['com'], gpu['qpos_cotangent'],
                               gpu['qvel_cotangent'], substeps=PROTOCOL['substeps'], ctrl=gpu['ctrl'],
                               normal=PROTOCOL['normal'], clamp_kv=PROTOCOL['clamp_kv'])
    for _ in range(PROTOCOL['warmup']):
        backward(forward())
    torch.cuda.synchronize()
    forward_ms, backward_ms, combined_ms = [], [], []
    engine.read_diagnostics(reset=True)
    for _ in range(PROTOCOL['repeats']):
        start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        begin = time.perf_counter()
        start.record()
        result = forward()
        middle.record()
        gradients = backward(result)
        end.record()
        end.synchronize()
        combined_ms.append((time.perf_counter()-begin)*1000)
        forward_ms.append(start.elapsed_time(middle))
        backward_ms.append(middle.elapsed_time(end))
    diagnostics = engine.read_diagnostics(reset=True)
    names = ['qpos', 'qvel', 'warm', 'forward_info', 'tape', 'gq', 'gv', 'gu', 'backward_info']
    arrays = {name: value.detach().cpu().numpy() for name, value in zip(names, [*result, *gradients])}
    effective, decoded = valid_tape_arrays(arrays['tape'], model)
    np.savez_compressed(args.out/'arrays.npz', **arrays)
    np.savez_compressed(args.out/'valid_tape.npz', **effective)
    finite = all(np.isfinite(arrays[key]).all() for key in ('qpos', 'qvel', 'warm', 'gq', 'gv', 'gu'))
    strict_mask = tape.FLAG_UNSUPPORTED | tape.FLAG_NONCONV | tape.FLAG_SINGULAR | tape.FLAG_VERIFY_DIFF
    strict_valid = bool(finite and not np.any(decoded['flags'] & strict_mask)
                        and not np.any(arrays['forward_info'][:, 3])
                        and not np.any(arrays['backward_info']))
    report = dict(aabb=args.aabb, code_revision=args.code_revision, artifact=str(artifact),
        selected_states=len(selection), environments=PROTOCOL['environments'], finite=finite, strict_valid=strict_valid,
        resolved_macro=protocol['resolved_macro'], source_prefixed_units=resolved_units,
        environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
        timings=dict(forward_gpu_ms=forward_ms, backward_gpu_ms=backward_ms, forward_backward_wall_ms=combined_ms,
            median_forward_gpu_ms=float(np.median(forward_ms)), median_backward_gpu_ms=float(np.median(backward_ms)),
            median_forward_backward_wall_ms=float(np.median(combined_ms))),
        maximum_contacts=int(decoded['ncon'].max()), maximum_rows=int(decoded['nefc'].max()),
        forward_flags=np.unique(decoded['flags']).tolist(), backward_flags=np.unique(arrays['backward_info']).tolist(),
        diagnostics=diagnostics, candidate_cull_statistics=None,
        candidate_cull_statistics_note='The engine API exposes contact counts and status, without runtime broad-phase counters')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def bit_comparison(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(equal=False, left_shape=list(left.shape), right_shape=list(right.shape),
                    left_dtype=str(left.dtype), right_dtype=str(right.dtype))
    a, b = np.ascontiguousarray(left), np.ascontiguousarray(right)
    bits_equal = a.view(np.uint8) == b.view(np.uint8)
    result = dict(equal=bool(bits_equal.all()), differing_bytes=int(np.sum(~bits_equal)))
    if a.dtype.kind == 'f':
        result['max_absolute_error'] = float(np.max(np.abs(a.astype(float)-b.astype(float)), initial=0))
    return result


def compare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    protocols = [json.loads((path/'protocol.json').read_text()) for path in (args.disabled, args.enabled)]
    left_protocol, right_protocol = [{key: value for key, value in protocol.items() if key not in ('aabb', 'resolved_macro')}
                                     for protocol in protocols]
    if (protocols[0]['aabb'] != 0 or protocols[1]['aabb'] != 1
            or protocols[0]['resolved_macro']['value'] != 0 or protocols[1]['resolved_macro']['value'] != 1
            or left_protocol != right_protocol):
        raise ValueError('Conditions must use the same declared protocol and differ only in AABB 0 versus 1')
    comparisons = {}
    for filename in ('inputs.npz', 'arrays.npz', 'valid_tape.npz'):
        with np.load(args.disabled/filename, allow_pickle=False) as left, np.load(args.enabled/filename, allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise ValueError(f'{filename} array names differ')
            for key in left.files:
                if filename == 'arrays.npz' and key == 'tape':
                    continue  # The valid field words are compared from valid_tape.npz.
                comparisons[f'{filename}/{key}'] = bit_comparison(left[key], right[key])
    reports = [json.loads((path/'report.json').read_text()) for path in (args.disabled, args.enabled)]
    same_selection = json.loads((args.disabled/'selection.json').read_text()) == json.loads((args.enabled/'selection.json').read_text())
    before, after = [row['timings']['median_forward_backward_wall_ms'] for row in reports]
    result = dict(bitwise_equal=same_selection and all(row['equal'] for row in comparisons.values()),
                  strict_valid=all(row['strict_valid'] for row in reports), same_selection=same_selection,
                  comparisons=comparisons, conditions=reports,
                  forward_backward_wall_speed_ratio=before/after)
    result['passed'] = result['bitwise_equal'] and result['strict_valid']
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='operation', required=True)
    execute = subparsers.add_parser('run')
    for name in ('scene', 'reference', 'screen', 'out'):
        execute.add_argument(f'--{name}', type=Path, required=True)
    execute.add_argument('--aabb', type=int, choices=[0, 1], required=True)
    execute.add_argument('--code-revision', required=True)
    comparison = subparsers.add_parser('compare')
    comparison.add_argument('--disabled', type=Path, required=True)
    comparison.add_argument('--enabled', type=Path, required=True)
    comparison.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args) if args.operation == 'run' else compare(args)


if __name__ == '__main__':
    main()
