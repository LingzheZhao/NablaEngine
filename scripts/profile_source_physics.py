"""Read existing DK_PROFILE phase counters on the source AABB check's fixed batch.

Provisioned CUDA execution only. Reuses check_convex_aabb.py's packets, cotangents,
128 worlds, four substeps, ten solver iterations, three warmups and ten repeats.
DK_PROFILE adds warp synchronization, clock reads and atomic counter accumulation.
Its timings describe the instrumented kernels. Phase values are lane-0 clock64
cycles summed across worlds and substeps, distinct from elapsed GPU milliseconds.
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np

import check_convex_aabb as checker


PROFILE_ACCESS = r'''
extern "C" int @UNIT@_profile_reset(cudaStream_t stream) {
  static const unsigned long long zero[NPROF > NAPROF ? NPROF : NAPROF] = {};
  cudaError_t error;
@RESET_FORWARD@
  error = cudaMemcpyToSymbolAsync(g_aprof, zero, NAPROF * sizeof(*zero),
                                  0, cudaMemcpyHostToDevice, stream);
  return (int)error;
}
extern "C" int @UNIT@_profile_read(unsigned long long* forward, unsigned long long* adjoint) {
  cudaError_t error;
@READ_FORWARD@
  if (adjoint) {
    error = cudaMemcpyFromSymbol(adjoint, g_aprof, NAPROF * sizeof(*adjoint));
    if (error != cudaSuccess) return (int)error;
  }
  return 0;
}
'''


def profile_access(unit, forward):
    reset = read = ''
    if forward:
        reset = '''  error = cudaMemcpyToSymbolAsync(g_prof, zero, NPROF * sizeof(*zero),
                                  0, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess) return (int)error;'''
        read = '''  if (forward) {
    error = cudaMemcpyFromSymbol(forward, g_prof, NPROF * sizeof(*forward));
    if (error != cudaSuccess) return (int)error;
  }'''
    return (PROFILE_ACCESS.replace('@UNIT@', unit).replace('@RESET_FORWARD@', reset)
            .replace('@READ_FORWARD@', read))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'reference', 'screen', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    args.aabb = 1

    import torch
    from nablaengine import jax_ffi, torch_ffi as engine
    from nablaengine.g1_engine import adjoint, step

    generate, forward, backward = jax_ffi.library_sources, engine.forward, engine.backward
    calls = dict(forward=0, backward=0)
    handles = {}

    def sources(*values, **options):
        units, constants, defines = generate(*values, **options)
        for filename, code in units.items():
            if filename == 'dk_ffi.cu':
                continue
            unit = Path(filename).stem
            has_forward = f'int {unit}_forward(' in code
            units[filename] = '#define DK_PROFILE 1\n' + code + profile_access(unit, has_forward)
        return units, constants, defines

    def counted_forward(*values, **options):
        calls['forward'] += 1
        return forward(*values, **options)

    def counted_backward(*values, **options):
        result = backward(*values, **options)
        calls['backward'] += 1
        if calls['backward'] == checker.PROTOCOL['warmup']:
            info = engine.init()
            handles['library'] = ctypes.CDLL(info['path'])
            handles['artifact_version'] = info['artifact_version']
            for kind, unit in (('forward', info['forward_unit']),
                               ('adjoint', f"dk_kv{checker.PROTOCOL['clamp_kv']}_n{checker.PROTOCOL['normal']}")):
                reset = getattr(handles['library'], unit + '_profile_reset')
                reset.argtypes = [ctypes.c_void_p]
                code = reset(torch.cuda.current_stream().cuda_stream)
                if code:
                    raise RuntimeError(f'{kind} profile reset failed with CUDA error {code}')
                read = getattr(handles['library'], unit + '_profile_read')
                read.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                handles[kind] = read
        return result

    jax_ffi.library_sources = sources
    engine.forward, engine.backward = counted_forward, counted_backward
    try:
        checker.run(args)
    finally:
        jax_ffi.library_sources = generate
        engine.forward, engine.backward = forward, backward
    expected = checker.PROTOCOL['warmup'] + checker.PROTOCOL['repeats']
    if calls != dict(forward=expected, backward=expected):
        raise RuntimeError(f'Unexpected profiled launch counts {calls}')

    forward_cycles = np.zeros(len(step.PROF_PHASES), np.uint64)
    adjoint_cycles = np.zeros(len(adjoint.PROF_PHASES), np.uint64)
    for kind, values in (('forward', forward_cycles), ('adjoint', adjoint_cycles)):
        pointers = (values.ctypes.data, None) if kind == 'forward' else (None, values.ctypes.data)
        code = handles[kind](*pointers)
        if code:
            raise RuntimeError(f'{kind} profile read failed with CUDA error {code}')

    launches = checker.PROTOCOL['repeats']
    divisor = launches * checker.PROTOCOL['environments'] * checker.PROTOCOL['substeps']
    def reading(module, values):
        total = int(values[:module.PROF_PARTITION].sum())
        return dict(
            raw_counter_sums=dict(zip(module.PROF_PHASES, map(int, values))),
            counters_per_world_substep=dict(zip(module.PROF_PHASES, (values / divisor).tolist())),
            partition_percent={name: 100 * int(value) / total for name, value in
                               zip(module.PROF_PHASES[:module.PROF_PARTITION], values[:module.PROF_PARTITION])},
        )
    report = dict(
        code_revision=args.code_revision, artifact_version=handles['artifact_version'],
        protocol=checker.PROTOCOL, aabb=1, recorded_launches=dict(forward=launches, adjoint=launches),
        counter_reset='On the current CUDA stream after warmup backward 3, before the existing warmup synchronization',
        scope='DK_PROFILE instrumented source physics phase diagnosis',
        measurement_effect='Warp synchronization, clock64 reads and atomic accumulation are enabled in production kernels',
        units='Partition and subphase counters use lane-0 clock64 cycles. count_* counters count their named events.',
        forward=reading(step, forward_cycles), adjoint=reading(adjoint, adjoint_cycles),
    )
    (args.out / 'phase-profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
