"""Compare static mesh AABB dispatch using identical recorded physical packets.

Run each mode in a separate provisioned GPU process, then compare the saved arrays.
Warm-up and timed calls reuse the same packets and include forward plus backward.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    import mujoco
    import torch
    from nablaengine import jax_ffi, torch_ffi as engine
    from nablaengine.g1_engine import tape

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[0,190,235,320])
    parser.add_argument('--aabb',type=int,choices=[0,1],required=True)
    parser.add_argument('--environments',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['DK_CACHE_DIR']=str(args.out/'build')
    model=mujoco.MjModel.from_binary_path(str(args.scene/'model.mjb'))
    contract=json.loads((args.scene/'contract.json').read_text())
    report=json.loads((args.run/'report.json').read_text())
    generate=jax_ffi.library_sources
    def selected_sources(*values,**options):
        units,constants,defines=generate(*values,**options)
        units={name:f'#define DK_CONVEX_AABB {args.aabb}\n'+source for name,source in units.items()}
        return units,constants,defines
    jax_ffi.library_sources=selected_sources
    device=torch.cuda.get_device_capability()
    artifact=engine.build(f'sm_{device[0]}{device[1]}',model=model,foot_pairs=contract['foot_pairs'],set_current=False)
    engine.init(artifact=artifact)
    packets=[next(row['trace'][0]['inputs'] for row in report['executions']
                  if row['frame']==frame and row['requested_controls']==1) for frame in args.frames]
    selected=[packets[i%len(packets)] for i in range(args.environments)]
    def tensor(key):
        return torch.tensor(np.asarray([p[key] for p in selected]),device='cuda',dtype=torch.float32)
    q,v,warm,ctrl=tensor('qpos'),tensor('qvel'),tensor('warm'),tensor('ctrl')
    friction=torch.ones(args.environments,device='cuda')
    com=torch.zeros((args.environments,3),device='cuda')
    cq=torch.ones_like(q)/q.shape[1]
    cv=torch.ones_like(v)/v.shape[1]
    def read():
        result=engine.forward(q,v,warm,ctrl,friction,com,iterations=50,substeps=4,tape=True)
        gradients=engine.backward(result[4],friction,com,cq,cv,substeps=4,ctrl=ctrl)
        return result,gradients
    for _ in range(3):read()
    torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(args.repeats):result,gradients=read()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    arrays={name:value.detach().cpu().numpy() for name,value in
            zip(['qpos','qvel','warm','forward_info','tape','gq','gv','gu','backward_info'],[*result,*gradients])}
    decoded=tape.decode(arrays['tape'].reshape(args.environments,4,-1),model)
    np.savez_compressed(args.out/'arrays.npz',**arrays)
    summary=dict(aabb=args.aabb,environments=args.environments,repeats=args.repeats,
                 milliseconds_forward_backward=elapsed*1000/args.repeats,
                 frames=args.frames,artifact=str(artifact),max_contacts=int(decoded['ncon'].max()),
                 forward_flags=np.unique(decoded['flags']).tolist(),
                 backward_flags=np.unique(arrays['backward_info']).tolist(),
                 finite=all(np.isfinite(arrays[k]).all() for k in ['qpos','qvel','gq','gv','gu']))
    (args.out/'report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__=='__main__':
    main()
