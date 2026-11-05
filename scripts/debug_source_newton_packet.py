"""Trace Newton iterations for one unchanged recorded source-state packet on GPU."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main():
    import mujoco
    import torch
    from nablaengine import jax_ffi, torch_ffi as engine
    from nablaengine.g1_engine import tape
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--substep',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    os.environ['DK_CACHE_DIR']=str(args.out/'build')
    model=mujoco.MjModel.from_binary_path(str(args.scene/'model.mjb'))
    contract=json.loads((args.scene/'contract.json').read_text())
    saved=np.load(args.execution,allow_pickle=False)
    control,index=divmod(args.substep,4)
    record=tape.decode(saved[f'control_{control}_tape'].reshape(4,-1),model)
    q,v=record['qpos'][index],record['qvel'][index]
    if args.substep:
        previous,previous_index=divmod(args.substep-1,4)
        warm=tape.decode(saved[f'control_{previous}_tape'].reshape(4,-1),model)['qacc'][previous_index]
    else:warm=saved['input_warm']
    generate=jax_ffi.library_sources
    def debug_sources(*values,**options):
        units,constants,defines=generate(*values,**options)
        return {name:'#define DK_NEWTON_STOP_DEBUG 1\n#define DK_CONVEX_AABB 0\n'+source for name,source in units.items()},constants,defines
    jax_ffi.library_sources=debug_sources
    cc=torch.cuda.get_device_capability()
    artifact=engine.build(f'sm_{cc[0]}{cc[1]}',model=model,foot_pairs=contract['foot_pairs'],set_current=False)
    engine.init(artifact=artifact)
    def tensor(value):return torch.tensor(value[None],device='cuda',dtype=torch.float32)
    ctrl=saved['input_u']
    result=engine.forward(tensor(q),tensor(v),tensor(warm),tensor(ctrl),torch.ones(1,device='cuda'),
                          torch.zeros((1,3),device='cuda'),iterations=int(model.opt.iterations),substeps=1,tape=True)
    torch.cuda.synchronize()
    data=mujoco.MjData(model);data.qpos[:]=q;data.qvel[:]=v;data.ctrl[:]=ctrl;data.qacc_warmstart[:]=warm
    mujoco.mj_step(model,data)
    mass=np.empty((model.nv,model.nv));mujoco.mj_fullM(model,data,mass)
    decoded=tape.decode(result[4].cpu().numpy().reshape(1,-1),model)
    report=dict(execution=str(args.execution),substep=args.substep,iterations=int(model.opt.iterations),
                native_flags=decoded['flags'].tolist(),native_qacc=decoded['qacc'].tolist(),
                native_qacc_adv=decoded['qacc_adv'].tolist(),mujoco_qacc=data.qacc.tolist(),
                mujoco_force_rhs=(data.qfrc_smooth+data.qfrc_constraint).tolist(),
                mujoco_solver_residual=(mass@data.qacc-data.qfrc_smooth-data.qfrc_constraint).tolist(),
                native_next_qvel=result[1].cpu().numpy().tolist(),mujoco_next_qvel=data.qvel.tolist())
    (args.out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
