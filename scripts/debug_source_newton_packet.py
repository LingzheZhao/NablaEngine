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
    parser.add_argument('--substeps',type=int,choices=[1,4],default=1)
    parser.add_argument('--qualification-report',type=Path)
    parser.add_argument('--direction-index',type=int,default=0)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.substeps==4 and args.substep!=0:parser.error('four-substep comparison starts from the original control input')
    args.out.mkdir(parents=True,exist_ok=False)
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
        return {name:'#include <stdio.h>\n#define DK_NEWTON_STOP_DEBUG 1\n#define DK_CONVEX_AABB 0\n'+source for name,source in units.items()},constants,defines
    jax_ffi.library_sources=debug_sources
    cc=torch.cuda.get_device_capability()
    artifact=engine.build(f'sm_{cc[0]}{cc[1]}',model=model,foot_pairs=contract['foot_pairs'],set_current=False)
    engine.init(artifact=artifact)
    def tensor(value):return torch.tensor(value[None],device='cuda',dtype=torch.float32)
    ctrl=saved['input_u']
    result=engine.forward(tensor(q),tensor(v),tensor(warm),tensor(ctrl),torch.ones(1,device='cuda'),
                          torch.zeros((1,3),device='cuda'),iterations=int(model.opt.iterations),substeps=args.substeps,tape=True)
    torch.cuda.synchronize()
    data=mujoco.MjData(model);data.qpos[:]=q;data.qvel[:]=v;data.ctrl[:]=ctrl;data.qacc_warmstart[:]=warm
    for _ in range(args.substeps):mujoco.mj_step(model,data)
    mass=np.empty((model.nv,model.nv));mujoco.mj_fullM(model,data,mass)
    decoded=tape.decode(result[4].cpu().numpy().reshape(args.substeps,-1),model)
    report=dict(execution=str(args.execution),substep=args.substep,substeps=args.substeps,iterations=int(model.opt.iterations),
                native_flags=decoded['flags'].tolist(),native_qacc=decoded['qacc'].tolist(),
                native_qacc_adv=decoded['qacc_adv'].tolist(),mujoco_qacc=data.qacc.tolist(),
                mujoco_force_rhs=(data.qfrc_smooth+data.qfrc_constraint).tolist(),
                mujoco_solver_residual=(mass@data.qacc-data.qfrc_smooth-data.qfrc_constraint).tolist(),
                native_next_qvel=result[1].cpu().numpy().tolist(),mujoco_next_qvel=data.qvel.tolist())
    if args.qualification_report:
        qualification=json.loads(args.qualification_report.read_text())
        selected=qualification['directions'][args.direction_index]
        direction=np.asarray(selected['direction'],dtype=np.float64)
        cq=np.asarray(qualification['terminal_cotangent']['qpos'])
        cv=np.asarray(qualification['terminal_cotangent']['qvel'])
        gradient=engine.backward(result[4],torch.ones(1,device='cuda'),torch.zeros((1,3),device='cuda'),
                                 tensor(cq.astype(np.float32)),tensor(cv.astype(np.float32)),substeps=args.substeps,ctrl=tensor(ctrl))
        torch.cuda.synchronize()
        if selected['block']=='qvel':
            report['analytic_directional_derivative']=float(gradient[1].cpu().numpy()[0]@direction)
        elif selected['block']=='ctrl':
            report['analytic_directional_derivative']=float(gradient[2].cpu().numpy()[0]@direction)
        report['analytic_flags']=gradient[3].cpu().numpy().tolist()
        report['finite_differences']=[]
        report['direction']=selected
        for epsilon in [1e-3,5e-4,1e-4,5e-5,1e-5,5e-6]:
            trials=[]
            for sign in [-1,1]:
                qi=q.astype(np.float64).copy();vi=v.astype(np.float64).copy();ui=ctrl.astype(np.float64).copy()
                if selected['block']=='qpos_tangent':mujoco.mj_integratePos(model,qi,direction,sign*epsilon)
                elif selected['block']=='qvel':vi+=sign*epsilon*direction
                elif selected['block']=='ctrl':ui+=sign*epsilon*direction
                else:raise ValueError('unsupported diagnostic direction block')
                # Both solvers consume exactly the same rounded physical packet.
                qi=qi.astype(np.float32);vi=vi.astype(np.float32);ui=ui.astype(np.float32)
                native=engine.forward(tensor(qi),tensor(vi),tensor(warm),tensor(ui),torch.ones(1,device='cuda'),
                                      torch.zeros((1,3),device='cuda'),iterations=int(model.opt.iterations),substeps=args.substeps,tape=True)
                torch.cuda.synchronize()
                md=mujoco.MjData(model);md.qpos[:]=qi;md.qvel[:]=vi;md.ctrl[:]=ui;md.qacc_warmstart[:]=warm
                for _ in range(args.substeps):mujoco.mj_step(model,md)
                nq=native[0].cpu().numpy()[0];nv=native[1].cpu().numpy()[0]
                flags=tape.decode(native[4].cpu().numpy().reshape(args.substeps,-1),model)['flags'].tolist()
                trials.append(dict(sign=sign,input_qpos=qi.tolist(),input_qvel=vi.tolist(),input_ctrl=ui.tolist(),
                                   native_qpos=nq.tolist(),native_qvel=nv.tolist(),native_flags=flags,
                                   mujoco_qpos=md.qpos.tolist(),mujoco_qvel=md.qvel.tolist(),
                                   native_objective=float(nq@cq+nv@cv),mujoco_objective=float(md.qpos@cq+md.qvel@cv)))
            report['finite_differences'].append(dict(epsilon=epsilon,trials=trials,
                native=(trials[1]['native_objective']-trials[0]['native_objective'])/(2*epsilon),
                mujoco=(trials[1]['mujoco_objective']-trials[0]['mujoco_objective'])/(2*epsilon)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
