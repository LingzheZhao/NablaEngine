"""Compare saved passive-force substeps with same-state MuJoCo solves.

Run in provisioned compute with the source revision that owns the saved tapes.
No engine compilation is required. Retains all row residuals, forces and zones.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    import mujoco
    import check_box_step_gradients as checker
    from nablaengine.g1_engine import tape
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--damping',type=float,required=True)
    parser.add_argument('--frictionloss',type=float,required=True)
    parser.add_argument('--hinges-only',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.run/'report.json').read_text())
    model,*_=checker.generated_model(report['condition'])
    start=6 if args.hinges_only else 0
    model.dof_damping[start:]=args.damping;model.dof_frictionloss[start:]=args.frictionloss
    rows=[]
    for condition in report['conditions']:
        name=condition['state'];controls=condition['controls']
        captured=np.load(args.run/f'{name}_{controls}_controls.npz',allow_pickle=False)
        decoded=tape.decode(captured['base_tape'].reshape(-1,tape.rec_words(model)),model)
        state=next(s for s in report['initial_states']if s['name']==name)
        chain=mujoco.MjData(model);chain.qpos[:]=state['q'];chain.qvel[:]=state['v'];chain.ctrl[:]=state['u']
        for i in range(len(decoded['qpos'])):
            data=mujoco.MjData(model);data.qpos[:]=decoded['qpos'][i];data.qvel[:]=decoded['qvel'][i]
            data.ctrl[:]=state['u'];data.qacc_warmstart[:]=decoded['qacc'][i-1] if i else 0
            mujoco.mj_forward(model,data)
            v_before=data.qvel.copy();q_before=data.qpos.copy()
            row=dict(state=name,controls=controls,substep=i,
                     qacc_max_error=float(np.max(np.abs(data.qacc-decoded['qacc'][i]))),
                     engine_qacc=decoded['qacc'][i].tolist(),mujoco_qacc=data.qacc.tolist(),
                     engine_zone=decoded['zone'][i].tolist(),mujoco_row_type=data.efc_type.tolist(),
                     mujoco_row_id=data.efc_id.tolist(),mujoco_row_force=data.efc_force.tolist(),
                     mujoco_row_D=data.efc_D.tolist(),mujoco_aref=data.efc_aref.tolist(),
                     mujoco_J=np.asarray(data.efc_J).tolist(),qvel=v_before.tolist(),qpos=q_before.tolist())
            mujoco.mj_step(model,data);mujoco.mj_step(model,chain)
            if i+1<len(decoded['qpos']):
                row.update(same_state_next_q_error=float(np.max(np.abs(data.qpos-decoded['qpos'][i+1]))),
                           same_state_next_v_error=float(np.max(np.abs(data.qvel-decoded['qvel'][i+1]))),
                           chain_next_v_error=float(np.max(np.abs(chain.qvel-decoded['qvel'][i+1]))))
            rows.append(row)
    (args.out/'report.json').write_text(json.dumps(dict(damping=args.damping,frictionloss=args.frictionloss,rows=rows),indent=2))


if __name__=='__main__':main()
