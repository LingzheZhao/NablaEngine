"""Compare saved engine substeps with MuJoCo on identical input packets.

Run in provisioned compute. This advances MuJoCo only, reusing saved engine tape.
Every substep retains same-packet contacts, accelerations and independent-chain errors.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from nablaengine.g1_engine import tape

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--controls', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(args.model))
    report = json.loads((args.run/'report.json').read_text())
    reading = next(row for row in report['executions'] if row['frame'] == args.frame
                   and row['requested_controls'] == args.controls)
    captured = np.load(args.run/f'frame_{args.frame}_{args.controls}_controls.npz', allow_pickle=False)
    records = tape.decode(captured['tapes'].reshape(-1, tape.rec_words(model)), model)
    chain = mujoco.MjData(model)
    chain.qpos[:] = captured['input_qpos']
    chain.qvel[:] = captured['input_qvel']
    pair_ids = {frozenset((int(a),int(b))): pair for pair,(a,b) in
                enumerate(zip(model.pair_geom1,model.pair_geom2))}
    rows = []
    count = len(records['qpos'])
    for substep in range(count):
        control = substep//4
        packet = mujoco.MjData(model)
        packet.qpos[:] = records['qpos'][substep]
        packet.qvel[:] = records['qvel'][substep]
        packet.qacc_warmstart[:] = records['qacc'][substep-1] if substep else reading['trace'][0]['inputs']['warm']
        packet.ctrl[:] = reading['trace'][control]['inputs']['ctrl']
        chain.ctrl[:] = packet.ctrl
        mujoco.mj_forward(model,packet)
        contacts = [dict(pair=pair_ids.get(frozenset((int(c.geom1),int(c.geom2)))),
                         geoms=[int(c.geom1),int(c.geom2)],distance=float(c.dist),
                         position=c.pos.tolist(),frame=c.frame.reshape(3,3).tolist()) for c in packet.contact]
        row = dict(substep=substep,input_qpos=packet.qpos.tolist(),input_qvel=packet.qvel.tolist(),
                   engine_pairs=records['pair'][substep].tolist(),engine_slots=records['slot'][substep].tolist(),
                   engine_qacc=records['qacc'][substep].tolist(),mujoco_same_packet_qacc=packet.qacc.tolist(),
                   qacc_max_error=float(np.max(np.abs(packet.qacc-records['qacc'][substep]))),
                   mujoco_same_packet_contacts=contacts,
                   independent_chain_pre_qpos_max_error=float(np.max(np.abs(chain.qpos-packet.qpos))),
                   independent_chain_pre_qvel_max_error=float(np.max(np.abs(chain.qvel-packet.qvel))))
        # Advance both MuJoCo packets only in the provisioned diagnostic run.
        mujoco.mj_step(model,packet)
        mujoco.mj_step(model,chain)
        next_q = records['qpos'][substep+1] if substep+1<count else captured['qpos'][-1]
        next_v = records['qvel'][substep+1] if substep+1<count else captured['qvel'][-1]
        row.update(same_packet_next_qpos_max_error=float(np.max(np.abs(packet.qpos-next_q))),
                   same_packet_next_qvel_max_error=float(np.max(np.abs(packet.qvel-next_v))),
                   independent_chain_next_qpos_max_error=float(np.max(np.abs(chain.qpos-next_q))),
                   independent_chain_next_qvel_max_error=float(np.max(np.abs(chain.qvel-next_v))))
        rows.append(row)
        # Reuse the existing static contact diagnostic on this exact engine PRE state.
        packet_root=args.out/f'packet_{substep}'
        packet_root.mkdir()
        np.savez_compressed(packet_root/f'frame_{args.frame}_1_controls.npz',
                            input_qpos=records['qpos'][substep],input_qvel=records['qvel'][substep])
    (args.out/'report.json').write_text(json.dumps(dict(frame=args.frame,source=str(args.run),substeps=rows),indent=2))


if __name__ == '__main__':
    main()
