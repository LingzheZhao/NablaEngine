"""Prepare selected metadata terrains with the existing decomposition and scene producers.

Execute on explicitly provisioned compute. Each terrain has its own model and
fresh output directory. --plan-only writes commands and known budgets without
running decomposition or model construction. No scheduler or storage adapter.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'dataset', 'metadata', 'fixture', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('artifact-id', 'code-revision', 'upstream-revision'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--terrain-index', type=int, action='append')
    parser.add_argument('--coacd', nargs=2, action='append', default=[], metavar=('TERRAIN_ID', 'DIRECTORY'),
                        help='Reuse a completed current-format decomposition, without modifying it')
    parser.add_argument('--operation', choices=('coacd', 'scenes', 'all'), default='all')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--visual-model', type=Path)
    parser.add_argument('--visual-artifact-id')
    args = parser.parse_args()
    for name in ('upstream', 'dataset', 'metadata', 'fixture', 'output', 'visual_model'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if bool(args.visual_model) != bool(args.visual_artifact_id):
        parser.error('--visual-model and --visual-artifact-id must be supplied together')
    reuse = {}
    for terrain_id, directory in args.coacd:
        terrain_id = int(terrain_id)
        if terrain_id in reuse:
            parser.error('Each terrain may have one selected decomposition')
        reuse[terrain_id] = Path(directory).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    status = dict(state='started', code_revision=args.code_revision, upstream_revision=args.upstream_revision,
                  artifact_id=args.artifact_id, operation=args.operation, conditions=[])

    def write_status():
        (args.output / 'execution.json').write_text(json.dumps(status, indent=2))

    def fail(kind, value, tb):
        status.update(state='failed', failure_type=kind.__name__, failure=str(value),
                      traceback=''.join(traceback.format_exception(kind, value, tb)))
        write_status()
        sys.__excepthook__(kind, value, tb)

    write_status()
    sys.excepthook = fail
    metadata = yaml.safe_load(args.metadata.read_text())
    terrains = {row['terrain_id']: row for row in metadata['terrains']}
    if len(terrains) != len(metadata['terrains']):
        raise ValueError('Metadata terrain IDs must be unique')
    selected = list(terrains) if args.terrain_index is None else args.terrain_index
    if len(set(selected)) != len(selected) or not set(selected) <= terrains.keys():
        raise ValueError('Selected terrain IDs must be unique metadata entries')
    if not reuse.keys() <= set(selected):
        raise ValueError('Selected decomposition must belong to a selected terrain')
    for motion in metadata['motion_files']:
        if motion['terrain_id'] not in terrains:
            raise ValueError('Every metadata motion must bind an existing terrain')
    scripts = Path(__file__).resolve().parent
    index = dict(artifact_id=args.artifact_id, metadata=str(args.metadata), code_revision=args.code_revision,
                 upstream_revision=args.upstream_revision, fixture=str(args.fixture), terrains=[],
                 motions=[dict(motion_index=i, **row) for i, row in enumerate(metadata['motion_files'])],
                 layout=dict(mode='separate_models', input_to_single_scene_translation_m=[15., 8., 0.],
                             proposed_combined_tile_size_m=[30., 16.], proposed_combined_columns=3,
                             combined_model_constructed=False))

    def write_index():
        completed = [row for row in index['terrains'] if row.get('hulls') is not None]
        hulls = sum(row['hulls'] for row in completed)
        index['budget'] = dict(known_terrain_ids=[row['terrain_id'] for row in completed],
                               unknown_terrain_ids=[i for i in selected if i not in [row['terrain_id'] for row in completed]],
                               known_hulls=hulls, combined_pairs_for_known_hulls=469 + 33*hulls,
                               combined_collision_geoms_for_known_hulls=33+hulls,
                               combined_visual_geoms_if_source_imported=35 if args.visual_model else 0,
                               formulas=dict(pairs='fixture_self_pairs + robot_collision_geoms * sum(hulls)',
                                             collision_geoms='robot_collision_geoms + sum(hulls)'),
                               assumed_fixture_self_pairs=469, assumed_robot_collision_geoms=33)
        (args.output / 'index.json').write_text(json.dumps(index, indent=2))

    for terrain_id in selected:
        entry = terrains[terrain_id]
        position = list(terrains).index(terrain_id)
        tile = [30.*(position % 3), 16.*(position // 3), 0.]
        root = args.output / f'terrain_{terrain_id}'
        coacd = reuse.get(terrain_id, root / 'coacd')
        decomposition_id = f'{args.artifact_id}/terrain/{terrain_id}/coacd'
        row = dict(terrain_id=terrain_id, terrain_file=entry['terrain_file'], coacd=str(coacd),
                   scene=str(root / 'scene'), proposed_combined_translation_m=tile,
                   proposed_input_to_combined_translation_m=[15.+tile[0], 8.+tile[1], 0.],
                   motion_indices=[i for i, motion in enumerate(metadata['motion_files']) if motion['terrain_id'] == terrain_id],
                   state='planned', commands=[], hulls=None)
        index['terrains'].append(row)
        if terrain_id in reuse:
            execution = json.loads((coacd / 'execution.json').read_text())
            result = json.loads((coacd / 'result.json').read_text())
            if execution['state'] != 'completed' or execution.get('synthetic_hulls_only'):
                raise ValueError(f'Terrain {terrain_id} selected decomposition did not complete with real hulls')
            if result['terrain_index'] != terrain_id or result['terrain_file'] != entry['terrain_file']:
                raise ValueError(f'Terrain {terrain_id} selected decomposition disagrees with metadata')
            if execution['artifact_id'] != result['artifact_id']:
                raise ValueError(f'Terrain {terrain_id} decomposition identities disagree')
            decomposition_id = result['artifact_id']
            row['hulls'] = len(result['hulls'])
        elif args.operation == 'scenes':
            raise ValueError(f'Terrain {terrain_id} scene construction requires its selected completed decomposition')
        else:
            row['commands'].append([sys.executable, str(scripts / 'coacd_audit.py'),
                '--upstream', str(args.upstream), '--dataset', str(args.dataset), '--metadata', str(args.metadata),
                '--upstream-revision', args.upstream_revision, '--terrain-index', str(terrain_id),
                '--artifact-id', decomposition_id, '--output', str(coacd)])
        if args.operation in ('scenes', 'all'):
            command = [sys.executable, str(scripts / 'prepare_terrain_scene.py'), '--fixture', str(args.fixture),
                       '--coacd', str(coacd), '--artifact-id', f'{args.artifact_id}/terrain/{terrain_id}/scene',
                       '--terrain-artifact-id', decomposition_id, '--code-revision', args.code_revision,
                       '--output', str(root / 'scene')]
            if args.visual_model:
                command += ['--visual-model', str(args.visual_model), '--visual-artifact-id', args.visual_artifact_id]
            row['commands'].append(command)
        row['coacd_artifact_id'] = decomposition_id
        write_index()
        if args.plan_only:
            continue
        root.mkdir()
        try:
            for command in row['commands']:
                operation = 'coacd' if Path(command[1]).name == 'coacd_audit.py' else 'scene'
                row['state'] = f'running_{operation}'
                write_index()
                with (root / f'{operation}.log').open('wb') as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
                if operation == 'coacd':
                    result = json.loads((coacd / 'result.json').read_text())
                    row['hulls'] = len(result['hulls'])
                    write_index()
            result = json.loads((coacd / 'result.json').read_text())
            row.update(state='completed', hulls=len(result['hulls']))
            if args.operation in ('scenes', 'all'):
                contract = json.loads((root / 'scene/contract.json').read_text())
                row['counts'] = contract['counts']
        except Exception as error:
            row.update(state='failed', failure_type=type(error).__name__, failure=str(error))
        status['conditions'].append(dict(terrain_id=terrain_id, state=row['state']))
        write_index()
        write_status()
    failed = [row['terrain_id'] for row in index['terrains'] if row['state'] == 'failed']
    status.update(state='planned' if args.plan_only else ('failed' if failed else 'completed'), failed_terrain_ids=failed)
    write_status()
    write_index()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
