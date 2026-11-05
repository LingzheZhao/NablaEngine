"""Run upstream selected-terrain CoACD functions verbatim from a fixed inspected checkout.
Outputs are new directories only. Surface errors are sampled diagnostics, not equivalence proofs.
"""

import argparse, ast, json, pathlib, types, os, sys, platform, importlib.metadata, traceback
sys.dont_write_bytecode = True

import numpy as np
import trimesh, coacd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata", help="Released metadata YAML, required with --terrain-index")
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--terrain-index", type=int)
    selection.add_argument("--terrain-file", help="Terrain STL path relative to --dataset")
    p.add_argument("--artifact-id", required=True)
    p.add_argument("--upstream-revision", required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--wrapper-smoke", action="store_true")
    a = p.parse_args()
    a.upstream = str(pathlib.Path(a.upstream).resolve())
    a.dataset = str(pathlib.Path(a.dataset).resolve())
    if a.metadata:
        a.metadata = str(pathlib.Path(a.metadata).resolve())
    a.output = str(pathlib.Path(a.output).resolve())
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(
        state="started",
        upstream_declared_revision=a.upstream_revision,
        versions={
            "python": platform.python_version(),
            **{k: importlib.metadata.version(k) for k in ["numpy", "trimesh", "coacd"]},
        },
        arguments=vars(a),
        limitations="OBB vertex residual does not establish equivalence: face coverage and volume must also be checked.",
    )

    def write_status():
        (out / "execution.json").write_text(json.dumps(status, indent=2))

    write_status()
    # Keep native-library logs and any incidental relative files beside outputs.
    os.chdir(out)

    def record_failure(kind, value, tb):
        status.update(
            state="failed",
            failure_type=kind.__name__,
            failure=str(value),
            traceback="".join(traceback.format_exception(kind, value, tb)),
        )
        write_status()
        sys.__excepthook__(kind, value, tb)

    sys.excepthook = record_failure
    base = pathlib.Path(a.upstream) / "src/instinct_mj/terrains/trimesh"
    tree = ast.parse((base / "mesh_terrains.py").read_text())
    ns = dict(np=np, trimesh=trimesh, coacd=coacd, os=os)
    # Preserve every literal module assignment, including annotated globals.
    # Do not execute imports, cache registration, or hash/cache functions.
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    ns[target.id] = value
    funcs = [x for x in tree.body if isinstance(x, ast.FunctionDef)]
    utils_tree = ast.parse((base / "utils.py").read_text())
    funcs = [x for x in utils_tree.body if isinstance(x, ast.FunctionDef)] + funcs
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        )
                    ]
                    + funcs,
                    type_ignores=[],
                )
            ),
            "<upstream-functions>",
            "exec",
        ),
        ns,
    )
    cfg = {}
    for c in ast.parse((base / "mesh_terrains_cfg.py").read_text()).body:
        if isinstance(c, ast.ClassDef) and c.name == "MotionMatchedTerrainCfg":
            for x in c.body:
                if isinstance(x, ast.AnnAssign):
                    try:
                        cfg[x.target.id] = ast.literal_eval(x.value)
                    except (ValueError, TypeError):
                        pass
    cfg.update(
        collision_coacd_threshold=0.04,
        collision_coacd_resolution=3000,
        collision_coacd_decimate=False,
        collision_coacd_max_ch_vertex=256,
        collision_coacd_visualize_collision_hulls=True,
        use_input_origin_frame=True,
    )
    c = types.SimpleNamespace(**cfg)
    terrain_file = a.terrain_file
    if a.terrain_index is not None:
        if not a.metadata:
            p.error("--terrain-index requires --metadata")
        import yaml
        metadata = yaml.safe_load(pathlib.Path(a.metadata).read_text())
        selected = [t for t in metadata["terrains"] if t["terrain_id"] == a.terrain_index]
        if len(selected) != 1:
            raise ValueError("Terrain ID must select exactly one metadata entry")
        terrain_file = selected[0]["terrain_file"]
    source = pathlib.Path(a.dataset) / terrain_file
    status.update(artifact_id=a.artifact_id, terrain_file=terrain_file,
                  terrain_index=a.terrain_index, source=str(source),
                  frame={"input_to_world_translation_m": [15.0, 8.0, 0.0],
                         "coacd_z_alignment_applies_to": "collision hulls only"})
    write_status()
    mesh, border = ns["_load_motion_matched_terrain_mesh"](
        str(source), str(source), (30, 16), True, True
    )
    kw = ns["_coacd_run_kwargs"](c)
    if a.prepare_only:
        status.update(
            state="prepared",
            requested_kwargs=kw,
            effective_kwargs=ns["_adapt_coacd_kwargs_for_mesh"](kw, mesh),
            config=cfg,
        )
        write_status()
        return
    if a.wrapper_smoke:

        def fake_run_coacd(input_mesh, **kwargs):
            status["spy_kwargs"] = kwargs
            return [
                (
                    np.array(
                        [[15, 8, 0], [16, 8, 0], [15, 9, 0], [15, 8, 1]], dtype=float
                    ),
                    np.array(
                        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
                    ),
                )
            ]

        ns["coacd"] = types.SimpleNamespace(
            Mesh=coacd.Mesh, set_log_level=coacd.set_log_level, run_coacd=fake_run_coacd
        )
        status["synthetic_hulls_only"] = True
    parts = ns["_run_coacd_decomposition"](mesh, kw, f"terrain{a.terrain_index}", "off")
    z = ns["_compute_coacd_auto_align_z"](c, mesh, parts)
    report = dict(
        artifact_id=a.artifact_id, terrain_file=terrain_file, terrain_index=a.terrain_index,
        frame=status["frame"],
        config=cfg,
        requested_kwargs=kw,
        effective_kwargs=ns["_adapt_coacd_kwargs_for_mesh"](kw, mesh),
        border_height=border,
        auto_align_z=z,
        hulls=[],
    )
    for i, (v, f) in enumerate(parts):
        v = v.copy()
        v[:, 2] += z
        h = trimesh.Trimesh(v, f, process=False)
        h.export(out / f"hull_{i}.ply")
        transform, extents = trimesh.bounds.oriented_bounds(h)
        local = trimesh.transform_points(v, transform)
        residual = np.min(abs(abs(local) - extents / 2), axis=1)
        report["hulls"].append(
            dict(
                vertices=len(v),
                faces=len(f),
                obb_extents=extents.tolist(),
                obb_vertex_surface_max_m=float(residual.max()),
                volume=float(h.volume),
                obb_volume=float(np.prod(extents)),
            )
        )
    merged = trimesh.util.concatenate(
        [trimesh.Trimesh(v + np.array([0, 0, z]), f, process=False) for v, f in parts]
    )
    _, dist, _ = trimesh.proximity.closest_point_naive(
        merged, mesh.vertices[:: max(1, len(mesh.vertices) // 1000)]
    )
    report["raw_vertex_to_hulls_sample"] = dict(
        count=len(dist), mean_m=float(dist.mean()), max_m=float(dist.max())
    )
    (out / "result.json").write_text(json.dumps(report, indent=2))
    status.update(state="completed")
    write_status()


if __name__ == "__main__":
    main()
