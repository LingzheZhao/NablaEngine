"""Run upstream terrain0 CoACD functions verbatim from a fixed inspected checkout.
Outputs are new directories only. Surface errors are sampled diagnostics, not equivalence proofs.
"""

import argparse, ast, json, pathlib, types, os, sys, platform, importlib.metadata, traceback
import numpy as np
import trimesh, coacd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--upstream-revision", required=True)
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
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
    funcs = [x for x in tree.body if isinstance(x, ast.FunctionDef)]
    utils_tree = ast.parse((base / 'utils.py').read_text())
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
    source = (
        pathlib.Path(a.dataset)
        / "20251106_diveroll4_simpleLab_noWall/simplelab_enlarged_noWall.stl"
    )
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
    parts = ns["_run_coacd_decomposition"](mesh, kw, "terrain0", "off")
    z = ns["_compute_coacd_auto_align_z"](c, mesh, parts)
    report = dict(
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
