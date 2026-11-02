"""MuJoCo Warp with the unmerged adjoint pull request: 'mjw_pr1535' (etaoxing/mujoco_warp @ 357a75d, the head of the
open google-deepmind/mujoco_warp#1535 "hybrid analytic differentiability", unmodified) and 'mjw_pr1535_fix' (357a75d +
etaoxing/mujoco_warp#1 = 4af093f, the two public fixes of that pull request: the pyramidal-contact skip and the
actuator-clamp multiplier). Environment 'mjw-pr': mujoco 3.17.0 + warp-lang 1.18.0 with the tree on PYTHONPATH
(benchmarks/envs/envs.toml, made by benchmarks/envs/setup.sh); setup refuses any other tree (package source hash).

Forward (op fwd; fwd_train is the same callable, recorded as alias_of: the PR's backward recomputes the forward from
the stored inputs, so the forward a backward needs is the plain one): the tree's own 3.12 mjw.step, CUDA graph of
[copy the inputs into Data; substeps x mjw.step] as in mjw.py (mjw_common.MjwBase), after mjw.enable_grad() (which the
PR's demos call at import; it installs the backward hooks and a safe sqrt, so it changes the forward kernels too).

Backward (op bwd): the adjoint-graph pattern of the PR's contrib/diffsim/_rollout.py (Rollout._backward_step and
_capture; Apache-2.0, The Newton Developers), applied to one control step of `substeps` steps: a chain of Data
D[0..substeps] with D[0].qpos / qvel / ctrl requiring gradients and mjw.create_backward_context(m, D[0]); then, captured
as one CUDA graph, [copy the inputs into D[0]; under mjw.backward_context: the forward recomputed under wp.Tape as
mjw.step(m, D[k], D[k+1]); tape.zero(); the output cotangent into D[-1].qpos.grad / qvel.grad; tape.backward()]; the
gradient is D[0].qpos.grad, D[0].qvel.grad and the sum of D[k].ctrl.grad over the substeps (ctrl is held). op vjp =
fwd graph + bwd graph, every call from the loaded states. gradient() runs the bwd graph for the given cotangents and
returns the harness convention (qpos in tangent form via base.qpos_tangent). No Jacobian API (jac not_supported).

Variants: default (capacity "procedure" as mjw), matched (64 contacts / 288 rows per world, NablaEngine's
capacity), warn_none (opt.warn_overflow = 0, as NablaEngine's reference _Sim).
Options as mjw_common.MjwBase (capacity, capacity_file, nconmax, naconmax, njmax, warn).
"""

import warp as wp

from benchmarks.bench.adapters import mjw_common

TREES = {  # mjw_common.tree_sha256 of the mujoco_warp package (setup.sh's checkouts; git commit for reference)
    "mjw_pr1535": dict(commit="357a75d60a56d67d476942a1b6e54b3045ee8e87",
                       tree_sha256="58f6458cb8065c9dea0b6a304723c112adc8859c2d8f4e21caaec47582a20481"),
    "mjw_pr1535_fix": dict(commit="4af093f02a3f48f0009ca7c916674b928136f310",
                           tree_sha256="e8978fc74dc912b685810aae3429538dc0c6e4844b6a8335b491edf7b30f437a"),
}


@wp.kernel(enable_backward=False)
def _accum(a: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w, i = wp.tid()
    out[w, i] += a[w, i]


class MjwPrAdapter(mjw_common.MjwAdjointBase):
    id = "mjw_pr1535"
    label = "MuJoCo Warp + unmerged PR #1535 @357a75d"
    native = "unmerged-upstream"
    tracks = ("A",)
    variants = {"default": {}, "matched": {"capacity": "matched"}, "warn_none": {"warn": "none"}}

    def _check_source(self, src):
        want = TREES[self.id]
        if src["tree_sha256"] != want["tree_sha256"]:
            raise RuntimeError(f"{self.id}: `import mujoco_warp` resolves to {src['path']} (source hash "
                               f"{src['tree_sha256'][:12]}, git {src.get('git')}), not the expected tree "
                               f"{want['commit'][:7]} ({want['tree_sha256'][:12]}): check PYTHONPATH / envs.toml "
                               f"(benchmarks/envs/setup.sh --envs mjw-pr)")

    def _build(self, o):
        super()._build(o)
        mjw = self.mjw
        self.D = [self._data() for _ in range(self.substeps + 1)]
        for a in (self.D[0].qpos, self.D[0].qvel, self.D[0].ctrl):
            a.requires_grad = True
        self.bc = mjw.create_backward_context(self.m, self.D[0])

    def _bwd_body(self):
        """The adjoint of one control step (module docstring; contrib/diffsim/_rollout.py's _backward_step)."""
        mjw, D = self.mjw, self.D
        self._load(D[0])
        tape = wp.Tape()
        with mjw.backward_context(self.bc):
            with tape:
                for k in range(self.substeps):
                    mjw.step(self.m, D[k], D[k + 1])
            tape.zero()
            wp.copy(D[-1].qpos.grad, self.cq)
            wp.copy(D[-1].qvel.grad, self.cv)
            tape.backward()
        wp.copy(self.gq, D[0].qpos.grad)
        wp.copy(self.gv, D[0].qvel.grad)
        self.gu.zero_()
        for k in range(self.substeps):
            wp.launch(_accum, dim=self.gu.shape, inputs=[D[k].ctrl.grad], outputs=[self.gu])

    def _bwd(self):
        self._launch("bwd")

    def _vjp(self):
        self._launch("fwd")
        self._launch("bwd")

    def _ops_extra(self):
        return dict(fwd_train=self._fwd, bwd=self._bwd, vjp=self._vjp)

    def _prepare_extra(self, op):
        out = {}
        if op in ("fwd_train", "vjp"):
            out.update(self._capture("fwd", self._fwd_body))
        if op in ("bwd", "vjp"):
            for k, v in self._capture("bwd", self._bwd_body, first="compile_s").items():
                out[k] = out.get(k, 0.0) + v
        return out

    def semantics(self):
        fix = self.id == "mjw_pr1535_fix"
        return dict(
            gradient="analytic adjoint of the published PR (implicit function theorem at the forward's converged "
                     "Newton solution; the forward recomputed under wp.Tape, full solve), discrete decisions frozen",
            pyramidal=("a pyramidal contact is skipped in the adjoint only when every edge row is satisfied "
                       "(etaoxing/mujoco_warp#1, commit c8c8890)" if fix else
                       "a pyramidal contact is skipped in the adjoint when its first edge row is satisfied, even if "
                       "other edges carry force (as published; fixed by etaoxing/mujoco_warp#1)"),
            clamp=("on dofs whose actuator force is clamped by jnt_actfrcrange the multiplier of the qpos and ctrl "
                   "paths is zeroed (etaoxing/mujoco_warp#1, commit 4af093f); the qvel path keeps the actuator "
                   "damping -kv (NablaEngine clamp_kv 0)" if fix else
                   "no clamp handling: on dofs whose actuator force is clamped the qpos, qvel and ctrl paths keep the "
                   "unclamped actuator derivative (as published)"),
            bwd="the PR's backward step: the forward recomputed under wp.Tape from the inputs (the PR keeps no "
                "residuals), then tape.backward; vjp = fwd + bwd",
            ctrl="ctrl held over the substeps: the per-substep ctrl gradients summed",
            qpos="MuJoCo Warp's qpos-form gradient mapped to the 35-dimensional tangent form (base.qpos_tangent)",
            solver_exit="per batch (graph_conditional); the forward with mjw.enable_grad() (backward hooks, safe "
                        "sqrt)")

    def _provenance_extra(self):
        return dict(expected_tree=TREES[self.id], enable_grad=True,
                    backward="contrib/diffsim/_rollout.py pattern: chain of substeps + 1 Data, wp.Tape under "
                             "mjw.backward_context, captured as one CUDA graph")


class MjwPrFixAdapter(MjwPrAdapter):
    id = "mjw_pr1535_fix"
    label = "MuJoCo Warp + unmerged PR #1535 @357a75d + etaoxing/mujoco_warp#1 (4af093f)"
    native = "labelled"


ADAPTERS = {"mjw_pr1535": MjwPrAdapter, "mjw_pr1535_fix": MjwPrFixAdapter}
