# Contact capacity qualification

The 96-contact / 416-row extension passes bounded tests of storage, native execution,
contact Jacobians and Hessians on one NVIDIA CMP 170HX. It does not establish that
96 contacts cover every possible state. Callers must keep the overflow checks active;
the NablaMimic reproduction rejects a training update with lost physics.

The model is the current donor G1: 36 qpos coordinates, 35 velocity coordinates,
29 actuators, 31 bodies and 502 declared collision pairs. The extension keeps its
collision geometry, physical constants, Newton settings and adjoint options.
Contact sets use three 32-bit tiles; DOF masks retain their original meaning.
Hessian accumulation keeps all world-contact groups before self contacts.
A tape record grows from 1472 to 1504 float32 words, with 13 row-zone words and
the same 7/9/8/8-bit header. Old libraries must be rebuilt.

The source and dependency revisions, input arrays, complete diagnostics and failed
conditions are recorded beside the provisioned runs. The public donor's
[capacity study](https://github.com/LingzheZhao/NablaMimic/tree/codex/pytorch-nablaengine/studies/pytorch)
contains the runnable programs and frozen protocol. Numerical margins were fixed
before execution; each condition used a fresh output root.

| Condition | Result | What it establishes |
| --- | --- | --- |
| Current model and tape | 36 checks passed | Capacity bounds, actual generated layout, fresh boundary fields through contact 95 and row 415, and rejection of the old tape width |
| Same-build native interfaces | 122 checks passed | JAX/Torch forward, tape, kinematics, VJP, Inductor and changed-input CUDA Graph execution |
| Prior 64/288 versus current 96/416 | 999 checks passed; all observed array differences zero | Eighty shared first/midpoint reference states with fixed friction/COM samples and cotangents, four substeps and ten Newton iterations; active contact order and zones matched exactly |
| Expanded collision stress | 26 checks passed | 43,040 manufactured states, all real contact counts 65–96 represented, 212 states with exactly 96 contacts, and 3,600 states above 96 correctly flagged; maximum raw count 136 |
| High-contact interfaces | 61 checks passed | One valid state for every count 65–96; native outputs and VJPs bitwise across Torch/JAX, eager/Inductor and two changed-input/cotangent graph replays |
| Independent device Jacobian/Hessian | 67 checks passed | Ten manufactured-contact counts through 97, mixed world/self contacts in every tile, exact DOF support, and nonzero contact-95 contribution |

The expanded collision pool uses 1,024 seeded poses with hinge angles in [-pi,pi],
321 opposing hip/ankle-roll poses and 32 root heights in [-2.0,0.9] m. These include
joint-limit violations and deep floor penetration and are capacity tests, not
tracking demonstrations. All raw rows and overflow cases remain saved. The first
5,120-state pool reached only 82 contacts and failed its required overflow coverage;
that failed condition remains excluded, with unchanged thresholds. The larger pool
is a separate condition. Valid-world derivative checks exclude the explicitly
flagged overflow group while retaining its evidence.

The independent oracle uses MuJoCo double-precision point Jacobians and mass matrix
with a dense mass-plus-constraint Hessian. Only collision discovery is replaced by
manufactured contacts; the actual add-contact, contact-Jacobian and Hessian code is
retained. Float checks use rtol 1e-4 and atol 1e-5. Same-build VJP agreement checks
interface and execution consistency; it is not an independent finite-difference
accuracy result for every high-contact trajectory. Finite iteration-limit,
nonconvergence and singular-contact diagnostics remain visible, so passing does
not assert Newton convergence.

A separate full-size Torch smoke completed twelve updates at 4,096 training and
collector worlds, 64-control windows and 1,000 replay rows. Mean update time was
0.862154 seconds, compared with 0.867294 seconds in the earlier 64-contact smoke;
this pair of short observations does not establish a speed improvement. Peak
allocated/reserved memory was 31.815/38.076 GiB, with every invalid-state, capacity,
nonfinite-gradient and replay-mismatch counter zero. Setup took 167.394 seconds
and the complete child process took 215.278 seconds. The production compiled versus
CUDA Graph training check passed 1,923 checks, including 612 bitwise-identical arrays
and independent collector-history copy checks. Earlier strict eager/compiler
failures remain separate and are not reclassified.

These capacity results qualify the declared condition for a learning experiment.
A subsequent three-hour Torch run completed 12,640 updates with no recorded native
invalidations; its policy returns and separate body-fidelity limits are reported in
[the donor reproduction results](https://github.com/LingzheZhao/NablaMimic/blob/codex/pytorch-nablaengine/REPRODUCTION.md).
This does not establish every possible state's capacity needs, other GPU models,
or the original paper's cross-simulator throughput figures.

A later update from the row-capacity branch adds `nefc` to the Torch artifact metadata.
All eight physics/code-generation files and all 3,376 package string constants,
including CUDA templates, remain identical to the qualified kernel revision.
A fresh one-world artifact check passed: interface constants, standalone loading
without MuJoCo/JAX/NVCC, bitwise forward/backward outputs and refusal of a stale-source
artifact. Serialized metadata records 96 contacts, 416 rows and 1504 tape words;
the fixture's conservative row requirement is 413. This is metadata/interface
validation, not a second training experiment. The optional matched Warp benchmark
and tape-evidence requests at 96/416 remain unvalidated and carry no new throughput claim.
