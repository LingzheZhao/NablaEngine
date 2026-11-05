# Contact capacity qualification

The current default is 128 active contacts and 576 constraint rows. Candidate pairs and collision geoms have
16-bit indices, independently of the active-contact capacity. The row allocation covers 128 condim-3 contacts,
35 dof frictionloss rows and 29 hinge-limit rows. Tape format 4 uses uint32 contact entries and an 8/10/6/8-bit
count header; Torch artifacts use version 4. Current primitives emit slots 0 through 7.

## Current sparse-contact candidate-index check

At the [qualified implementation](https://github.com/LingzheZhao/NablaEngine/tree/41c820ae0b9ee9533f2625e2a1e7b2f5270d6167),
`scripts/check_pair_capacity.py` passed all three generated model conditions on an NVIDIA GeForce RTX 3090,
with MuJoCo 3.17.0, NumPy 2.5.1 and PyTorch 2.13.0+cu129. Each condition ran one and four controls, with four
substeps per control and a 50-iteration Newton cap, starting at the script's `qpos0_asymmetric_contact` state
with foot friction 1 and zero torso COM offset. Both horizons passed forward comparison with MuJoCo and all nine
directional-gradient conditions, using double-precision MuJoCo finite differences at steps 1e-5 and 5e-6.
Forward position/velocity margins are 2e-4/2e-2; gradient margins are atol 0.003 and rtol 0.03.

The large input has 577 total geoms, 542 collision geoms and 15,231 unique explicit pairs. Its 15,184 distant
box/robot pairs expand the candidate table; a physically equivalent 47-pair control retains the same geoms and
active support. The original 502-pair fixture is a third condition. These generated inputs match the intended
full-scene index counts; they are not the source G1 with all six CoACD terrains. All conditions actually use 28
active contacts and 112 rows per substep. High active pair IDs are 2547–2553 and 11691–11697. At both horizons,
the large model agrees with the 47-pair control in final position, velocity and VJP under the declared margins.
The large model's largest generalized-position/velocity differences from MuJoCo are 8.64e-8/8.10e-6 after one
control and 2.47e-7/4.76e-6 after four. All unmodified tape and backward flags are zero, with no contact/row
overflow, nonconvergence, singularity, nonfinite gradient or replay mismatch.

Negative controls mask actual tape pair IDs to eight or eleven bits. Both masks change 112 contact entries at
one control and 448 at four controls; all four conditions trigger replay mismatches. Their intentionally invalid
backward diagnostics remain separate from the clean positive checks.

For 256 repeated worlds and four substeps, the large model's forward-plus-backward CUDA-event time has median
3.292 ms over ten measurements after three warmups; the 47-pair control reads 1.396 ms and the original fixture
1.022 ms. This is one control's sparse-contact execution, excluding host transfers, with no speed pass threshold.
Compiled sizes are 94,240 bytes for `Sim`, 24,544 for `Adj` and 6400 per tape record. The split forward requests
52,848 dynamic shared bytes for `k_pre` and 8176 for `k_post`, with zero static shared bytes; both kernels opt in
to their requested sizes. The device's per-block opt-in limit is 101,376 bytes.

An earlier check compiled the large model but failed its first forward launch because `k_pre` requested more
than the default 48 KiB without opting in. The corrected check above retains the same physical margins. These
results qualify wide candidate indices with sparse active contacts. They do not qualify dense 128-contact
execution, all-six-terrain dynamics, task fidelity or full training throughput.

## Earlier 96-contact / 416-row qualification

The 96-contact / 416-row extension passes bounded tests of storage, native execution,
contact Jacobians and Hessians on one NVIDIA CMP 170HX. It does not establish that
96 contacts cover every possible state. Callers must keep the overflow checks active;
the NablaMimic reproduction rejects a training update with lost physics.

The model in that qualification is the donor G1: 36 qpos coordinates, 35 velocity coordinates,
29 actuators, 31 bodies and 502 declared collision pairs. The extension keeps its
collision geometry, physical constants, Newton settings and adjoint options.
Contact sets use three 32-bit tiles; DOF masks retain their original meaning.
Hessian accumulation keeps all world-contact groups before self contacts.
A tape record grows from 1472 to 1504 float32 words, with 13 row-zone words and
the same 7/9/8/8-bit header. Old libraries must be rebuilt.

For that qualification, the source and dependency revisions, input arrays, complete diagnostics and failed
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
