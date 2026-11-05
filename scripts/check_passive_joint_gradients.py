"""Run the existing generated-state gradient protocol with declared passive forces.

Use the same CLI as check_box_step_gradients.py on provisioned CUDA compute.
The fixture file is never rewritten. AABB dispatch is disabled for this comparison.
"""
from pathlib import Path
import runpy


def main():
    import sys
    import subprocess
    import argparse
    from nablaengine import fixtures, jax_ffi

    # The reused checker dispatches condition=both through child processes.
    # Route those children through this entry point so passive setup is retained.
    launch = subprocess.run
    checker = str(Path(__file__).with_name("check_box_step_gradients.py"))
    def passive_launch(command, *args, **kwargs):
        if isinstance(command, list) and len(command)>1 and command[0]==sys.executable and command[1]==checker:
            command=[command[0],str(Path(__file__).resolve()),*command[2:],
                     "--passive-damping",str(selected.passive_damping),
                     "--passive-frictionloss",str(selected.passive_frictionloss)]
        return launch(command,*args,**kwargs)
    subprocess.run=passive_launch

    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--passive-damping",type=float,default=.5)
    parser.add_argument("--passive-frictionloss",type=float,default=.1)
    selected,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    load_model = fixtures.load_model
    def passive_model(*args, **kwargs):
        model, metadata = load_model(*args, **kwargs)
        model.dof_damping[6:] = selected.passive_damping
        model.dof_frictionloss[6:] = selected.passive_frictionloss
        assert (model.dof_damping[6:]==selected.passive_damping).all() and (model.dof_frictionloss[6:]==selected.passive_frictionloss).all()
        return model, metadata
    fixtures.load_model = passive_model
    generate = jax_ffi.library_sources
    def passive_sources(*args, **kwargs):
        units, constants, defines = generate(*args, **kwargs)
        return {name:'#define DK_CONVEX_AABB 0\n'+source for name,source in units.items()},constants,defines
    jax_ffi.library_sources = passive_sources
    runpy.run_path(str(Path(__file__).with_name('check_box_step_gradients.py')),run_name='__main__')


if __name__ == '__main__':
    main()
