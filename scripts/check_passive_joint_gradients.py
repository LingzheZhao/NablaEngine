"""Run the existing generated-state gradient protocol with declared passive forces.

Use the same CLI as check_box_step_gradients.py on provisioned CUDA compute.
The fixture file is never rewritten. AABB dispatch is disabled for this comparison.
"""
from pathlib import Path
import runpy


def main():
    import sys
    import subprocess
    from nablaengine import fixtures, jax_ffi

    # The reused checker dispatches condition=both through child processes.
    # Route those children through this entry point so passive setup is retained.
    launch = subprocess.run
    checker = str(Path(__file__).with_name("check_box_step_gradients.py"))
    def passive_launch(command, *args, **kwargs):
        if isinstance(command, list) and len(command)>1 and command[0]==sys.executable and command[1]==checker:
            command=[command[0],str(Path(__file__).resolve()),*command[2:]]
        return launch(command,*args,**kwargs)
    subprocess.run=passive_launch

    load_model = fixtures.load_model
    def passive_model(*args, **kwargs):
        model, metadata = load_model(*args, **kwargs)
        model.dof_damping[6:] = .5
        model.dof_frictionloss[6:] = .1
        assert (model.dof_damping[6:]==.5).all() and (model.dof_frictionloss[6:]==.1).all()
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
