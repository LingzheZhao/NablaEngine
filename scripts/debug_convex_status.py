"""Run the terrain scene checker with per-pair convex failure diagnostics.

Use exactly the arguments of check_terrain_scene.py, with a fresh output root.
This changes generated compilation units only by enabling diagnostic printing.
Select and record a fixed engine revision before running this script.
"""
from pathlib import Path
import runpy


def main():
    from nablaengine import jax_ffi

    generate_sources = jax_ffi.library_sources

    def diagnostic_sources(*args, **kwargs):
        units, constants, defines = generate_sources(*args, **kwargs)
        units = {name: '#define DK_CONVEX_STATUS_DEBUG 1\n' + source
                 for name, source in units.items()}
        return units, constants, defines

    jax_ffi.library_sources = diagnostic_sources
    runpy.run_path(str(Path(__file__).with_name('check_terrain_scene.py')), run_name='__main__')


if __name__ == '__main__':
    main()
