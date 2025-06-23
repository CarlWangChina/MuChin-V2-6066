from ._explorers import MyExplorer
from fix_shell_js_paths import main

TRACK_A = ['0d19c1c6', '7ecf8ec1', 'c511e2ab', '7d865c68']

@MyExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=8,
        time=3 * 24 * 60,
        partition='learnlab'
    )
    for sig in TRACK_A:
        xp = main.get_xp_from_sig(sig)
        parent = xp.cfg.continue_from
        xp = main.get_xp_from_sig(parent)
        launcher(xp.argv)
        launcher(xp.argv, {'quant.diffq': 1e-4})
        launcher(xp.argv, {'quant.diffq': 3e-4})