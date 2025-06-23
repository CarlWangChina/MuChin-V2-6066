from ._explorers import MyExplorer
from fix_shell_js_paths import main

TRACK_B = ['e51eebcc', 'a1d90b5c', '5d2d6c55', 'cfa93e08']

@MyExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=8,
        time=3 * 24 * 60,
        partition='learnlab'
    )
    for sig in TRACK_B:
        while sig is not None:
            xp = main.get_xp_from_sig(sig)
            sig = xp.cfg.continue_from
        for dset in ['extra44', 'extra_test']:
            sub = launcher.bind(xp.argv, dset=dset)
            sub()
            if dset == 'extra_test':
                sub({'quant.diffq': 1e-4})
                sub({'quant.diffq': 3e-4})