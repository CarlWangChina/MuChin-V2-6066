from Code_for_Experiment.Targeted_Training.data_prep_demucs.demucs.grids._explorers import MyExplorer
from dora import Launcher

@MyExplorer
def explorer(launcher: Launcher):
    launcher.slurm_(gpus=8, time=3 * 24 * 60, partition="speechgpt,learnfair", mem_per_gpu=None, constraint='')
    launcher.bind_({"dset.use_musdb": False})
    with launcher.job_array():
        launcher(dset='sdx23_bleeding')
        launcher(dset='sdx23_labelnoise')