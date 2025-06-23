from Code_for_Experiment.Targeted_Training.data_prep_demucs.demucs.grids._explorers import MyExplorer

class MyExplorer(MyExplorer):
    def explorer(self, launcher):
        launcher.slurm_(
            gpus=8,
            time=3 * 24 * 60,
            partition='devlab,learnlab'
        )
        launcher.bind_({'ema.epoch': [0.9, 0.95]})
        launcher.bind_({'ema.batch': [0.9995, 0.9999]})
        launcher.bind_({'epochs': 600})
        base = {'model': 'demucs', 'demucs.dconv_mode': 0, 'demucs.gelu': False, 'demucs.lstm_layers': 2}
        newt = {'model': 'demucs', 'demucs.normalize': True}
        hdem = {'model': 'hdemucs'}
        svd = {'svd.penalty': 1e-5, 'svd': 'base2'}
        with launcher.job_array():
            for model in [base, newt, hdem]:
                sub = launcher.bind(model)
                if model is base:
                    sub(epochs=360)
                    continue
                sub(svd)
                sub(svd, seed=43)
                if model == newt:
                    sub()
                    abl = sub.bind(svd)
                    abl({'ema.epoch': [], 'ema.batch': []})
                    abl({'demucs.dconv_lstm': 10})
                    abl({'demucs.dconv_attn': 10})
                    abl({'demucs.dconv_attn': 10, 'demucs.dconv_lstm': 10, 'demucs.lstm_layers': 2})
                    abl({'demucs.dconv_mode': 0})
                    abl({'demucs.gelu': False})