import unittest
import torch
import itertools
from tqdm import tqdm
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams, get_hparams
from ama_prof_divi.modules.diffusion.sampler_list import get_sampler, get_available_samplers
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.samplers.sampler_base import Sampler

logger = logging.getLogger(__name__)

ARG_CHOICES = {
    "ddim": {
        "set_alpha_to_one": [True, False],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "thresholding": [True, False],
        "clip_sample": [True, False],
    },
    "ddpm": {
        "variance_type": ["fixed_small", "fixed_small_log", "fixed_large", "fixed_large_log", "learned", "learned_range"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "thresholding": [True, False],
        "clip_sample": [True, False],
    },
    "dpm++2m": {
        "algorithm_type": ["dpmsolver++", "sde-dpmsolver++", "dpmsolver", "sde-dpmsolver"],
        "solver_type": ["midpoint", "heun"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "variance_type": ["learned", "learned_range", None],
        "thresholding": [True, False],
        "clip_sample": [True, False],
        "euler_at_final": [True, False],
        "lower_order_final": [True, False],
        "solver_order": [1, 2, 3],
        "use_karras_sigmas": [True, False],
        "use_lu_lambdas": [True, False],
    },
    "unipc": {
        "solver_type": ["bh1", "bh2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "use_karras_sigmas": [True, False],
        "predict_x0": [True, False],
        "thresholding": [True, False],
        "solver_order": [1, 2, 3],
        "lower_order_final": [True, False],
    },
    "euler": {
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "interpolation_type": ["linear", "log_linear"],
        "use_karras_sigmas": [True, False],
        "time_step_type": ["discrete", "continuous"],
    },
    "heun": {
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "use_karras_sigmas": [True, False],
        "clip_sample": [True, False],
    },
}

def _get_all_args_combinations(arg_choices: dict) -> list[dict]:
    if len(arg_choices) == 0:
        return [{}]
    arg_names = list(arg_choices.keys())
    product = itertools.product(*arg_choices.values())
    return [{arg_names[i]: arg for i, arg in enumerate(args)} for args in product]

class TestDiffusionSamplers(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        self.device = get_hparams()["ama-prof-divi"]["device"]

    def test_get_available_samplers(self):
        logger.info("test_get_available_samplers")
        args = {}
        sampler_names = get_available_samplers()
        for sampler_name in sampler_names:
            logger.info("Testing sampler %s", sampler_name)
            sampler = get_sampler(sampler_name, args, device=self.device)
            self.assertIsNotNone(sampler)

    def test_all_samplers(self):
        logger.info("test_all_samplers")
        sampler_names = get_available_samplers()
        for sampler_name in sampler_names:
            self.assertTrue(sampler_name in ARG_CHOICES)
            args_list = _get_all_args_combinations(ARG_CHOICES[sampler_name])
            for args in tqdm(args_list, desc="Testing sampler {}".format(sampler_name)):
                if sampler_name == "dpm++2m":
                    if args["solver_order"] == 3 and args["algorithm_type"].startswith("sde-"):
                        continue
                sampler = get_sampler(sampler_name, args, device=self.device)
                self.assertIsNotNone(sampler)
                self._test_sampler_with_data(sampler)
                sampler = get_sampler(sampler_name, args, training=True, device=self.device)
                self._test_add_noise(sampler)

    def _test_sampler_with_data(self, sampler: Sampler):
        time_steps = sampler.time_steps
        self.assertGreater(sampler.num_inference_steps, 0)
        self.assertEqual(sampler.num_inference_steps, len(time_steps))
        x = torch.randn(1, 1, 16).to(self.device)
        if sampler.name == "ddpm" and (sampler.variance_type == "learned_range" or sampler.variance_type == "learned"):
            noise_pred = torch.randn(x.shape[0], x.shape[1] * 2, x.shape[2]).to(self.device)
        else:
            noise_pred = torch.randn(x.shape).to(self.device)
        states = {}
        for t in time_steps:
            x = sampler.scale_model_input(x, t)
            r = sampler.sample(noise_pred, t, x, states=states)["prev_sample"]
            self.assertEqual(r.shape, x.shape)

    def _test_add_noise(self, sampler: Sampler):
        time_steps = sampler.time_steps
        x = torch.randn(1, 1, 32).to(self.device)
        noise = torch.randn(1, 1, 32).to(self.device)
        r = sampler.add_noise(x, noise, time_steps)
        self.assertEqual(r.shape, (len(time_steps), x.shape[1], x.shape[2]))