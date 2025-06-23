import unittest
import torch
import diffusers
import inspect

from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams, get_hparams
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.sampler_list import get_sampler

_samplerlogger = logging.get_logger(__name__)
BETA_SCHEDULE_CHOICES = ["linear", "scaled_linear", "squaredcos_cap_v2"]

PREDICTION_TYPE_CHOICES = ["epsilon", "sample", "v_prediction"]

TIMESTEP_SPACING_CHOICES = ["linspace", "leading", "trailing"]

NUM_TRAINING_STEPS = 1000
NUM_INFERENCE_STEPS = 20
BETA_START = 1e-4
BETA_END = 0.02
STEPS_OFFSET = 0
THRESHOLDING = True
DYNAMIC_THRESHOLDING_RATIO = 0.995
CLIP_SAMPLE = True
CLIP_SAMPLE_RANGE = 1.0
SOLVER_ORDER = 2
USE_KARRAS_SIGMAS = False
USE_LU_LAMBDAS = False
DDPM_VARIANCE_TYPE = "fixed_small"
DPM2M_ALGORITHM_TYPE = "dpmsolver++"
DPM2M_SOLVER_TYPE = "midpoint"
DPM2M_USE_EULAR_AT_FINAL = False
DPM2M_USE_LOWER_ORDER_FINAL = True
DPM2M_FINAL_SIGMAS_TYPE = "zero"
EULER_INTERPOLATION_TYPE = "linear"
EULER_TIME_STEP_TYPE = "discrete"
HEUN_CLIP_SAMPLE = True

class TestDiffusionSamplers2(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        self.device = get_hparams()["ama-prof-divi"]["device"]

    def _get_samplers_on_choices(self, get_sampler_func_name: str, training: bool, **kwargs):
        method = getattr(self, get_sampler_func_name)
        self.assertIsNotNone(method)
        for beta_schedule in BETA_SCHEDULE_CHOICES:
            for prediction_type in PREDICTION_TYPE_CHOICES:
                for timestep_spacing in TIMESTEP_SPACING_CHOICES:
                    with self.subTest(training=training, beta_schedule=beta_schedule, prediction_type=prediction_type, timestep_spacing=timestep_spacing):
                        m_sampler, d_sampler = method(beta_schedule=beta_schedule, prediction_type=prediction_type, timestep_spacing=timestep_spacing, training=training, **kwargs)
                        yield m_sampler, d_sampler

    def _get_samplers_ddim(self, beta_schedule: str, prediction_type: str, timestep_spacing: str, training: bool, **kwargs):
        d_ddim = diffusers.schedulers.DDIMScheduler(
            num_train_timesteps=NUM_TRAINING_STEPS,
            beta_start=BETA_START,
            beta_end=BETA_END,
            beta_schedule=beta_schedule,
            steps_offset=STEPS_OFFSET,
            thresholding=THRESHOLDING,
            dynamic_thresholding_ratio=DYNAMIC_THRESHOLDING_RATIO,
            clip_sample=CLIP_SAMPLE,
            clip_sample_range=CLIP_SAMPLE_RANGE,
            timestep_spacing=timestep_spacing,
            prediction_type=prediction_type
        )
        if not training:
            d_ddim.set_timesteps(num_inference_steps=NUM_INFERENCE_STEPS)
        m_ddim_args = {
            "num_training_steps": NUM_TRAINING_STEPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "beta_schedule": beta_schedule,
            "steps_offset": STEPS_OFFSET,
            "thresholding": THRESHOLDING,
            "dynamic_thresholding_ratio": DYNAMIC_THRESHOLDING_RATIO,
            "clip_sample": CLIP_SAMPLE,
            "clip_sample_range": CLIP_SAMPLE_RANGE,
            "timestep_spacing": timestep_spacing,
            "prediction_type": prediction_type
        }

        m_ddim = get_sampler("ddim", m_ddim_args, training=training, device=self.device)
        return m_ddim, d_ddim

    def _get_samplers_ddpm(self, beta_schedule: str, prediction_type: str, timestep_spacing: str, training: bool, **kwargs):
        d_ddpm = diffusers.schedulers.DDPMScheduler(
            num_train_timesteps=NUM_TRAINING_STEPS,
            beta_start=BETA_START,
            beta_end=BETA_END,
            beta_schedule=beta_schedule,
            variance_type=DDPM_VARIANCE_TYPE,
            steps_offset=STEPS_OFFSET,
            thresholding=THRESHOLDING,
            dynamic_thresholding_ratio=DYNAMIC_THRESHOLDING_RATIO,
            clip_sample=CLIP_SAMPLE,
            clip_sample_range=CLIP_SAMPLE_RANGE,
            timestep_spacing=timestep_spacing,
            prediction_type=prediction_type,
        )
        if not training:
            d_ddpm.set_timesteps(num_inference_steps=NUM_INFERENCE_STEPS)
        m_ddpm_args = {
            "num_training_steps": NUM_TRAINING_STEPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "beta_schedule": beta_schedule,
            "variance_type": DDPM_VARIANCE_TYPE,
            "steps_offset": STEPS_OFFSET,
            "thresholding": THRESHOLDING,
            "dynamic_thresholding_ratio": DYNAMIC_THRESHOLDING_RATIO,
            "clip_sample": CLIP_SAMPLE,
            "clip_sample_range": CLIP_SAMPLE_RANGE,
            "timestep_spacing": timestep_spacing,
            "prediction_type": prediction_type
        }

        m_ddpm = get_sampler("ddpm", m_ddpm_args, training=training, device=self.device)
        return m_ddpm, d_ddpm

    def _get_samplers_unipc(self, beta_schedule: str, prediction_type: str, timestep_spacing: str, training: bool, **kwargs):
        d_unipc = diffusers.schedulers.UniPCMultistepScheduler(
            num_train_timesteps=NUM_TRAINING_STEPS,
            beta_start=BETA_START,
            beta_end=BETA_END,
            beta_schedule=beta_schedule,
            solver_order=SOLVER_ORDER,
            steps_offset=STEPS_OFFSET,
            thresholding=THRESHOLDING,
            dynamic_thresholding_ratio=DYNAMIC_THRESHOLDING_RATIO,
            timestep_spacing=timestep_spacing,
            prediction_type=prediction_type,
            use_karras_sigmas=USE_KARRAS_SIGMAS
        )
        if not training:
            d_unipc.set_timesteps(num_inference_steps=NUM_INFERENCE_STEPS)
        m_unipc_args = {
            "num_training_steps": NUM_TRAINING_STEPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "beta_schedule": beta_schedule,
            "solver_order": SOLVER_ORDER,
            "steps_offset": STEPS_OFFSET,
            "thresholding": THRESHOLDING,
            "dynamic_thresholding_ratio": DYNAMIC_THRESHOLDING_RATIO,
            "timestep_spacing": timestep_spacing,
            "prediction_type": prediction_type,
            "use_karras_sigmas": USE_KARRAS_SIGMAS
        }

        m_unipc = get_sampler("unipc", m_unipc_args, training=training, device=self.device)
        return m_unipc, d_unipc

    def _get_samplers_dpm2m(self, beta_schedule: str, prediction_type: str, timestep_spacing: str, training: bool, **kwargs):
        d_dpm2m = diffusers.schedulers.DPMSolverMultistepScheduler(
            num_train_timesteps=NUM_TRAINING_STEPS,
            beta_start=BETA_START,
            beta_end=BETA_END,
            beta_schedule=beta_schedule,
            solver_order=SOLVER_ORDER,
            lower_order_final=DPM2M_USE_LOWER_ORDER_FINAL,
            euler_at_final=DPM2M_USE_EULAR_AT_FINAL,
            steps_offset=STEPS_OFFSET,
            thresholding=THRESHOLDING,
            dynamic_thresholding_ratio=DYNAMIC_THRESHOLDING_RATIO,
            algorithm_type=DPM2M_ALGORITHM_TYPE,
            solver_type=DPM2M_SOLVER_TYPE,
            timestep_spacing=timestep_spacing,
            prediction_type=prediction_type,
            use_karras_sigmas=USE_KARRAS_SIGMAS,
            use_lu_lambdas=USE_LU_LAMBDAS,
            final_sigmas_type=DPM2M_FINAL_SIGMAS_TYPE
        )
        if not training:
            d_dpm2m.set_timesteps(num_inference_steps=NUM_INFERENCE_STEPS)
        m_dpm2m_args = {
            "num_training_steps": NUM_TRAINING_STEPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "beta_schedule": beta_schedule,
            "solver_order": SOLVER_ORDER,
            "lower_order_final": DPM2M_USE_LOWER_ORDER_FINAL,
            "euler_at_final": DPM2M_USE_EULAR_AT_FINAL,
            "steps_offset": STEPS_OFFSET,
            "thresholding": THRESHOLDING,
            "dynamic_thresholding_ratio": DYNAMIC_THRESHOLDING_RATIO,
            "timestep_spacing": timestep_spacing,
            "prediction_type": prediction_type,
            "use_karras_sigmas": USE_KARRAS_SIGMAS,
            "use_lu_lambdas": USE_LU_LAMBDAS,
            "final_sigmas_type": DPM2M_FINAL_SIGMAS_TYPE
        }

        m_dpm2m = get_sampler("dpm++2m", m_dpm2m_args, training=training, device=self.device)
        return m_dpm2m, d_dpm2m

    def _get_samplers_euler(self, beta_schedule: str, prediction_type: str, timestep_spacing: str, training: bool, **kwargs):
        d_euler = diffusers.schedulers.EulerDiscreteScheduler(
            num_train_timesteps=NUM_TRAINING_STEPS,
            beta_start=BETA_START,
            beta_end=BETA_END,
            beta_schedule=beta_schedule,
            steps_offset=STEPS_OFFSET,
            interpolation_type=EULER_INTERPOLATION_TYPE,
            timestep_spacing=timestep_spacing,
            prediction_type=prediction_type,
            use_karras_sigmas=USE_KARRAS_SIGMAS,
            timestep_type=EULER_TIME_STEP_TYPE,
            rescale_betas_zero_snr=False
        )
        if not training:
            d_euler.set_timesteps(num_inference_steps=NUM_INFERENCE_STEPS)
        m_euler_args = {
            "num_training_steps": NUM_TRAINING_STEPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "beta_schedule": beta_schedule,
            "steps_offset": STEPS_OFFSET,
            "interpolation_type": EULER_INTERPOLATION_TYPE,
            "timestep_spacing": timestep_spacing,
            "prediction_type": prediction_type,
            "use_karras_sigmas": USE_KARRAS_SIGMAS,
            "time_step_type": EULER_TIME_STEP_TYPE
        }

        m_euler = get_sampler("euler", m_euler_args, training=training, device=self.device)
        return m_euler, d_euler

    def _get_samplers_heun(self, beta_schedule: str, prediction_type: str, timestep_spacing: str, training: bool, **kwargs):
        d_heun = diffusers.schedulers.HeunDiscreteScheduler(
            num_train_timesteps=NUM_TRAINING_STEPS,
            beta_start=BETA_START,
            beta_end=BETA_END,
            beta_schedule=beta_schedule,
            steps_offset=STEPS_OFFSET,
            timestep_spacing=timestep_spacing,
            prediction_type=prediction_type,
            use_karras_sigmas=USE_KARRAS_SIGMAS,
            clip_sample=HEUN_CLIP_SAMPLE
        )
        if not training:
            d_heun.set_timesteps(num_inference_steps=NUM_INFERENCE_STEPS)
        m_heun_args = {
            "num_training_steps": NUM_TRAINING_STEPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "beta_schedule": beta_schedule,
            "steps_offset": STEPS_OFFSET,
            "timestep_spacing": timestep_spacing,
            "prediction_type": prediction_type,
            "use_karras_sigmas": USE_KARRAS_SIGMAS,
            "clip_sample": HEUN_CLIP_SAMPLE
        }

        m_heun = get_sampler("heun", m_heun_args, training=training, device=self.device)
        return m_heun, d_heun

    def _test_training(self, m_sampler, d_sampler):
        original_samples = torch.randn(10, 64, 200)
        noise = torch.randn(10, 64, 200)
        time_steps = torch.randint(0, 1000, (10,))
        _samplerlogger.info(f"Test training: sampler={m_sampler.name}, beta_schedule={m_sampler.beta_schedule}, prediction_type={m_sampler.prediction_type}, timestep_spacing={m_sampler.timestep_spacing}")
        mn = m_sampler.add_noise(original_samples=original_samples, noise=noise, time_steps=time_steps)
        dn = d_sampler.add_noise(original_samples=original_samples, noise=noise, timesteps=time_steps)
        diff = (mn.cpu() - dn).abs().max()
        self.assertLess(diff, 1e-3, f"Diff {diff} is too large.")

    def _self_inference(self, m_sampler, d_sampler):
        model_output = torch.randn(10, 64, 200)
        sample = torch.randn(10, 64, 200)
        _samplerlogger.info(f"Test inference: sampler={m_sampler.name}, beta_schedule={m_sampler.beta_schedule}, prediction_type={m_sampler.prediction_type}, timestep_spacing={m_sampler.timestep_spacing}")
        sig = inspect.signature(d_sampler.step)
        generator_supported = "generator" in sig.parameters
        generator = torch.Generator()
        states = {}

        for t in m_sampler.time_steps:
            generator.manual_seed(t.long().item())
            scaled_model_output = m_sampler.scale_model_input(model_output.to(m_sampler.device), t.to(m_sampler.device))
            m_out = m_sampler.sample(model_output=scaled_model_output, time_step=t.to(m_sampler.device), sample=sample.to(m_sampler.device), generator=generator, states=states)["prev_sample"]

            generator.manual_seed(t.long().item())
            scaled_model_output = d_sampler.scale_model_input(model_output.cpu(), t.cpu())
            if generator_supported:
                d_out = d_sampler.step(model_output=scaled_model_output, timestep=t.cpu(), sample=sample.cpu(), generator=generator).prev_sample
            else:
                d_out = d_sampler.step(model_output=scaled_model_output, timestep=t.cpu(), sample=sample.cpu()).prev_sample
            diff = (m_out.cpu() - d_out).abs().max()
            self.assertLess(diff, 1e-2, f"Diff {diff} is too large.")

    def test_sampler_ddim_training(self):
        _samplerlogger.info("Testing the DDIM sampler for training.")
        for m_ddim, d_ddim in self._get_samplers_on_choices("_get_samplers_ddim", training=True):
            self.assertTrue(m_ddim.training_mode)
            self.assertTrue(torch.allclose(m_ddim.betas.cpu(), d_ddim.betas))
            self.assertTrue(torch.allclose(m_ddim.alphas.cpu(), d_ddim.alphas))
            self.assertTrue(torch.allclose(m_ddim.time_steps.cpu(), d_ddim.timesteps))
            self._test_training(m_ddim, d_ddim)

    def test_sampler_ddim_inference(self):
        _samplerlogger.info("Testing the DDIM sampler for inference.")
        for m_ddim, d_ddim in self._get_samplers_on_choices("_get_samplers_ddim", training=False):
            self.assertFalse(m_ddim.training_mode)
            self.assertTrue(torch.allclose(m_ddim.time_steps.cpu(), d_ddim.timesteps), f"m_ddim.time_steps = {m_ddim.time_steps}, d_ddim.timesteps = {d_ddim.timesteps}")
            self._self_inference(m_ddim, d_ddim)

    def test_sampler_ddpm_training(self):
        _samplerlogger.info("Testing the DDPM sampler for training.")
        for m_ddpm, d_ddpm in self._get_samplers_on_choices("_get_samplers_ddpm", training=True):
            self.assertTrue(m_ddpm.training_mode)
            self.assertTrue(torch.allclose(m_ddpm.betas.cpu(), d_ddpm.betas))
            self.assertTrue(torch.allclose(m_ddpm.alphas.cpu(), d_ddpm.alphas))
            self.assertTrue(torch.allclose(m_ddpm.time_steps.cpu(), d_ddpm.timesteps.long()))
            self._test_training(m_ddpm, d_ddpm)

    def test_sampler_ddpm_inference(self):
        _samplerlogger.info("Testing the DDPM sampler for inference.")
        for m_ddpm, d_ddpm in self._get_samplers_on_choices("_get_samplers_ddpm", training=False):
            self.assertFalse(m_ddpm.training_mode)
            self.assertTrue(torch.allclose(m_ddpm.time_steps.cpu(), d_ddpm.timesteps.long()), f"m_ddpm.time_steps = {m_ddpm.time_steps}, d_ddpm.timesteps = {d_ddpm.timesteps}")
            self._self_inference(m_ddpm, d_ddpm)

    def test_sampler_unipc_training(self):
        _samplerlogger.info("Testing the UniPC sampler for training.")
        for m_unipc, d_unipc in self._get_samplers_on_choices("_get_samplers_unipc", training=True):
            self.assertTrue(m_unipc.training_mode)
            self.assertTrue(torch.allclose(m_unipc.betas.cpu(), d_unipc.betas))
            self.assertTrue(torch.allclose(m_unipc.alphas.cpu(), d_unipc.alphas))
            self.assertTrue(torch.allclose(m_unipc.sigmas.cpu(), d_unipc.sigmas, atol=1e-4))
            self.assertTrue(torch.allclose(m_unipc.time_steps.cpu(), d_unipc.timesteps.long()))
            self._test_training(m_unipc, d_unipc)

    def test_sampler_unipc_inference(self):
        _samplerlogger.info("Testing the UniPC sampler for inference.")
        for m_unipc, d_unipc in self._get_samplers_on_choices("_get_samplers_unipc", training=False):
            self.assertFalse(m_unipc.training_mode)
            self.assertTrue(torch.allclose(m_unipc.sigmas.cpu(), d_unipc.sigmas, atol=1e-4))
            self.assertTrue(torch.allclose(m_unipc.time_steps.cpu(), d_unipc.timesteps.long()), f"m_unipc.time_steps = {m_unipc.time_steps}, d_unipc.timesteps = {d_unipc.timesteps}")
            self._self_inference(m_unipc, d_unipc)

    def test_sampler_dpm2m_training(self):
        _samplerlogger.info("Testing the DPM2M sampler for training.")
        for m_dpm2m, d_dpm2m in self._get_samplers_on_choices("_get_samplers_dpm2m", training=True):
            self.assertTrue(m_dpm2m.training_mode)
            self.assertTrue(torch.allclose(m_dpm2m.betas.cpu(), d_dpm2m.betas))
            self.assertTrue(torch.allclose(m_dpm2m.alphas.cpu(), d_dpm2m.alphas))
            self.assertTrue(torch.allclose(m_dpm2m.sigmas.cpu(), d_dpm2m.sigmas, atol=1e-4))
            self.assertTrue(torch.allclose(m_dpm2m.time_steps.cpu(), d_dpm2m.timesteps.long()))
            self._test_training(m_dpm2m, d_dpm2m)

    def test_sampler_dpm2m_inference(self):
        _samplerlogger.info("Testing the DPM2M sampler for inference.")
        for m_dpm2m, d_dpm2m in self._get_samplers_on_choices("_get_samplers_dpm2m", training=False):
            self.assertFalse(m_dpm2m.training_mode)
            self.assertTrue(torch.allclose(m_dpm2m.sigmas.cpu(), d_dpm2m.sigmas, atol=1e-4))
            self.assertTrue(torch.allclose(m_dpm2m.time_steps.cpu(), d_dpm2m.timesteps.long()), f"m_dpm2m.time_steps = {m_dpm2m.time_steps}, d_dpm2m.timesteps = {d_dpm2m.timesteps}")
            self._self_inference(m_dpm2m, d_dpm2m)

    def test_sampler_euler_training(self):
        _samplerlogger.info("Testing the Euler sampler for training.")
        for m_euler, d_euler in self._get_samplers_on_choices("_get_samplers_euler", training=True):
            self.assertTrue(m_euler.training_mode)
            self.assertTrue(torch.allclose(m_euler.betas.cpu(), d_euler.betas))
            self.assertTrue(torch.allclose(m_euler.alphas.cpu(), d_euler.alphas))
            self.assertTrue(torch.allclose(m_euler.sigmas.cpu(), d_euler.sigmas, atol=1e-4))
            self.assertTrue(torch.allclose(m_euler.time_steps.cpu(), d_euler.timesteps.long()))
            self._test_training(m_euler, d_euler)

    def test_sampler_euler_inference(self):
        _samplerlogger.info("Testing the Euler sampler for inference.")
        for m_euler, d_euler in self._get_samplers_on_choices("_get_samplers_euler", training=False):
            self.assertFalse(m_euler.training_mode)
            self.assertTrue(torch.allclose(m_euler.sigmas.cpu(), d_euler.sigmas, atol=1e-4))
            self.assertTrue(torch.allclose(m_euler.time_steps.cpu(), d_euler.timesteps), f"m_euler.time_steps = {m_euler.time_steps}, d_euler.timesteps = {d_euler.timesteps}")
            self._self_inference(m_euler, d_euler)

    def test_sampler_heun_training(self):
        _samplerlogger.info("Testing the Heun sampler for training.")
        for m_heun, d_heun in self._get_samplers_on_choices("_get_samplers_heun", training=True):
            self.assertTrue(m_heun.training_mode)
            self.assertTrue(torch.allclose(m_heun.betas.cpu(), d_heun.betas))
            self.assertTrue(torch.allclose(m_heun.alphas.cpu(), d_heun.alphas))
            self.assertTrue(torch.allclose(m_heun.sigmas.cpu(), d_heun.sigmas, atol=1e-4))
            self.assertTrue(torch.allclose(m_heun.time_steps.cpu(), d_heun.timesteps))
            self._test_training(m_heun, d_heun)

    def test_sampler_heun_inference(self):
        _samplerlogger.info("Testing the Heun sampler for inference.")
        for m_heun, d_heun in self._get_samplers_on_choices("_get_samplers_heun", training=False):
            self.assertFalse(m_heun.training_mode)
            self.assertTrue(torch.allclose(m_heun.sigmas.cpu(), d_heun.sigmas, atol=1e-4))
            self.assertTrue(torch.allclose(m_heun.time_steps.cpu(), d_heun.timesteps), f"m_heun.time_steps = {m_heun.time_steps}, d_heun.timesteps = {d_heun.timesteps}")
            self._self_inference(m_heun, d_heun)