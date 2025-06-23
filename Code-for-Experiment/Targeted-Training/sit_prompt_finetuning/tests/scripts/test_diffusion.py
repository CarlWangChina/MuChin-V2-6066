import unittest
import torch
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from music_dit2.modules.diffusion import DDIMSampler, DiT, TrainingLoss, Diffusion

logger = get_logger(__name__)

class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_ddim_timesteps(self):
        logger.info("Test ddim_timesteps")
        ddim_sampler = DDIMSampler(beta_start=1e-4,
                                   beta_end=0.02,
                                   beta_schedule='linear',
                                   timestep_spacing='leading',
                                   num_training_timesteps=1000,
                                   device=self.device)
        self.assertEqual(ddim_sampler.betas.size(), (1000,))
        for spacing in ["linspace", "leading", "trailing"]:
            ddim_sampler.set_inference_timesteps(10, spacing)
            self.assertEqual(ddim_sampler.inference_timesteps.size(), (10,))
            logger.info("Time steps for %s: %s", spacing, ddim_sampler.inference_timesteps)

    def test_ddim_add_noise(self):
        logger.info("Test ddim_add_noise")
        ddim_sampler = DDIMSampler(beta_start=1e-4,
                                   beta_end=0.02,
                                   beta_schedule='linear',
                                   timestep_spacing='leading',
                                   num_training_timesteps=1000,
                                   device=self.device)
        x = torch.randn(10, 100, 128, device=self.device)
        noise = torch.randn(x.size(), device=self.device)
        time_steps = torch.randint(0, 1000, (10,), device=self.device)
        x_noisy = ddim_sampler.add_noise(x, noise, time_steps)
        self.assertEqual(x_noisy.size(), x.size())

    def test_ddim_step(self):
        logger.info("Test ddim_step")
        ddim_sampler = DDIMSampler(beta_start=1e-4,
                                   beta_end=0.02,
                                   beta_schedule='linear',
                                   timestep_spacing='leading',
                                   num_training_timesteps=1000,
                                   device=self.device)
        ddim_sampler.set_inference_timesteps(10, "leading")
        dit = DiT(input_dim=64,
                  hidden_dim=128,
                  num_layers=4,
                  num_heads=8,
                  device=self.device)
        x = torch.randn(10, 100, 64, device=self.device)
        condition = torch.randn(10, 100, 128).to(self.device)
        time_step_idx = 0
        model_output = dit(x,
                           timesteps=ddim_sampler.inference_timesteps[time_step_idx],
                           condition=condition)
        x_prev, x_orig = ddim_sampler.denoise(model_output=model_output,
                                              timestep_index=time_step_idx,
                                              sample=x,
                                              eta=1.0)
        self.assertEqual(x_prev.size(), x.size())
        self.assertEqual(x_orig.size(), x.size())

    def test_ddim_training_loss(self):
        logger.info("Test ddim_mse_loss")
        ddim_sampler = DDIMSampler(beta_start=1e-4,
                                   beta_end=0.02,
                                   beta_schedule='linear',
                                   timestep_spacing='leading',
                                   num_training_timesteps=1000,
                                   device=self.device)
        mse_loss = TrainingLoss(sampler=ddim_sampler, loss_type="mse")
        kl_loss = TrainingLoss(sampler=ddim_sampler, loss_type="kl")
        x_start = torch.randn(10, 100, 128, device=self.device)
        noise = torch.randn(x_start.size(), device=self.device)
        noisy_samples = torch.randn(x_start.size(), device=self.device)
        predicted_noise = torch.randn(x_start.size(), device=self.device)
        timesteps = torch.randint(0, 1000, (10,), device=self.device)
        predicted_log_variance = torch.randn(x_start.size(), device=self.device)
        loss = mse_loss(x_start=x_start,
                        noise=noise,
                        noisy_samples=noisy_samples,
                        predicted_noise=predicted_noise,
                        timesteps=timesteps,
                        predicted_log_variance=predicted_log_variance)
        self.assertEqual(loss.dim(), 0)
        loss = kl_loss(x_start=x_start,
                       noise=noise,
                       noisy_samples=noisy_samples,
                       predicted_noise=predicted_noise,
                       timesteps=timesteps,
                       predicted_log_variance=predicted_log_variance)
        self.assertEqual(loss.dim(), 0)

    def test_diffusion_wrapper(self):
        logger.info("Test diffusion wrapper")
        ddim_sampler = DDIMSampler(beta_start=1e-4,
                                   beta_end=0.02,
                                   beta_schedule='linear',
                                   timestep_spacing='leading',
                                   num_training_timesteps=1000,
                                   device=self.device)
        dit = DiT(input_dim=64,
                  hidden_dim=128,
                  num_layers=4,
                  num_heads=8,
                  use_cross_attention=True,
                  context_dim=128,
                  device=self.device)
        training_loss = TrainingLoss(sampler=ddim_sampler, loss_type="mse")
        diffusion = Diffusion(sampler=ddim_sampler,
                              model=dit,
                              training_loss=training_loss)
        samples = torch.randn(10, 100, 64, device=self.device)
        prompt = torch.randn(10, 20, 64, device=self.device)
        conditions = torch.randn(10, 100, 128, device=self.device)
        context = torch.randn(10, 40, 128, device=self.device)
        context_mask = torch.ones(10, 40, device=self.device)
        loss = diffusion.training_step(samples=samples,
                                       conditions=conditions,
                                       context=context,
                                       context_mask=context_mask,
                                       prompt_length=20)
        self.assertEqual(loss.dim(), 0)
        ddim_sampler.set_inference_timesteps(20)
        generated = diffusion.generate(torch.randn(samples.size(), device=self.device),
                                       prompt=prompt,
                                       conditions=conditions,
                                       context=context,
                                       context_mask=context_mask,
                                       eta=0.5)
        self.assertEqual(generated.size(), samples.size())
        generated = diffusion.generate(torch.randn(samples.size(), device=self.device),
                                       prompt=prompt,
                                       conditions=conditions,
                                       context=context,
                                       context_mask=context_mask,
                                       cfg_scale=0.7,
                                       eta=0.5)
        self.assertEqual(generated.size(), samples.size())