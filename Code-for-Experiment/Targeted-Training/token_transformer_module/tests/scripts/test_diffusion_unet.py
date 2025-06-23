import unittest
import torch
from ama_prof_divi.utils import init_hparams, post_init_hparams, get_hparams
import logging
from ama_prof_divi.modules.diffusion.unet import UnetModel, UnetModelArgs, ControlNetModel
logger = logging.getLogger(__name__)
UNET_ARGS = UnetModelArgs(
    in_channels=128,
    out_channels=128,
    model_channels=128,
    num_res_blocks=2,
    context_dim=512,
    attention_resolutions=(32, 16, 8, 4, 2, 1),
    dropout=0.0,
    channel_mult=(1, 2, 4, 8, 8, 16, 32),
    conv_resample=True,
    num_heads=8,
    use_transformer=True,
    transformer_depth=1,
    use_scale_shift_norm=False,
    res_block_updown=False,
    use_time_embedding=True,
    dims=1
)

class TestDiffusionUnet(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        self.device = get_hparams()["ama-prof-divi"]["device"]
        self.unet = UnetModel(UNET_ARGS).to(self.device)
        self.controlnet = ControlNetModel(UNET_ARGS).to(self.device)

    def test_unet(self):
        logger.info("Testing UnetModel...")
        self.assertIsNotNone(self.unet)
        x = torch.randn(3, 128, 256).to(self.device)
        time_steps = torch.Tensor([8, 8, 8]).to(self.device)
        context = torch.randn(3, 512, 1).to(self.device)
        y = self.unet(x, time_steps=time_steps, context=context)
        self.assertEqual(x.shape, y.shape)

    def test_controlnet(self):
        logger.info("Testing ControlNetModel...")
        self.assertIsNotNone(self.controlnet)
        x = torch.randn(4, 128, 256).to(self.device)
        time_steps = torch.Tensor([8, 8, 8, 8]).to(self.device)
        condition = torch.randn(x.shape).to(self.device)
        context = torch.randn(4, 512, 1).to(self.device)
        c = self.controlnet(x, time_steps=time_steps, control_condition=condition, context=context)
        y = self.unet(x, time_steps=time_steps, context=context, controlnet_output=c)
        self.assertEqual(x.shape, y.shape)

    def test_blocks(self):
        logger.info("Testing blocks...")
        self.assertIsNotNone(self.unet)
        for block in self.unet.block_desc:
            logger.info(block)
            logger.info("")