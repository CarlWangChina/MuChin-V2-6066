import torch
import unittest
from dataclasses import replace
from einops import rearrange
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams, get_hparams
from ama_prof_divi.modules.embeddings import RotaryPosEmbedding
from ama_prof_divi.modules.transformers import TransformerModelArgs, TransformerEncoder, TransformerDecoder, Generator, InferAccelerationCache

logger = logging.get_logger(__name__)

class TestTransformers(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        self.hparams = get_hparams()
        self.device = "cpu"
        self.args = TransformerModelArgs(
            dim=512,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            max_seq_len=256,
            vocab_size=100,
            kv_dim=None,
        )

    def test_rotary_pos_embedding(self):
        logger.info('test_rotary_pos_embedding')
        rota_pos_emb = RotaryPosEmbedding(dim=512, max_seq_len=self.args.max_seq_len, device=self.device)
        x = torch.ones((1, self.args.max_seq_len, 512)).to(self.device)
        emb = rota_pos_emb(x)
        self.assertEqual(x.shape, emb.shape)

    def test_transformer_encoder(self):
        logger.info('test_transformer_encoder')
        encoder = TransformerEncoder(self.args, device=self.device)
        self.assertIsNotNone(encoder)
        x = torch.randint(0, self.args.vocab_size, (2, self.args.max_seq_len))
        output = encoder(x)
        self.assertEqual(output.shape, (*x.shape, self.args.dim))

    def test_transformer_encoder_with_sinusoidal_pos_embedding(self):
        logger.info('test_transformer_encoder_with_sinusoidal_pos_embedding')
        args = replace(self.args, pos_embedding="sinusoidal")
        encoder = TransformerEncoder(args, device=self.device)
        self.assertIsNotNone(encoder)
        x = torch.randint(0, args.vocab_size, (2, args.max_seq_len))
        output = encoder(x)
        self.assertEqual(output.shape, (*x.shape, args.dim))

    def test_transformer_encoder_mq(self):
        logger.info('test_transformer_encoder_mq')
        args = replace(self.args, num_quantization_groups=8)
        encoder = TransformerEncoder(args, device=self.device)
        self.assertIsNotNone(encoder)
        x = torch.randint(0, args.vocab_size, (2, args.max_seq_len, 8))
        output = encoder(x)
        self.assertEqual(output.shape, (*x.shape[:2], args.dim))

    def test_transformer_encoder_with_sinusoidal_pos_embedding_mq(self):
        logger.info('test_transformer_encoder_with_sinusoidal_pos_embedding_mq')
        args = replace(self.args, pos_embedding="sinusoidal", num_quantization_groups=8)
        encoder = TransformerEncoder(args, device=self.device)
        self.assertIsNotNone(encoder)
        x = torch.randint(0, args.vocab_size, (2, args.max_seq_len, 8))
        output = encoder(x)
        self.assertEqual(output.shape, (*x.shape[:2], args.dim))

    def test_transformer_decoder_self_attention(self):
        logger.info('test_transformer_decoder_self_attention')
        decoder = TransformerDecoder(self.args, device=self.device)
        self.assertIsNotNone(decoder)
        x = torch.randint(0, self.args.vocab_size, (2, self.args.max_seq_len))
        output = decoder(x)
        self.assertEqual(output.shape, (*x.shape, self.args.vocab_size))

    def test_transformer_decoder_self_attention_mq(self):
        logger.info('test_transformer_decoder_self_attention_mq')
        args = replace(self.args, num_quantization_groups=8)
        decoder = TransformerDecoder(args, device=self.device)
        self.assertIsNotNone(decoder)
        x = torch.randint(0, self.args.vocab_size, (2, self.args.max_seq_len, 8))
        output = decoder(x)
        self.assertEqual(output.shape, (*x.shape, self.args.vocab_size))

    def test_transformer_decoder_cross_attention(self):
        logger.info('test_transformer_decoder_cross_attention')
        decoder = TransformerDecoder(self.args, device=self.device)
        x = torch.randint(0, self.args.vocab_size, (2, self.args.max_seq_len))
        context = torch.randn(2, self.args.max_seq_len, self.args.dim).to(self.device)
        output = decoder(x, context=context)
        self.assertEqual(output.shape, (*x.shape, self.args.vocab_size))

    def test_transformer_decoder_cross_attention_mq(self):
        logger.info('test_transformer_decoder_cross_attention_mq')
        args = replace(self.args, num_quantization_groups=8)
        decoder = TransformerDecoder(args, device=self.device)
        x = torch.randint(0, self.args.vocab_size, (2, self.args.max_seq_len, 8))
        context = torch.randn(2, self.args.max_seq_len, self.args.dim).to(self.device)
        output = decoder(x, context=context)
        self.assertEqual(output.shape, (*x.shape, self.args.vocab_size))

    def test_generator(self):
        logger.info('test_generator')
        gen = Generator(self.args, start_id=3, end_id=1, sep_id=2, pad_id=-1, device=self.device)
        self.assertIsNotNone(gen)
        prompts = [[3], [3, 8, 6, 4], [3, 2], [3, 4, 5]]
        contexts = [None, torch.randn(len(prompts), self.args.max_seq_len, self.args.dim).to(self.device)]
        cache = InferAccelerationCache(self.args)
        for ctx in contexts:
            out_tokens = gen.generate(prompts, context=ctx, cache=cache, max_gen_len=10, temperature=0.6, top_p=0.9, pos_bias=0)
            self.assertEqual(len(out_tokens), len(prompts))

    def test_generator_mq(self):
        logger.info('test_generator_mq')
        args = replace(self.args, num_quantization_groups=8)
        gen = Generator(args, start_id=3, end_id=1, sep_id=2, pad_id=-1, device=self.device)
        self.assertIsNotNone(gen)
        prompts = [torch.randint(0, args.vocab_size, (20, 8)) for _ in range(4)]
        contexts = [None, torch.randn(len(prompts), self.args.max_seq_len, self.args.dim).to(self.device)]
        cache = InferAccelerationCache(self.args)
        for ctx in contexts:
            out_tokens = gen.generate(prompts, context=ctx, cache=cache, max_gen_len=10, temperature=0.6, top_p=0.9, pos_bias=0)
            self.assertEqual(len(out_tokens), len(prompts))

    def test_generator_forward_pass(self):
        logger.info("test_generator_forward_pass")
        prompt_size = 8
        seq_len = 16
        vocab_size = self.args.vocab_size
        start_token_id = 81
        pad_token_id = 0
        end_token_id = 82
        sep_token_id = 83
        sentence = torch.randint(0, 80, (seq_len,))
        sentence[:prompt_size] = torch.Tensor([i + 1 for i in range(prompt_size)]).to(torch.long)
        contexts = torch.randn(1, seq_len, self.args.dim).to(self.device)
        gen = Generator(self.args, start_id=start_token_id, end_id=end_token_id, pad_id=pad_token_id, sep_id=sep_token_id, device=self.device)
        self.assertIsNotNone(gen)
        inputs = gen.prepare_for_autoregressive_training(sentences=[sentence], contexts=contexts, prompt_len=prompt_size, add_start_id_at_beginning=True, insert_sep_id_after_prompt=True, device=self.device)
        logger.info("all_tokens:\n{}\n".format(inputs["tokens"]))
        logger.info("labels: \n{}\n".format(inputs["labels"]))
        contexts_list = [None, inputs["context"]]
        for ctx in contexts_list:
            output = gen(inputs["tokens"], labels=inputs["labels"], context=ctx)
            logits = output["logits"]
            loss = output["loss"]
            self.assertEqual(logits.shape, (*inputs["tokens"].shape, vocab_size))
            self.assertEqual(loss.shape, ())
            logger.info("logits: {}\n".format(logits.shape))
            logger.info("loss: {}\n".format(loss))

    def test_generator_forward_pass_mq(self):
        logger.info("test_generator_forward_pass_mq")
        prompt_size = 4
        seq_len = 20
        num_q = 8
        vocab_size = self.args.vocab_size
        start_token_id = 81
        pad_token_id = 0
        end_token_id = 82
        sep_token_id = 83
        sentence = torch.randint(0, vocab_size, (seq_len, num_q))
        sentence[:prompt_size, :] = (torch.Tensor([i + 1 for i in range(prompt_size)]).unsqueeze(-1).to(torch.long))
        contexts = torch.randn(1, seq_len, self.args.dim).to(self.device)
        args = replace(self.args, num_quantization_groups=num_q)
        gen = Generator(args, start_id=start_token_id, end_id=end_token_id, sep_id=sep_token_id, pad_id=pad_token_id, device=self.device)
        inputs = gen.prepare_for_autoregressive_training(sentences=[sentence], contexts=contexts, prompt_len=prompt_size, add_start_id_at_beginning=True, insert_sep_id_after_prompt=True, device=self.device)
        logger.info("[mq] all_tokens:\n{}\n".format(inputs["tokens"].shape))
        logger.info("[mq] labels: \n{}\n".format(inputs["labels"].shape))
        contexts_list = [None, inputs["context"]]
        for ctx in contexts_list:
            output = gen(inputs["tokens"], labels=inputs["labels"], context=ctx)
            logits = output["logits"]
            loss = output["loss"]
            self.assertEqual(logits.shape, (*inputs["tokens"].shape, vocab_size))
            self.assertEqual(loss.shape, ())
            logger.info("logits: {}\n".format(logits.shape))
            logger.info("loss: {}\n".format(loss))