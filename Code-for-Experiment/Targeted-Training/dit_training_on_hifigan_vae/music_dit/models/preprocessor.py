import torch
import torch.nn as nn
import math
import random
from typing import Optional, List, Dict, Tuple
from music_dit.utils import get_logger, get_hparams, get_audio_utils, Lyrics
from music_dit.modules.clap import ClapEncoder
from music_dit.modules.tiktoken import TikTokenWrapper
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.encodec.encodec import EncodecVAE

logger = get_logger(__name__)

class MusicDiTPreprocessor(nn.Module):
    def __init__(self):
        super(MusicDiTPreprocessor, self).__init__()
        hparams = get_hparams()
        logger.info("Initializing the data pre-processor...")
        self.input_dim = hparams.model.dit.input_dim
        self.hidden_dim = hparams.model.dit.hidden_dim
        self.vae_frame_size = hparams.model.vae.frame_size
        self.clap_embedding_dim = hparams.model.clap.embedding_dim
        assert self.input_dim == hparams.model.vae.embedding_dim, \
            (f"Bad configuration:  The VAE embedding dimension ({hparams.model.vae.embedding_dim}) "
             f"should be the same as the input dimension of the DIT model ({self.input_dim}).")
        self.audio_utils = get_audio_utils()
        self.vae = EncodecVAE()
        assert self.vae.embedding_dim == self.input_dim, \
            (f"Bad configuration:  The VAE embedding dimension ({self.vae.embedding_dim}) should be the same as "
             f"the input dimension of the DIT model ({self.input_dim}).")
        assert self.vae.frame_rate == self.vae_frame_size, \
            (f"Bad configuration:  The VAE frame rate ({self.vae.frame_rate}) should be the same as "
             f"the VAE frame size ({self.vae_frame_size}).")
        self.vae_time_unit = self.vae.segment_stride / self.vae.sampling_rate
        self.clap_encoder = ClapEncoder()
        self.clap_segment_len = self.clap_encoder.max_clip_samples // self.clap_encoder.sampling_rate
        assert self.clap_embedding_dim == self.clap_encoder.joint_embedding_dim, \
            (f"Bad configuration:  The CLAP embedding dimension ({self.clap_embedding_dim}) should be the same as "
             f"the joint embedding dimension of the CLAP encoder ({self.clap_encoder.joint_embedding_dim}).")
        self.lyrics_tokenizer = TikTokenWrapper()
        assert self.lyrics_tokenizer.n_vocab == hparams.model.lyrics.vocab_size, \
            (f"Bad configuration:  The vocabulary size of the lyric tokenizer ({self.lyrics_tokenizer.n_vocab}) "
             f"should be the same as the vocabulary size of the lyrics ({hparams.model.lyrics.vocab_size}).")
        assert self.lyrics_tokenizer.PAD_Token == hparams.model.lyrics.padding_token, \
            (f"Bad configuration:  The padding token ID of the lyric tokenizer ({self.lyrics_tokenizer.PAD_Token}) "
             f"should be the same as the padding token ID of the lyrics ({hparams.model.lyrics.padding_token}).")
        self.vae_sampling_rate = self.vae.sampling_rate
        self.vae_num_channels = self.vae.num_channels
        self.demux_sampling_rate = self.audio_utils.demucs_sampling_rate
        self.demux_num_channels = self.audio_utils.demucs_num_channels
        self.clap_sampling_rate = self.clap_encoder.sampling_rate
        self.clap_num_channels = self.clap_encoder.num_channels
        self.lyric_vocab_size = self.lyrics_tokenizer.n_vocab

    @torch.no_grad()
    def split_audio(self,
                    audio: torch.Tensor,
                    *,
                    sampling_rate: int,
                    max_batches: int,
                    lyrics: Optional[List[Lyrics]] = None,
                    min_length: float = 20.0,
                    max_length: float = 30.0,
                    prompt_length: float = 20.0,
                    vocal_threshold: float = 0.1) -> List[Dict[str, Optional[torch.Tensor]]]:
        assert min_length <= max_length, "The minimum length should be less than or equal to the maximum length."
        min_length = int(min_length / self.vae_time_unit + 0.5)
        max_length = int(max_length / self.vae_time_unit + 0.5)
        prompt_length = int(prompt_length / self.vae_time_unit + 0.5)
        audio_resampled_vae = self.audio_utils.resample(audio,
                                                        sampling_rate,
                                                        self.vae_sampling_rate,
                                                        self.vae_num_channels) \
            if sampling_rate != self.vae_sampling_rate or audio.size(0) != self.vae_num_channels \
            else audio
        audio_resampled_clap = self.audio_utils.resample(audio,
                                                         sampling_rate,
                                                         self.clap_sampling_rate,
                                                         self.clap_num_channels) \
            if sampling_rate != self.clap_sampling_rate or audio.size(0) != self.clap_num_channels \
            else audio
        vae_embeddings, vae_scales = self.vae.encode(audio_resampled_vae,
                                                     sampling_rate=self.vae_sampling_rate)
        vae_embeddings = vae_embeddings.detach()
        vae_scales = vae_scales.detach()
        begin = prompt_length + int(math.ceil(self.clap_segment_len / 2 / self.vae_time_unit))
        end = vae_embeddings.size(0)
        end = end - int(math.ceil(self.clap_segment_len / 2 / self.vae_time_unit))
        if end - begin < max_length:
            logger.error("The audio is too short.  Should be at least %d seconds.  Skipped",
                         prompt_length + max_length + self.clap_segment_len + 1)
            return []
        if lyrics is not None:
            audio_vocals = self.audio_utils.demucs_separate(audio,
                                                            sampling_rate,
                                                            ["vocals"])[0][0]
            vocal_mask = self._get_vocal_mask(audio_vocals,
                                              self.audio_utils.demucs_sampling_rate,
                                              vae_embeddings.size(0),
                                              vocal_threshold)
            lyrics = self._filter_lyrics_by_vocal_mask(lyrics, vocal_mask, begin, end)
            intervals = self._get_intervals_between_lyrics(lyrics, begin, end)
            start_points, end_points, lyrics = self._sample_segments_with_lyrics(intervals,
                                                                                 max_batches,
                                                                                 min_length,
                                                                                 max_length,
                                                                                 lyrics)
        else:
            start_points = torch.randint(begin, end - max_length, size=(max_batches,))
            end_points = start_points + max_length
            lyrics = None
        results = []
        for i in range(max_batches):
            results.append(self._get_segments(audio=audio,
                                              sampling_rate=sampling_rate,
                                              vae_embeddings=vae_embeddings,
                                              vae_scales=vae_scales,
                                              audio_resampled_clap=audio_resampled_clap,
                                              prompt_length=prompt_length,
                                              start_point=start_points[i].item(),
                                              end_point=end_points[i].item(),
                                              lyrics=lyrics[i] if lyrics is not None else None))
        return results

    @torch.no_grad()
    def _get_segments(self,
                      audio: torch.Tensor,
                      sampling_rate: int,
                      vae_embeddings: torch.Tensor,
                      vae_scales: torch.Tensor,
                      audio_resampled_clap: torch.Tensor,
                      prompt_length: int,
                      start_point: int,
                      end_point: int,
                      lyrics: Optional[str] = None) -> Dict[str, Optional[torch.Tensor]]:
        audio_segment = audio[:, start_point * sampling_rate:end_point * sampling_rate]
        vae_segment = vae_embeddings[start_point:end_point]
        vae_scales_segment = vae_scales[start_point:end_point]
        vae_len = vae_segment.size(0)
        prompt = vae_embeddings[start_point - prompt_length:start_point]
        clap_embedding = torch.zeros(vae_len, self.clap_embedding_dim, device=audio.device)
        for j in range(vae_len):
            half_clip_samples = self.clap_encoder.max_clip_samples // 2
            clap_begin = (int((start_point + j) * self.vae_time_unit * self.clap_encoder.sampling_rate + 0.5)
                          - half_clip_samples)
            clap_end = clap_begin + self.clap_encoder.max_clip_samples
            audio_resampled_clap_segment = audio_resampled_clap[:, clap_begin:clap_end]
            clap_embedding[j] = self.clap_encoder.get_audio_embedding(audio_resampled_clap_segment).detach()
        lyrics_tokens = None
        if lyrics is not None:
            lyrics_tokens = self.lyrics_tokenizer.encode(lyrics)
            lyrics_tokens = torch.tensor(lyrics_tokens).long().to(audio.device)
        return {
            "audio": audio_segment,
            "sampling_rate": sampling_rate,
            "vae": vae_segment,
            "vae_scales": vae_scales_segment,
            "prompt": prompt,
            "clap": clap_embedding,
            "lyrics": lyrics_tokens
        }

    def _get_vocal_mask(self,
                        vocal: torch.Tensor,
                        sampling_rate: int,
                        length: int,
                        vocal_threshold: float) -> torch.Tensor:
        vocal_mask = torch.zeros(length, device=vocal.device)
        vocal = vocal.mean(dim=0, keepdim=False).abs()
        threshold = vocal.mean().item() * vocal_threshold
        for i in range(length):
            start = int(i * self.vae_time_unit * sampling_rate)
            end = int((i + 1) * self.vae_time_unit * sampling_rate)
            vocal_mask[i] = (vocal[start:end].mean() > threshold).long()
        return vocal_mask

    def _filter_lyrics_by_vocal_mask(self,
                                     lyrics: List[Lyrics],
                                     vocal_mask: torch.Tensor,
                                     begin: int,
                                     end: int,
                                     min_percentage: float = 0.95) -> List[Lyrics]:
        filtered_lyrics = []
        for lyric in lyrics:
            start_time = int(lyric.start_time / self.vae_time_unit + 0.5)
            end_time = int(lyric.end_time / self.vae_time_unit + 0.5)
            if start_time < begin or end_time > end:
                continue
            vocal_duration = vocal_mask[start_time:end_time].sum().item()
            if vocal_duration >= (end_time - start_time) * min_percentage:
                filtered_lyrics.append(lyric)
        return filtered_lyrics

    def _get_intervals_between_lyrics(self,
                                      lyrics: List[Lyrics],
                                      begin: int,
                                      end: int) -> List[Tuple[int, int]]:
        intervals = []
        current_pos = begin
        for lyric in lyrics:
            start_time = int(lyric.start_time / self.vae_time_unit + 0.5)
            end_time = int(lyric.end_time / self.vae_time_unit + 0.5)
            assert start_time >= current_pos, "The start time of the lyric should be greater than or equal to the " \
                                              "end of the previous lyric."
            intervals.append((current_pos, start_time))
            current_pos = end_time
        assert current_pos <= end
        intervals.append((current_pos, end))
        assert len(intervals) == len(lyrics) + 1
        return intervals

    def _sample_segments_with_lyrics(self,
                                     intervals: List[Tuple[int, int]],
                                     max_batches: int,
                                     min_length: int,
                                     max_length: int,
                                     lyrics: List[Lyrics]) \
            -> Tuple[torch.Tensor, torch.Tensor, List[Optional[str]]]:
        assert len(intervals) == len(lyrics) + 1
        start_points = torch.zeros(max_batches, dtype=torch.long)
        end_points = torch.zeros(max_batches, dtype=torch.long)
        out_lyrics = []
        for i in range(max_batches):
            while True:
                start_interval_idx = random.randint(0, len(intervals) - 1)
                start_interval = intervals[start_interval_idx]
                if start_interval[1] == start_interval[0]:
                    start_point = start_interval[0]
                else:
                    start_point = random.randint(start_interval[0], start_interval[1] - 1)
                end_interval = None
                end_interval_idx = None
                for j in range(start_interval_idx, len(intervals)):
                    if intervals[j][1] - start_point >= min_length:
                        end_interval = intervals[j]
                        end_interval_idx = j
                    if intervals[j][1] - start_point >= max_length:
                        break
                if end_interval is None:
                    continue
                if end_interval[0] - start_point > max_length:
                    continue
                if end_interval[1] == end_interval[0]:
                    end_point = end_interval[0]
                else:
                    end_point = min(random.randint(start_point + min_length, end_interval[1]),
                                    start_point + max_length)
                start_points[i] = start_point
                end_points[i] = end_point
                if start_interval_idx == end_interval_idx:
                    out_lyrics.append(None)
                else:
                    out_lyric = lyrics[start_interval_idx]
                    for j in range(start_interval_idx + 1, end_interval_idx):
                        out_lyric = out_lyric.cat(lyrics[j])
                    out_lyrics.append(out_lyric.text)
                break
        diff = end_points - start_points
        assert min_length <= diff.min() <= diff.max() <= max_length
        return start_points, end_points, out_lyrics