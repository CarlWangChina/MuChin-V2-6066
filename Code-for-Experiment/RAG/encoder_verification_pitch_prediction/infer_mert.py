import os
import torch
import pretty_midi
import numpy as np
import logging
from argparse import ArgumentParser
from tqdm import tqdm
from Code_for_Experiment.RAG.encoder_verification_pitch_prediction.utils import predict

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("path", type=str, help="input path")
parser.add_argument("--output", "-o", type=str, help="output path")
parser.add_argument("--sec_per_chunk", "-s", type=float, help="seconds per chunk")
parser.add_argument("--note_threshold", "-n", type=float, help="the shortest duration of one note /s")
parser.add_argument(
    "--window_size",
    "-w",
    type=float,
    help="Relative window size of each chunk, which is the overlap between two neighboring windows /s. Real window size is sec_per_chunk + 2 * window_size.",
)
parser.set_defaults(
    output="output.mid", sec_per_chunk=6, note_threshold=0.15, window_size=1
)
args = parser.parse_args()

data_path = args.path
output_path = args.output
sec_per_chunk = args.sec_per_chunk
nthresh = args.note_threshold
window_size = args.window_size
freq = 75
frames_per_chunk = round(sec_per_chunk * freq)
fwindow = round(freq * window_size)
path = "./model_save/mert_model_240125.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(path, map_location=device)
model.eval()

with open(data_path, "rb") as f:
    mert = f.read()
    mert = np.frombuffer(mert, dtype=np.float32).reshape(-1, 1024)

data = torch.tensor(mert)
data = data.to(device)
full_data = data
full_pitches = []
logging.debug(data.shape)

for i in tqdm(range(0, full_data.size(0), frames_per_chunk)):
    start = max(0, i - fwindow)
    end = min(full_data.size(0), i + frames_per_chunk + fwindow)
    logging.debug(f"start = {start}, end = {end}")
    data = full_data[start:end]
    data = data.view(1, data.size(0), data.size(1))
    logging.debug(data)
    logging.debug(data.shape)
    out1, out2 = model(data)
    pitches = predict(out1).squeeze()
    pitches[pitches > 0] += 20
    if end > i + frames_per_chunk:
        pitches = pitches[: -(end - i - frames_per_chunk)]
    if start < i:
        pitches = pitches[i - start :]
    full_pitches.extend(pitches)

midi_file = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(0)
pre_pitch = 0
duration = 0
start = 0
for pitch in full_pitches:
    pitch = pitch.item()
    if pitch == pre_pitch:
        duration += 1 / 75
    else:
        if pre_pitch != 0 and duration >= nthresh:
            note = pretty_midi.Note(
                velocity=100, pitch=pre_pitch, start=start, end=start + duration
            )
            inst.notes.append(note)
        start += duration
        duration = 1 / 75
    pre_pitch = pitch

midi_file.instruments.append(inst)
midi_file.write(output_path)
logging.info(f"Write midi file to {output_path}")