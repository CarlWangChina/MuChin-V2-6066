import os
import torch
import yaml
import pretty_midi
import numpy as np
import logging
from argparse import ArgumentParser
from dataset import get_data
from utils import predict
from Code_for_Experiment.RAG.encoder_verification_pitch_prediction.metrics import accuracy

logging.basicConfig(level=logging.DEBUG)

parser = ArgumentParser()
parser.add_argument(
    "path",
    type=str,
    help="input path"
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    help="output path"
)
parser.set_defaults(
    path="input.npy",
    output="output.mid"
)
args = parser.parse_args()
data_path = args.path
output_path = args.output
path = "./model_save/mert_model_240125.pth"
device = "cuda:0"

model = torch.load(path, map_location=device)
model.eval()

with open("config/Code-for-Experiment/RAG/encoder_verification_pitch_prediction/config/Code-for-Experiment/RAG/encoder_verification_pitch_prediction/config/mert.yaml", "r") as f:
    config = yaml.safe_load(f)
config = config["data"]

if config["mert_path"] is None:
    use_token = True
else:
    use_token = False

if os.path.exists(data_path):
    if use_token:
        data = np.load(data_path)
    else:
        with open(data_path, 'rb') as f:
            mert = f.read()
            mert = np.frombuffer(mert, dtype=np.float32).reshape(-1, 1024)
        data = torch.tensor(mert[2175:3075])
else:
    trainset, _, testset = get_data(config)
    data = testset[5]
    data, label, _ = data
    data = torch.tensor(data)
    label = torch.tensor(label).to(device)
    output_path = output_path[:-4] + "_label.mid"
    pitches = label
    pitches[pitches > 0] += 20
    midi_file = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    pre_pitch = 0
    duration = 0
    start = 0
    for pitch in pitches:
        pitch = pitch.item()
        if pitch == pre_pitch:
            duration += 1 / 75
        else:
            if pre_pitch != 0 and duration >= 0.1:
                note = pretty_midi.Note(velocity=100, pitch=pre_pitch, start=start, end=start + duration)
                inst.notes.append(note)
            start += duration
            duration = 1 / 75
        pre_pitch = pitch
    midi_file.instruments.append(inst)
    midi_file.write(output_path)
    print(f"Write midi file to {output_path}")

if use_token:
    data = data.view(1, -1)
else:
    data = data.view(1, data.size(0), data.size(1))

data = data.to(device)
logging.debug(data.shape)

out1, out2 = model(data)
pitches = predict(out1).squeeze()
pitches[pitches > 0] += 20
pitches = pitches[150:-150]
midi_file = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(0)
pre_pitch = 0
duration = 0
start = 0
for pitch in pitches:
    pitch = pitch.item()
    if pitch == pre_pitch:
        duration += 1 / 75
    else:
        if pre_pitch != 0 and duration >= 0.1:
            note = pretty_midi.Note(velocity=100, pitch=pre_pitch, start=start, end=start + duration)
            inst.notes.append(note)
        start += duration
        duration = 1 / 75
    pre_pitch = pitch
midi_file.instruments.append(inst)
midi_file.write(output_path)
print(f"Write midi file to {output_path}")