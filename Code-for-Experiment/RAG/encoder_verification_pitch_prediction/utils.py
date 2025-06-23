import pretty_midi
import os
import math
import torch
import numpy as np

class Melody(list):
    def onset_representation(self, freq=75):
        end_time = self[-1].end
        representation = [0] * math.ceil(end_time * freq)
        for note in self:
            onset = note.start
            onset = round(onset * freq)
            pitch = note.pitch
            pitch += 1
            representation[onset] = pitch
        return representation

    def frame_repr(self, freq=75):
        end_time = self[-1].end
        representation = np.zeros(math.ceil(end_time * freq), dtype=np.int64)
        for note in self:
            onset = note.start
            onset = round(onset * freq)
            offset = note.end
            offset = round(offset * freq)
            pitch = note.pitch
            pitch += 1
            representation[onset:offset-1] = pitch
            representation[offset-1] = 0
        return representation

    def pitch_onset_repr(self, freq=75):
        end_time = self[-1].end
        representation = np.zeros(math.ceil(end_time * freq), dtype=np.int64)
        is_onset = np.zeros(math.ceil(end_time * freq), dtype=np.int64)
        for note in self:
            onset = note.start
            offset = note.end
            if offset - onset > 5:
                offset = onset + 5
            onset = round(onset * freq)
            offset = round(offset * freq)
            pitch = note.pitch
            pitch += 1
            representation[onset:offset] = pitch
            is_onset[onset] = 1
        return representation, is_onset

    def to_midi(self, path: str):
        midi_file = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        pre_pitch = 0
        duration = 0
        start = 0
        pitches, _ = self.pitch_onset_repr()
        for pitch in pitches:
            if pitch > 0:
                pitch -= 1
            if pitch == pre_pitch:
                duration += 1/75
            else:
                if pre_pitch != 0:
                    note = pretty_midi.Note(velocity=100,
                                            pitch=pre_pitch,
                                            start=start,
                                            end=start+duration)
                    inst.notes.append(note)
                start += duration
                duration = 1/75
            pre_pitch = pitch
        midi_file.instruments.append(inst)
        midi_file.write(path)
        print(f"write midi file to {path}")

class LeadSheet(pretty_midi.PrettyMIDI):
    def __init__(self, midi_file):
        super().__init__(midi_file)
        try:
            rhythm, chord, melody = self.instruments
            self.rhythm = rhythm.notes
            self.chord = chord.notes
            self.melody = Melody(melody.notes)
        except:
            self.rhythm, self.chord, self.melody = (None, None, None)

def load_tokens(dirpath: str):
    dirlist = os.listdir(dirpath)
    all_tokens = []
    for filename in dirlist:
        filepath = os.path.join(dirpath, filename)
        tokens = np.load(filepath)
        all_tokens.append(tokens)
    return all_tokens

def load_midi(dirpath: str):
    dirlist = os.listdir(dirpath)
    all_midi_files = []
    for midi_file in dirlist:
        midi_file_path = os.path.join(dirpath, midi_file)
        midi = LeadSheet(midi_file_path)
        all_midi_files.append(midi)
    return all_midi_files

def predict(out, thresh=None):
    pred = None
    if thresh is None:
        pred = torch.argmax(out, dim=-1)
    else:
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        assert len(out.shape) == 3
        prob_nonnull = 1 - torch.softmax(out, dim=-1)[:, :, 0]
        pred_nonnull = 1 + torch.argmax(out[:, :, 1:], dim=-1)
        assert prob_nonnull.shape == pred_nonnull.shape
        pred = torch.zeros_like(pred_nonnull)
        pred[prob_nonnull>=thresh] = pred_nonnull[prob_nonnull>=thresh]
    return pred

if __name__ == '__main__':
    midi_file = './data/midi/654a3df993a643d7ecad4451b901b9afd69c715b.mp3_5b.mid'
    midi = LeadSheet(midi_file)
    for note in midi.melody:
        print(note.start, note.end, note.pitch)
    framer, is_onset = midi.melody.pitch_onset_repr()
    print(list(framer))
    midi.melody.to_midi("output_valid.mid")