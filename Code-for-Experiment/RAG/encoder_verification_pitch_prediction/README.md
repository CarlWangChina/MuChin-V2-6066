# Scripts to verify the effectiveness of audio encoders (e.g., MERT) by using their output vectors or tokens to predict corresponding melody pitch for the Amateur-Professional Semantic Divide research.

## Melody-Refinement
Transcribe MERT / Tokens to pitches.

## Installation
```bash
conda create -n melody python=3.9
conda activate melody
pip install -r requirements.txt
```

## Training
### Train MERT to Pitches
```bash
python train.py -c config/mert.yaml -s model_save/your_model_name.pth
```

### Train Tokens to Pitches Using DP
**Note:** The initial version of Tokens performed poorly, so this part of the data has been removed from the directory and cannot be trained for the Token Model temporarily. If you need to test this part of the code, the tokens data has a backup in ZJ-A800-2 /data/xary/melody_refinement/data/tokens.

```bash
python train.py -c config/config_pitch_dp.yaml -s model_save/your_model_name.pth
```

## Inference
Only the MERT Model can extract melody successfully now.

**Note:** GitHub doesn't support large file uploads (> 100M). The MERT model path is in ZJ-A800-2 /data/xary/melody_refinement/model_save/mert_model_240125.pth. If you need to use it, you can copy it out first.

```bash
python infer_mert.py -o output/your_midi_name.mid (-s relative_window_size) (-n note_threshold) your/input.mert
```

This module is part of the MuChin project's targeted training framework for understanding the semantic differences between amateur and professional music descriptions.


