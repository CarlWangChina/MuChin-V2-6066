#Contains scripts for data cleaning, preprocessing, and model training related to the music understanding models.

# 5 Model Testing Code

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preprocessing

```bash
python inference/encodec-inference.py
python inference/jukebox-inference.py
python inference/music2vec-inference.py
python inference/extract_mert_features/extract_mert_features.py 95
python inference/extract_mert_features/extract_mert_features.py 330

python tools/buildTagDict.py
python tools/data_preprocess.py
python tools/encodec_decode.py
python tools/get_data_list.py
python tools/mean_jukebox.py
```

## Model Training

```bash
sh train.sh
```

## Model Testing

```bash
python predict/predict.py

