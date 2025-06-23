# Model Testing Code for Five Different Models

## Environment Configuration

```python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preprocessing

```python
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
```

This module provides comprehensive testing capabilities for various music understanding models as part of the MuChin project's intent fidelity evaluation framework. 