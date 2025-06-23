from datasets import load_dataset
from transformers import ClapModel, ClapProcessor, TrainingArguments, Trainer
from transformers.trainer_utils import SchedulerType
import torch
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import warnings
warnings.filterwarnings("ignore")

muchin_dataset = load_dataset('muchin_dataset_script.py', trust_remote_code=True)
processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

def preprocess(muchin_data):
    audio = [a['array'] for a in muchin_data['audio']]
    text = muchin_data['text']
    inputs = processor(text=text, audios=audio, return_tensors='pt', padding='max_length', sampling_rate=48000)
    return inputs

processed_dataset = muchin_dataset.map(preprocess, batch_size=12, batched=True, num_proc=os.cpu_count())

model = ClapModel.from_pretrained("laion/larger_clap_music")
model.to(device)
model.train()

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=20,
    load_best_model_at_end=False,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=100,
    learning_rate=2e-5,
    warmup_ratio=0.2,
    seed=3407,
    lr_scheduler_type=SchedulerType.COSINE,
    dataloader_drop_last=True,
    dataloader_num_workers=16)

def collate_fn(features):
    batch = {}
    first = features[0]
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    batch["return_loss"] = True
    return batch

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['test'],
    data_collator=collate_fn)

trainer.train()