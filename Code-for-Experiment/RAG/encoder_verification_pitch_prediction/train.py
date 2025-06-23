import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import logging
from tqdm import tqdm
from dataset import get_data, Tokens2MelodyDataLoader
from model import Tokens2PitchOnsetModel, MelodyTranscriptionModel
from metrics import accuracy, F1_score
from Code_for_Experiment.RAG.encoder_verification_pitch_prediction.utils import predict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, config):
        model_config = config['model']
        data_config = config['data']
        train_config = config['train']
        metrics_config = config["metrics"]
        modelname = model_config["name"]
        if modelname == "Tokens2PitchOnsetModel":
            self.model = Tokens2PitchOnsetModel(model_config)
        elif modelname == "MelodyTranscriptionModel":
            self.model = MelodyTranscriptionModel(model_config)
        elif os.path.exists(modelname):
            self.model = torch.load(modelname)
        else:
            raise ValueError()
        self.train_set, self.valid_set, self.test_set = get_data(data_config)
        logger.debug("Dataset done.")
        self.train_config = data_config["train"]
        self.valid_config = data_config["valid"]
        self.test_config = data_config["test"]
        config = train_config
        self.loss_fn = config["loss_fn"]
        self.optimizer = config["optimizer"]
        self.lr = config["learning_rate"]
        self.max_epoch = config["max_epoch"]
        self.early_stop = config["early_stop"]
        self.pitch_start = config["pitch_start"]
        self.device = config["device"]
        self.device_ids = config["device_ids"]
        self.is_save = config["is_save"]
        self.save_best = config["save_best"]
        self.save_path = config["save_path"]
        self.use_loss_weights = config["use_loss_weights"]
        self.thresh = config["predict_thresh"]
        self.smooth = config["smooth"]
        self.require_loss = config["require_loss"]
        self.scheduler_config = config["scheduler"]
        self.vocab_size = config["vocab_size"]

    def train(self):
        raise NotImplementedError()

class PitchOnsetTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        logger.debug("Training start.")
        torch.manual_seed(42)
        model = self.model
        logger.info(model)
        model = model.to(self.device)
        model = nn.DataParallel(model, device_ids=self.device_ids)
        logger.debug("Model DP done.")

        self.valid_loader = Tokens2MelodyDataLoader(self.valid_set, self.valid_config)
        self.test_loader = Tokens2MelodyDataLoader(self.test_set, self.test_config)
        self.train_loader = Tokens2MelodyDataLoader(self.train_set, self.train_config)
        logger.debug("Dataloader done.")

        pitches_weight = torch.ones(self.vocab_size, dtype=torch.float32).to(self.device, non_blocking=True)
        onset_weight = torch.ones(2, dtype=torch.float32).to(self.device, non_blocking=True)
        if self.use_loss_weights:
            for data in tqdm(self.train_loader):
                _, pitches, is_onset = data
                pitches = pitches.to(self.device, non_blocking=True).view(-1)
                is_onset = is_onset.to(self.device, non_blocking=True).view(-1)
                pitches_weight += torch.bincount(pitches, minlength=self.vocab_size)
                onset_weight += torch.bincount(is_onset, minlength=2)
            pitches_weight = 1 / pitches_weight
            onset_weight = 1 / onset_weight
            onset_weight = onset_weight[1] / onset_weight[0]

        logger.info(pitches_weight)
        logger.info(onset_weight)
        loss_fn_1 = nn.CrossEntropyLoss(weight=pitches_weight.to(self.device, non_blocking=True))
        loss_fn_2 = nn.BCELoss(weight=onset_weight.to(self.device, non_blocking=True))
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), self.lr)
        else:
            raise ValueError("Please set an optimizer from [\"Adam\", \"SGD\"]")

        scheduler = None
        if self.scheduler_config["warmup"]:
            d_model = self.scheduler_config["d_model"]
            warmup_steps = self.scheduler_config["warmup_steps"]
            def lr_lambda(step):
                step += 1
                return d_model**(-0.5) * min(step**(-0.5), step * (warmup_steps**(-1.5)))
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            step_size = self.scheduler_config["step_size"]
            gamma = self.scheduler_config["gamma"]
            scheduler = StepLR(optimizer, step_size, gamma)

        min_loss = 2147483647
        max_acc = 0
        early_stop_cnt = 0
        for epoch in range(self.max_epoch):
            logger.info(f"----- epoch: {epoch+1} -----")
            torch.manual_seed(epoch)
            model.train()
            total_loss = 0
            total_pitches_acc = 0
            total_onset_acc = 0
            for step, data in enumerate(self.train_loader):
                tokens, pitches, is_onset = data
                tokens = tokens.to(self.device, non_blocking=True)
                pitches = pitches.to(self.device, non_blocking=True)
                is_onset = is_onset.to(self.device, non_blocking=True).float()
                is_onset = is_onset.unsqueeze(2)
                optimizer.zero_grad()
                out = model(tokens)
                out1, out2 = out
                loss1 = loss_fn_1(out1.view(-1, out1.size(-1)), pitches.view(-1))
                loss2 = loss_fn_2(out2.view(-1), is_onset.view(-1))
                loss = [loss1, loss2, loss1 + loss2]
                loss = loss[self.require_loss]
                pitches_pred = predict(out1)
                pitches_acc = accuracy(pitches_pred, pitches)
                onset_pred = torch.round(out2).long()
                onset_acc = F1_score(onset_pred, is_onset)
                total_pitches_acc += pitches_acc
                total_onset_acc += onset_acc
                total_loss += loss
                loss.backward()
                optimizer.step()
                if self.scheduler_config["warmup"]:
                    scheduler.step()

            if not self.scheduler_config["warmup"]:
                scheduler.step()

            total_loss /= len(self.train_loader)
            logger.info(f"train loss = {total_loss}")
            total_pitches_acc /= len(self.train_loader)
            logger.info(f"train pitches acc = {total_pitches_acc}")

            model.eval()
            with torch.no_grad():
                total_loss1 = 0
                total_loss2 = 0
                total_pitches_acc = 0
                total_onset_acc = 0
                for data in self.valid_loader:
                    tokens, pitches, is_onset = data
                    tokens = tokens.to(self.device, non_blocking=True)
                    pitches = pitches.to(self.device, non_blocking=True)
                    is_onset = is_onset.to(self.device, non_blocking=True).float()
                    is_onset = is_onset.unsqueeze(2)
                    out = model(tokens)
                    out1, out2 = out
                    pitches_pred = predict(out1)
                    pitches_acc = accuracy(pitches_pred, pitches)
                    onset_pred = torch.round(out2).long()
                    onset_acc = F1_score(onset_pred, is_onset)
                    total_pitches_acc += pitches_acc
                    total_onset_acc += onset_acc
                total_pitches_acc /= len(self.valid_loader)
                total_onset_acc /= len(self.valid_loader)
                logger.info(f"validation pitches acc = {total_pitches_acc}")
                logger.info(f"max pitch acc = {max_acc}")
                if total_pitches_acc <= max_acc:
                    early_stop_cnt += 1
                    if early_stop_cnt >= self.early_stop:
                        print(f"Validation acc hasn't increased for {self.early_stop} epochs.")
                        print("Training process stopped.")
                        break
                else:
                    max_acc = total_pitches_acc
                    early_stop_cnt = 0
                    if self.is_save and self.save_best:
                        torch.save(model, self.save_path)
                if self.is_save and not self.save_best:
                    torch.save(model, self.save_path)
                torch.cuda.empty_cache()

if __name__ == '__main__':
    import yaml
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path of the yaml format config file"
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="save path of model"
    )
    parser.set_defaults(
        config="./config/config.yaml"
    )
    args = parser.parse_args()
    save_path = args.save_path
    with open(args.config, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["train"]["save_path"] = save_path
    logger.info(config)
    trainer = PitchOnsetTrainer(config)
    trainer.train()