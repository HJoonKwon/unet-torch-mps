import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from unet_torch_mps.utils.utils import load_checkpoint, set_logger
from unet_torch_mps.metrics.iou import calculate_miou


class Trainer:
    def __init__(self, model, optimizer, train_data_loader, valid_data_loader, config):
        self.num_epochs = config["num_epochs"]
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = config["device"] if "device" in config else "cpu"
        self.ckpt_path = config["ckpt"] if "ckpt" in config else None
        self.ckpt_saving_dir = (
            config["ckpt_saving_dir"] if "ckpt_saving_dir" in config else None
        )
        self.ckpt_interval = config["ckpt_interval"] if "ckpt_interval" in config else 1
        self.logger = set_logger()
        if self.ckpt_path is not None:
            self.start_epoch = load_checkpoint(
                self.model, self.optimizer, self.ckpt_path, self.logger
            )
        else:
            self.start_epoch = 0

        assert self.device in ["cpu", "cuda", "mps"] or self.device.startswith("cuda:")
        if not os.path.exists(self.ckpt_saving_dir):
            os.makedirs(self.ckpt_saving_dir)

        self.best_score = {
            "loss": {"epoch": 0, "value": float("inf")},
            "miou": {"epoch": 0, "value": 0},
        }

        self.logger.info(f"Trainer:: Using model: {self.model}")
        self.logger.info(f"Trainer:: Using device: {self.device}")
        self.logger.info(f"Trainer:: Starting epoch: {self.start_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            self.logger.info(f"Trainer:: Training epoch: {epoch}")
            train_loss = self.train_epoch()
            self.logger.info(
                f"Trainer:: Training epoch: {epoch} with train_loss: {train_loss}"
            )
            valid_loss, valid_miou = self.evaluate_epoch()
            self.logger.info(
                f"Trainer:: Training epoch: {epoch} with valid_loss: {valid_loss} and valid_miou: {valid_miou}"
            )
            is_best = self.save_best(valid_loss, valid_miou, epoch)
            self.logger.info(
                f"Trainer:: best loss | epoch={self.best_score['loss']['epoch']} | value={self.best_score['loss']['value']}"
            )
            self.logger.info(
                f"Trainer:: best miou | epoch={self.best_score['miou']['epoch']} | value={self.best_score['miou']['value']}"
            )

            if (epoch + 1) % self.ckpt_interval or is_best == 0:
                self.save_ckpt(epoch, train_loss, valid_loss, valid_miou, is_best)
                self.logger.info(f"Trainer:: Saved ckpt at epoch: {epoch}")

    def train_iter(self, img, target):
        pred = self.model(img)
        B, C, H, W = pred.shape
        pred = pred.permute(0, 2, 3, 1).reshape(B * H * W, C)
        target = target.view(B * H * W)
        loss = F.cross_entropy(pred, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for batch_idx, (img, target) in enumerate(self.train_data_loader):
            loss = self.train_iter(img.to(self.device), target.to(self.device))
            self.logger.info(
                f"training data({batch_idx}/{len(self.train_data_loader)}): loss: {loss}"
            )
            epoch_loss += loss
        epoch_loss = epoch_loss / (batch_idx + 1)
        return epoch_loss

    def evaluate_iter(self, img, target):
        pred = self.model(img)
        B, C, H, W = pred.shape
        miou = calculate_miou(pred, target)
        pred = pred.permute(0, 2, 3, 1).reshape(B * H * W, C)
        target = target.view(B * H * W)
        loss = F.cross_entropy(pred, target)
        return loss.item(), miou.item()

    def evaluate_epoch(self):
        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            epoch_loss = 0
            epoch_miou = 0
            for batch_idx, (img, target) in tqdm(
                enumerate(self.valid_data_loader), total=len(self.valid_data_loader)
            ):
                loss, miou = self.evaluate_iter(
                    img.to(self.device), target.to(self.device)
                )
                epoch_loss += loss
                epoch_miou += miou
            epoch_miou = epoch_miou / (batch_idx + 1)
            epoch_loss = epoch_loss / (batch_idx + 1)
        return epoch_loss, epoch_miou

    def save_best(self, loss, miou, epoch):
        is_best = False
        if loss < self.best_score["loss"]["value"]:
            self.best_score["loss"]["epoch"] = epoch
            self.best_score["loss"]["value"] = loss
            is_best = True
        if miou > self.best_score["miou"]["value"]:
            self.best_score["miou"]["epoch"] = epoch
            self.best_score["miou"]["value"] = miou
            is_best = True
        return is_best

    def save_ckpt(self, epoch, train_loss, valid_loss, miou, is_best):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "miou": miou,
        }
        best_str = "best_" if is_best else ""
        torch.save(ckpt, os.path.join(self.ckpt_save_dir, f"{best_str}ckpt_{epoch}.pt"))
