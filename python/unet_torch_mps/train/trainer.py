from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from unet_torch_mps.utils.utils import load_checkpoint, set_logger
from unet_torch_mps.metrics.iou import calculate_mean_iou


class Trainer:
    def __init__(
        self, model, optimizer, loss_fns, train_data_loader, valid_data_loader, config
    ):
        self.num_epochs = config["num_epochs"]
        self.lr = config["lr"]
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = config["device"]
        self.ckpt_path = config["ckpt"]
        self.loss_fns = loss_fns
        self.logger = set_logger()
        if self.ckpt_path is not None:
            self.start_epoch = load_checkpoint(
                self.model, self.optimizer, self.ckpt_path, self.logger
            )
        else:
            self.start_epoch = 0

        assert self.device in ["cpu", "cuda", "mps"]
        assert self.model.device == self.device

        self.logger.info(f"Trainer:: Using model: {self.model}")
        self.logger.info(f"Trainer:: Using device: {self.device}")
        self.logger.info(f"Trainer:: Starting epoch: {self.start_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info(f"Trainer:: Training epoch: {epoch}")
            train_loss, train_miou = self.train_epoch()
            self.logger.info(
                f"Trainer:: Training epoch: {epoch} with train_loss: {train_loss} and train_miou: {train_miou}"
            )
            valid_loss, valid_miou = self.evaluate_epoch()
            self.logger.info(
                f"Trainer:: Training epoch: {epoch} with valid_loss: {valid_loss} and valid_miou: {valid_miou}"
            )
            self.save_best()

    def train_iter(self, img, mask_gt):
        mask_pred_logit = self.model(img)
        loss = None
        for loss_fn in self.loss_fns:
            loss += loss_fn(mask_pred_logit, mask_gt)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        train_loss = loss.item()
        miou = calculate_mean_iou(mask_pred_logit, mask_gt)
        return train_loss, miou

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_miou = 0
        for batch_idx, (img, mask_gt) in enumerate(self.train_loader):
            iter_loss, iter_miou = self.train_iter(
                img.to(self.device), mask_gt.to(self.device)
            )
            self.logger.info(
                f"training data({batch_idx}/{len(self.train_loader)}): iter loss: {iter_loss} and iter miou: {iter_miou}"
            )
            total_loss += iter_loss
            total_miou += iter_miou
        avg_loss = total_loss / (batch_idx + 1)
        avg_miou = total_miou / (batch_idx + 1)
        return avg_loss, avg_miou

    def evaluate_iter(self, img, mask_gt):
        mask_pred_logit = self.model(img)
        loss = None
        for loss_fn in self.loss_fns:
            loss += loss_fn(mask_pred_logit, mask_gt)
        iter_loss = loss.item()
        iter_miou = calculate_mean_iou(mask_pred_logit, mask_gt)
        return iter_loss, iter_miou

    def evaluate_epoch(self):
        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            total_loss = 0
            total_miou = 0
            for batch_idx, (img, mask_gt) in tqdm(
                enumerate(self.valid_loader), total=len(self.valid_loader)
            ):
                iter_loss, iter_miou = self.evaluate_iter(
                    img.to(self.device), mask_gt.to(self.device)
                )
                total_loss += iter_loss
                total_miou += iter_miou
            avg_miou = total_miou / (batch_idx + 1)
            avg_loss = total_loss / (batch_idx + 1)
        return avg_loss, avg_miou

    def save_best(self):
        pass 