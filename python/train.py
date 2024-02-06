import os
import logging
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from unet_torch_mps.model.unet import Unet
from unet_torch_mps.model.resunet import ResUnet
from unet_torch_mps.dataset.cityscapes import CityScapesDataset, denormalize, id_values
from unet_torch_mps.loss.dice import TanimotoWithDualLoss

try:
    import wandb
    import matplotlib.pyplot as plt
except ImportError:
    wandb = None


def generate_dataloader(data_dir: str, split: str, batch_size: int):
    assert split in ["train", "val"]
    shuffle = drop_last = augment = split == "train"
    dataset = CityScapesDataset(
        data_dir=data_dir,
        split=split,
        augment=augment,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
    )
    return data_loader


def load_checkpoint(model, optimizer, ckpt_path, logger):
    if ckpt_path is not None:
        ckeckpoint = torch.load(ckpt_path)
        start_epoch = ckeckpoint["epoch"] + 1
        model.load_state_dict(ckeckpoint["model_state_dict"])
        optimizer.load_state_dict(ckeckpoint["optimizer_state_dict"])
        train_loss = ckeckpoint["train_loss"]
        valid_loss = ckeckpoint["valid_loss"]
        logger.info(
            f"Loaded ckpt from: {ckpt_path} @ epoch: {ckeckpoint['epoch']} with train_loss: {train_loss} and valid_loss: {valid_loss}"
        )
    else:
        start_epoch = 0
    return start_epoch


def train_epoch(model, train_loader, optimizer, device, logger, lossfn = None):
    model.train()
    epoch_loss = 0
    for batch_idx, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)
        pred = model(img)
        B, C, H, W = pred.shape
        if lossfn is not None:
            loss = lossfn(pred, target)
        else:
            target = target.view(B * H * W)
            pred = pred.permute(0, 2, 3, 1).reshape(B * H * W, C)
            loss = F.cross_entropy(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        logger.info(
            f"training data({batch_idx}/{len(train_loader)}): iter loss: {loss.item()}"
        )
    return epoch_loss / (batch_idx + 1)


def evaluate_epoch(model, valid_loader, device, lossfn=None):
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        epoch_loss = 0
        epoch_iou = 0
        gt_pred_list = []
        for batch_idx, (img, target) in tqdm(enumerate(valid_loader)):
            img, target = img.to(device), target.to(device)
            pred = model(img)
            pred_softmax = F.softmax(pred, dim=1)
            pred_argmax =pred_softmax.argmax(1)
            iou = (pred_argmax == target).float().mean()
            epoch_iou += iou.item()

            if wandb is not None and batch_idx <= 2:
                pred_argmax = pred_argmax[0].cpu().numpy()
                pred_argmax = id_values[pred_argmax]
                target_argmax = target[0].cpu().numpy()
                target_argmax = id_values[target_argmax]

                plt.subplot(1, 3, 1)
                plt.imshow(denormalize(img[0].permute(1, 2, 0).cpu().numpy()))
                plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.imshow(target_argmax)
                plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.imshow(pred_argmax)
                plt.axis("off")
                plt.tight_layout()
                gt_pred = wandb.Image(
                    plt,
                    caption=f"gt | pred",
                )
                gt_pred_list.append(gt_pred)

            B, C, H, W = pred.shape
            if lossfn is not None:
                loss = lossfn(pred_softmax, target)
            else:
                pred = pred.permute(0, 2, 3, 1).reshape(B * H * W, C)
                target = target.view(B * H * W)
                loss = F.cross_entropy(pred, target)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / (batch_idx + 1)
        epoch_iou = epoch_iou / (batch_idx + 1)
        if wandb is not None:
            wandb.log({"ground truths and predictions": gt_pred_list})
    return epoch_loss, epoch_iou


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def main(*args):
    if wandb is not None:
        wandb.init(project="unet-pytorch")
        wandb.config.update(args[0])
    args = args[0]
    num_classes = args.c
    epochs = args.e
    lr = args.lr
    ckpt_interval = args.ci
    ckpt_path = args.ckpt
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:2"
    else:
        device = "cpu"
    ckpt_save_dir = args.ckpt_save_dir
    os.makedirs(ckpt_save_dir, exist_ok=True)
    logger = set_logger()

    logger.info(f"device: {device}")
    # model = Unet(3, num_classes).to(device)
    model = ResUnet(3, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logger.info(f"number of parameters: {sum(p.numel() for p in model.parameters())}")

    valid_loader = generate_dataloader(args.d, "val", args.b)
    train_loader = (
        generate_dataloader(args.d, "train", args.b)
        if not args.sanity_check
        else valid_loader
    )
    start_epoch = load_checkpoint(model, optimizer, ckpt_path, logger)

    best_score = {
        "loss": {"epoch": 0, "value": float("inf")},
        "miou": {"epoch": 0, "value": 0},
    }

    tonimato_loss = TanimotoWithDualLoss(num_classes)

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(start_epoch, start_epoch + epochs):
            logger.info(f"epoch: {epoch}, lr:  {lr}")
            train_loss = train_epoch(model, train_loader, optimizer, device, logger, tonimato_loss)
            logger.info(f"training data: epoch loss: {train_loss}")

            valid_loss, mean_iou = evaluate_epoch(model, valid_loader, device, tonimato_loss)
            logger.info(
                f"validation data: avg loss: {valid_loss}, metric(mIoU): {mean_iou}"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "mean_iou": mean_iou,
                        "epoch": epoch,
                    }
                )
            best_checkpoint = False
            if valid_loss < best_score["loss"]["value"]:
                best_score["loss"]["epoch"] = epoch
                best_score["loss"]["value"] = valid_loss
                best_checkpoint = True
            if mean_iou > best_score["miou"]["value"]:
                best_score["miou"]["epoch"] = epoch
                best_score["miou"]["value"] = mean_iou
                best_checkpoint = True
            logger.info(
                f"best loss | epoch={best_score['loss']['epoch']} | value={best_score['loss']['value']}"
            )
            logger.info(
                f"best miou | epoch={best_score['miou']['epoch']} | value={best_score['miou']['value']}"
            )

            if (epoch + 1) % ckpt_interval == 0 or best_checkpoint:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    # Include any other relevant information
                }
                best_str = "best_" if best_checkpoint else ""
                torch.save(
                    checkpoint, os.path.join(ckpt_save_dir, f"{best_str}ckpt_{epoch}.pt")
                )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d", type=str, default="../training_data", help="dataset directory"
    )
    argparser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    argparser.add_argument("-e", type=int, default=5, help="number of epochs")
    argparser.add_argument("-c", type=int, default=30, help="number of classes")
    argparser.add_argument("-b", type=int, default=32, help="batch size")
    argparser.add_argument("-ci", type=int, default=5, help="ckpt interval")
    argparser.add_argument(
        "-sanity_check",
        action="store_true",
        help="sanity check mode. use validation data for training",
    )
    argparser.add_argument("-ckpt", type=str, default=None, help="ckpt path")
    argparser.add_argument(
        "-ckpt_save_dir", type=str, default="../tmp", help="dir to save ckpt"
    )
    args = argparser.parse_args()
    main(args)
