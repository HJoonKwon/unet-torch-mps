import os
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from unet_torch_mps.model.unet import Unet
from unet_torch_mps.dataset.cityscapes import CityScapesDataset
from unet_torch_mps.metrics.iou import calculate_mean_iou
from unet_torch_mps.loss.dice import DiceLoss


def generate_dataloader(
    data_dir: str, is_sanity_check: bool, is_create_dataset: bool, batch_size: int
):
    validation_data_dir = os.path.join(data_dir, "val")
    training_data_dir = (
        os.path.join(data_dir, "train") if not is_sanity_check else validation_data_dir
    )

    train_dataset = CityScapesDataset(
        img_and_mask_dir=training_data_dir,
        skip_img_mask_split=not is_create_dataset,
        augment=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataset = CityScapesDataset(
        img_and_mask_dir=validation_data_dir,
        skip_img_mask_split=not is_create_dataset,
        augment=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, valid_loader


def load_checkpoint(model, optimizer, ckpt_path):
    if ckpt_path is not None:
        ckeckpoint = torch.load(ckpt_path)
        start_epoch = ckeckpoint["epoch"] + 1
        model.load_state_dict(ckeckpoint["model_state_dict"])
        optimizer.load_state_dict(ckeckpoint["optimizer_state_dict"])
        train_loss = ckeckpoint["train_loss"]
        valid_loss = ckeckpoint["valid_loss"]
        print(
            f"Loaded ckpt from: {ckpt_path} @ epoch: {ckeckpoint['epoch']} with train_loss: {train_loss} and valid_loss: {valid_loss}"
        )
    else:
        start_epoch = 0
    return start_epoch


def train_epoch(model, train_loader, optimizer, loss_fns: list, device):
    model.train()
    train_loss = 0
    total_miou = 0
    cross_entropy_loss_fn, dice_loss_fn = loss_fns
    for batch_idx, (img, mask_gt) in enumerate(train_loader):
        img, mask_gt = img.to(device), mask_gt.to(device)
        mask_pred_logit = model(img)
        loss = cross_entropy_loss_fn(mask_pred_logit, mask_gt) + dice_loss_fn(
            mask_pred_logit, mask_gt
        )
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss = train_loss / (batch_idx + 1) * batch_idx + loss.item() / (
            batch_idx + 1
        )
        bactch_miou = calculate_mean_iou(mask_pred_logit, mask_gt)
        total_miou += bactch_miou
        print(
            f"training data({batch_idx}/{len(train_loader)}): avg loss: {train_loss}"
        )
    mean_iou = total_miou / (batch_idx + 1)
    return train_loss, mean_iou


def evaluate_epoch(model, valid_loader, loss_fns, device):
    model.eval()  # set model to evaluation mode
    cross_entropy_loss_fn, dice_loss_fn = loss_fns
    with torch.no_grad():
        valid_loss = 0
        total_miou = 0
        for batch_idx, (img, mask_gt) in tqdm(
            enumerate(valid_loader), total=len(valid_loader)
        ):
            img, mask_gt = img.to(device), mask_gt.to(device)
            mask_pred_logit = model(img)
            loss = cross_entropy_loss_fn(mask_pred_logit, mask_gt) + dice_loss_fn(
                mask_pred_logit, mask_gt
            )
            valid_loss += loss.item()
            batch_miou = calculate_mean_iou(mask_pred_logit, mask_gt)
            total_miou += batch_miou
        mean_iou = total_miou / (batch_idx + 1)
        valid_loss = valid_loss / len(valid_loader)
    return valid_loss, mean_iou


def main(*args):
    args = args[0]
    num_classes = args.c
    epochs = args.e
    lr = args.lr
    ckpt_interval = args.ci
    ckpt_path = args.ckpt
    is_create_dataset = args.create_dataset
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    ckpt_save_dir = "ckpt"
    os.makedirs(ckpt_save_dir, exist_ok=True)

    model = Unet(3, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = (
        torch.nn.CrossEntropyLoss() if num_classes > 1 else torch.nn.BCEWithLogitsLoss()
    )
    dice_loss_fn = DiceLoss()
    train_loader, valid_loader = generate_dataloader(
        args.d, args.sanity_check, is_create_dataset, args.b
    )
    start_epoch = load_checkpoint(model, optimizer, ckpt_path)

    best_score = {
        "loss": {"epoch": 0, "value": float("inf")},
        "miou": {"epoch": 0, "value": 0},
    }

    for epoch in range(start_epoch, start_epoch + epochs):
        print("epoch: ", epoch, "lr: ", lr)
        train_loss, mean_iou = train_epoch(
            model, train_loader, optimizer, [loss_fn, dice_loss_fn], device
        )
        print(
            f"training data: epoch loss: {train_loss}, epoch metric(mIoU): {mean_iou}"
        )

        valid_loss, mean_iou = evaluate_epoch(
            model, valid_loader, [loss_fn, dice_loss_fn], device
        )
        print(f"validation data: avg loss: {valid_loss}, metric(mIoU): {mean_iou}")
        if valid_loss < best_score["loss"]["value"]:
            best_score["loss"]["epoch"] = epoch
            best_score["loss"]["value"] = valid_loss
        if mean_iou > best_score["miou"]["value"]:
            best_score["miou"]["epoch"] = epoch
            best_score["miou"]["value"] = mean_iou
        print(f"best score: {best_score}")

        if (epoch + 1) % ckpt_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                # Include any other relevant information
            }
            torch.save(checkpoint, os.path.join(ckpt_save_dir, f"ckpt_{epoch}.pt"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d", type=str, default="../training_data", help="dataset directory"
    )
    argparser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    argparser.add_argument("-e", type=int, default=5, help="number of epochs")
    argparser.add_argument("-c", type=int, default=31, help="number of classes")
    argparser.add_argument("-b", type=int, default=4, help="batch size")
    argparser.add_argument("-ci", type=int, default=1, help="ckpt interval")
    argparser.add_argument(
        "-sanity_check",
        action="store_true",
        help="sanity check mode. use validation data for training",
    )
    argparser.add_argument("-create_dataset", action="store_true")
    argparser.add_argument("-ckpt", type=str, default=None, help="ckpt path")
    args = argparser.parse_args()
    main(args)
