import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unet_torch_mps.model.unet import Unet
from unet_torch_mps.dataset.cityscapes import CityScapesDataset
from unet_torch_mps.metrics.iou import IoU


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
    optimizer = torch.optim.Adam(model.parameters())
    if ckpt_path is not None:
        ckeckpoint = torch.load(ckpt_path)
        start_epoch = ckeckpoint["epoch"]
        model.load_state_dict(ckeckpoint["model_state_dict"])
        optimizer.load_state_dict(ckeckpoint["optimizer_state_dict"])
        train_loss = ckeckpoint["train_loss"]
        valid_loss = ckeckpoint["valid_loss"]
        print(
            f"Loaded ckpt from: {ckpt_path} @ epoch: {start_epoch} with train_loss: {train_loss} and valid_loss: {valid_loss}"
        )
    else:
        start_epoch = 0
    loss_fn = (
        torch.nn.CrossEntropyLoss() if num_classes > 1 else torch.nn.BCEWithLogitsLoss()
    )
    metric_fn = IoU(num_classes=num_classes, task="multiclass", per_class=False)
    data_dir = args.d
    is_sanity_check = args.sanity_check
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
        train_dataset, batch_size=args.b, shuffle=True, pin_memory=True, drop_last=True
    )
    valid_dataset = CityScapesDataset(
        img_and_mask_dir=validation_data_dir,
        skip_img_mask_split=not is_create_dataset,
        augment=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.b,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(start_epoch, start_epoch + epochs):
        print("epoch: ", epoch)
        train_loss = 0
        for batch_idx, (img, mask_gt) in enumerate(train_loader):
            img, mask_gt = img.to(device), mask_gt.to(device)
            mask_pred_logit = model(img)
            loss = loss_fn(mask_pred_logit, mask_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss / (batch_idx + 1) * batch_idx + loss.item() / (
                batch_idx + 1
            )
            mask_pred = torch.argmax(mask_pred_logit, dim=1)
            print(
                f"training data({batch_idx}/{len(train_loader)}): avg loss: {train_loss}, lr: {lr}"
            )

        metric = metric_fn(mask_pred.detach().cpu(), mask_gt.detach().cpu()).item()
        print(
            f"training data: avg loss: {train_loss}, metric(mIoU): {metric}, lr: {lr}"
        )

        with torch.no_grad():
            valid_loss = 0
            for batch_idx, (img, mask_gt) in tqdm(
                enumerate(valid_loader), total=len(valid_loader)
            ):
                img, mask_gt = img.to(device), mask_gt.to(device)
                model.eval()  # set model to evaluation mode
                mask_pred_logit = model(img)
                loss = loss_fn(mask_pred_logit, mask_gt)
                valid_loss += loss.item()
                mask_pred = torch.argmax(mask_pred_logit, dim=1)
            valid_loss = valid_loss / len(valid_loader)
            valid_metric = metric_fn(
                mask_pred.detach().cpu(), mask_gt.detach().cpu()
            ).item()
            print(
                f"validation data: avg loss: {valid_loss}, metric(mIoU): {valid_metric}"
            )
        model.train()  # set model to training mode
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
