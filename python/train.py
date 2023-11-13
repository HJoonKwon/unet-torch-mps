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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_save_dir = "ckpt"
    os.makedirs(ckpt_save_dir, exist_ok=True)

    model = Unet(3, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = (
        torch.nn.CrossEntropyLoss() if num_classes > 1 else torch.nn.BCEWithLogitsLoss()
    )
    metric_fn = IoU(num_classes=num_classes, task="multiclass", per_class=False).to(
        device
    )

    train_dataset = CityScapesDataset(img_and_mask_dir=args.d, skip_img_mask_split=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.b, shuffle=True, pin_memory=True, drop_last=True
    )
    for epoch in range(epochs):
        print("epoch: ", epoch)
        avg_loss = 0
        for batch_idx, (img, mask_gt) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            img, mask_gt = img.to(device), mask_gt.to(device)
            mask_pred_logit = model(img)
            loss = loss_fn(mask_pred_logit, mask_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            mask_pred = torch.argmax(mask_pred_logit, dim=1)
        metric = metric_fn(mask_pred, mask_gt)
        avg_loss = avg_loss / len(train_loader)
        print("avg loss: ", avg_loss, "lr: ", lr, "metric(mIoU): ", metric)
        if (epoch + 1) % ckpt_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # Include any other relevant information
            }
            torch.save(checkpoint, os.path.join(ckpt_save_dir, f"ckpt_{epoch}.pt"))
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d", type=str, default="../training_data/val", help="dataset directory"
    )
    argparser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    argparser.add_argument("-e", type=int, default=5, help="number of epochs")
    argparser.add_argument("-c", type=int, default=31, help="number of classes")
    argparser.add_argument("-b", type=int, default=4, help="batch size")
    argparser.add_argument("-ci", type=int, default=5, help="ckpt interval")
    args = argparser.parse_args()
    main(args)
