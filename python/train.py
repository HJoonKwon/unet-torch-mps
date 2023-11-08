import argparse
import torch
from unet_torch_mps.model.unet import Unet
from unet_torch_mps.dataset.cityscapes import CityScapesDataset
import torch.nn.functional as F
from tqdm import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"


def main(*args):
    args = args[0]
    num_classes = args.c
    epochs = args.e
    lr = args.lr
    model = Unet(3, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = (
        torch.nn.CrossEntropyLoss() if num_classes > 1 else torch.nn.BCEWithLogitsLoss()
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
            img_height, img_width = img.shape[2], img.shape[3]
            mask_pred = model(img)
            loss = loss_fn(mask_pred, mask_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss = avg_loss / len(train_loader)
        print("avg loss: ", avg_loss, "lr: ", lr)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d", type=str, default="../training_data/val", help="dataset directory"
    )
    argparser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    argparser.add_argument("-e", type=int, default=5, help="number of epochs")
    argparser.add_argument("-c", type=int, default=31, help="number of classes")
    argparser.add_argument("-b", type=int, default=4, help="batch size")
    args = argparser.parse_args()
    main(args)
