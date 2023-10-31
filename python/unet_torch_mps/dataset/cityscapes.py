import os
import torch
from torch.utils.data import Dataset
import torchvision as tv
from glob import glob

id_map = {
    0: (0, 0, 0),  # unlabelled
    1: (111, 74, 0),  # static
    2: (81, 0, 81),  # ground
    3: (128, 64, 127),  # road
    4: (244, 35, 232),  # sidewalk
    5: (250, 170, 160),  # parking
    6: (230, 150, 140),  # rail track
    7: (70, 70, 70),  # building
    8: (102, 102, 156),  # wall
    9: (190, 153, 153),  # fence
    10: (180, 165, 180),  # guard rail
    11: (150, 100, 100),  # bridge
    12: (150, 120, 90),  # tunnel
    13: (153, 153, 153),  # pole
    14: (153, 153, 153),  # polegroup
    15: (250, 170, 30),  # traffic light
    16: (220, 220, 0),  # traffic sign
    17: (107, 142, 35),  # vegetation
    18: (152, 251, 152),  # terrain
    19: (70, 130, 180),  # sky
    20: (220, 20, 60),  # person
    21: (255, 0, 0),  # rider
    22: (0, 0, 142),  # car
    23: (0, 0, 70),  # truck
    24: (0, 60, 100),  # bus
    25: (0, 0, 90),  # caravan
    26: (0, 0, 110),  # trailer
    27: (0, 80, 100),  # train
    28: (0, 0, 230),  # motorcycle
    29: (119, 11, 32),  # bicycle
    30: (0, 0, 142),  # license plate
}

id_map_tensor = torch.zeros((3, len(id_map)))
for key, val in id_map.items():
    id_map_tensor[:, key] = torch.Tensor(val)


class CityScapesDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = glob(os.path.join(img_dir, "*.jpg"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_and_mask = tv.io.read_image(img_path)
        height = img_and_mask.shape[1]
        img = img_and_mask[:, :, :height]
        mask = img_and_mask[:, :, height:]

        mask_expanded = mask.unsqueeze(1)
        id_map_expanded = id_map_tensor.unsqueeze(-1).unsqueeze(-1)

        distance = ((id_map_expanded - mask_expanded) ** 2).sum(dim=0)
        _, min_indices = torch.min(distance, dim=0)
        for i in range(id_map_tensor.shape[1]):
            id_mask = min_indices == i
            mask[:, id_mask] = i

        return img, mask[0, :, :].unsqueeze(0)
