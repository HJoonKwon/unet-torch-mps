import os
from torch.utils.data import Dataset
import cv2
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional
from tqdm import tqdm
import numpy as np

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
id_values = np.array(list(id_map.values()), dtype=np.uint8)
id_keys = np.array(list(id_map.keys()), dtype=np.uint8)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
max_pixel_value = 255.0


def denormalize(
    tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value=255.0,
    dtype=np.uint8,
):
    assert len(tensor.shape) == 3, "Tensor shape should be (H, W, C)"
    assert tensor.shape[2] == 3, "Tensor shape should be (H, W, C)"

    mean = np.array(mean, dtype=np.float32) * max_pixel_value  # (3, )
    std = np.array(std, dtype=np.float32) * max_pixel_value  #  (3, )

    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)

    tensor = tensor * std + mean
    tensor = np.clip(tensor.astype(dtype), 0, 255)

    return tensor


class CityScapesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # 'train' or 'val'
        augment: bool = False,
        img_size=(256, 256),  # (W, H)
    ):
        data_dir = os.path.join(data_dir, split)
        assert os.path.isdir(data_dir)
        self.data_dir = data_dir
        self.img_files = glob(os.path.join(data_dir, "img/*.jpg"))
        self.mask_files = glob(os.path.join(data_dir, "mask/*.jpg"))
        self.id_map_array = np.zeros((3, len(id_map)), dtype=np.uint8)
        for key, val in id_map.items():
            self.id_map_array[:, key] = np.array(val)
        self.normalize_and_to_tensor = A.Compose(
            [
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=max_pixel_value,
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )
        self.augment = self._construct_augmentation(*img_size) if augment else None

    def read_image_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _convert_mask_with_rgb_to_mask_with_id(self, mask: np.ndarray):
        """
        Reference:
            1. https://www.kaggle.com/code/tr1gg3rtrash/car-driving-segmentation-unet-from-scratch
            2. https://github.com/WhiteWolf47/cscapes_semantic_segmentation
        """
        values = id_values.reshape(1, 1, -1, 3)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1, -1)
        min_indexes = np.argmin(np.linalg.norm(values - mask, axis=3), axis=2)
        mask = id_keys[min_indexes][:, :, np.newaxis]
        return mask

    def _construct_augmentation(self, img_width, img_height):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomFog(p=0.2),
                A.RandomRain(p=0.2),
                A.RGBShift(p=0.2),
                A.RandomSunFlare(p=0.2),
                A.MotionBlur(p=0.2),
                A.Sharpen(p=0.2),
                A.ISONoise(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, p=0.3),
                A.Resize(img_height, img_width, p=1.0),
            ],
            p=0.9,
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = self.read_image_rgb(self.img_files[idx])
        mask = self.read_image_rgb(self.mask_files[idx])
        mask = self._convert_mask_with_rgb_to_mask_with_id(mask)
        if self.augment is not None:
            transformed = self.augment(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]
        transformed = self.normalize_and_to_tensor(image=img, mask=mask)
        img, mask = transformed["image"], transformed["mask"]
        return img, mask
