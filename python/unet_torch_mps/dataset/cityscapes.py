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


class CityScapesDataset(Dataset):
    def __init__(
        self,
        img_and_mask_dir: str,
        save_dir: Optional[str] = None,
        augment: bool = False,
        skip_img_mask_split: bool = False,
    ):
        assert os.path.isdir(img_and_mask_dir)
        assert save_dir is None or os.path.isdir(save_dir)
        self.img_and_mask_dir = img_and_mask_dir
        self.img_and_mask_paths = glob(os.path.join(img_and_mask_dir, "*.jpg"))
        self.id_map_array = np.zeros((3, len(id_map)), dtype=np.uint8)
        for key, val in id_map.items():
            self.id_map_array[:, key] = np.array(val)
        self.normalize_and_to_tensor = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        if not save_dir:
            save_dir = os.path.join(img_and_mask_dir, "split")
        self.img_dir = os.path.join(save_dir, "img")
        self.mask_dir = os.path.join(save_dir, "mask")

        if not skip_img_mask_split:
            os.makedirs(self.img_dir, exist_ok=True)
            os.makedirs(self.mask_dir, exist_ok=True)

            # split image and mask
            for i, img_and_mask_path in tqdm(enumerate(self.img_and_mask_paths)):
                img_and_mask = self.read_image_rgb(img_and_mask_path)
                img, mask = self.split_image_and_mask(img_and_mask)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
                file_name = os.path.basename(img_and_mask_path)
                cv2.imwrite(os.path.join(self.img_dir, file_name), img)
                cv2.imwrite(os.path.join(self.mask_dir, file_name), mask)

    def read_image_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def split_image_and_mask(self, img_and_mask):
        height = img_and_mask.shape[0]
        img = img_and_mask[:, :height]
        mask = img_and_mask[:, height:]
        return img, mask

    def _convert_mask_with_rgb_to_mask_with_id(
        self, mask: np.ndarray, id_map_array: np.ndarray
    ):
        assert mask.shape[2] == 3  # RGB
        assert id_map_array.shape[0] == 3  # RGB
        mask_expanded = mask[:, :, :, None]  # (H, W, 3, 1)
        id_map_array_expanded = id_map_array[None, None, :, :]  # (1, 1, 3, 30)
        distance = ((id_map_array_expanded - mask_expanded) ** 2).sum(
            axis=2
        )  # (H, W, 30)
        min_indices = np.argmin(distance, axis=2)  # (H, W)
        mask_with_id = min_indices.astype(np.uint8)
        return mask_with_id

    def __len__(self):
        return len(self.img_and_mask_paths)

    def __getitem__(self, idx):
        img_mask_path = self.img_and_mask_paths[idx]
        filename = os.path.basename(img_mask_path)
        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        img = self.read_image_rgb(img_path)
        mask = self.read_image_rgb(mask_path)
        mask = self._convert_mask_with_rgb_to_mask_with_id(mask, self.id_map_array)

        transformed = self.normalize_and_to_tensor(image=img, mask=mask)
        img, mask = transformed["image"], transformed["mask"]
        return img, mask[None]
