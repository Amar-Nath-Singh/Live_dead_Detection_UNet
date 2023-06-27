from config import *
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np


class CellDataset(Dataset):
    def __init__(self, list_img, transform=None) -> None:
        super(CellDataset).__init__()
        self.list_files = list_img
        self.transform = transform

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        alive_path = img_file[:-5] + "1.jpg"
        dead_path = img_file[:-5] + "2.jpg"
        input_image = np.array(Image.open(img_file))
        alive = np.array(Image.open(alive_path))
        dead = np.array(Image.open(dead_path))
        if IN_CHANNELS == 1:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

            # dead_mask,_, _ = cv2.split(dead)
            # _, alive_mask, _ = cv2.split(alive)
            # mask = np.zeros((alive_mask.shape[0], alive_mask.shape[1],1))
            # mask[dead_mask > 75] = 1
            # mask[alive_mask > 75] = 2

            hsv_image = cv2.cvtColor(dead, cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 1] = hsv_image[:, :, 1] * 20
            red_mask = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            hsv_image = cv2.cvtColor(alive, cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 1] = hsv_image[:, :, 1] * 50
            green_mask = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            red_mask = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)
            green_mask = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)

            mask = np.zeros((red_mask.shape[0], red_mask.shape[1], 1))

            mask[green_mask > 50] = 2
            mask[red_mask > 60] = 1

        if self.transform is not None:
            aug = self.transform(image=input_image, mask=mask)
            input_image = aug["image"]
            mask = aug["mask"]
            mask = torch.max(mask, dim=2)[0]
            
        return input_image, mask
