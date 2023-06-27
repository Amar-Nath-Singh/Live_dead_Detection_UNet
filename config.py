import torch
import random
import glob
from albumentations.pytorch import ToTensorV2
import albumentations as A

SEED = 69
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
DATA_LIST = list(glob.glob("dataset/**/*.0.jpg", recursive=True))
random.shuffle(DATA_LIST)
TRAIN_SIZE = int(len(DATA_LIST) * 0.8)
TEST_SIZE = len(DATA_LIST) - TRAIN_SIZE
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
BATCH_SIZE = 4
NUM_CLASSESS = 3
IN_CHANNELS = 1
LEARNING_RATE = 1e-4
num_epochs = 10
class_weights = torch.tensor([0.25, 5.0, 1.0]).to(DEVICE)
transform_val = A.Compose(
    [
        A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform = A.Compose(
    [
        A.RandomResizedCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
        # A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5] * IN_CHANNELS, std=[0.5] * IN_CHANNELS),
        ToTensorV2(),
    ]
)
