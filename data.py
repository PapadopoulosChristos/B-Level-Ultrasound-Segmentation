import torch
import numpy as np
import os
from glob import glob
from torch.utils.data import Dataset
import cv2
import albumentations as A

#  Class for the HC18 grandchallenge US dataset.
class UltrasoundDataset(Dataset):
    def __init__(self, data_dir, split):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        
        # Set different augmentations for training and validation.
        if split == "train":
            self.transforms = A.Compose(
                [
                    A.RandomBrightnessContrast(0.2, 0, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomCrop(height=270, width=400, p=0.2),
                    A.PadIfNeeded(
                        min_height=544,
                        min_width=800,
                        mask_value=cv2.BORDER_REFLECT,
                    ),
                    A.Resize(
                        height=544,
                        width=800,
                        always_apply=True,
                    ),
                ]
            )
        else:
            self.transforms = A.Compose(
                [
                    A.PadIfNeeded(
                        min_height=544,
                        min_width=800,
                        mask_value=cv2.BORDER_REFLECT,
                    ),
                    A.Resize(
                        height=544,
                        width=800,
                        always_apply=True,
                    ),
                ]
            )

        
        # Read files from split file
        # (manually set to be 90% train + 10% val from provided dataset)
        with open(
            os.path.join(self.data_dir, f"{split}_data.txt"), "r", encoding="utf-8"
        ) as f:
            lines = np.asarray(
                [
                    os.path.join(self.data_dir, i).strip().split(" ")
                    for i in f.readlines()
                ]
            )

        self.images, self.masks = lines[:, 0], lines[:, 1]

        # Set indexes for shuffling capability.
        self.idxs = np.arange(len(self.images))
        np.random.shuffle(self.idxs)

    def __len__(self):
        # Get dataset length.
        return len(self.images)

    def __getitem__(self, index):
        # Read image and mask.
        image = cv2.imread(self.images[self.idxs[index]], 0)
        mask = cv2.imread(self.masks[self.idxs[index]], 0)

        # process mask.
        conts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = cv2.fillPoly(mask, [conts[0][0].squeeze()], color=(255, 255, 255))

        # Get augmented samples.
        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        # Format samples to torch tensors.
        image = torch.Tensor(image[np.newaxis, :, :] / 255)
        mask = torch.Tensor(mask[np.newaxis, :, :] / 255)

        return image, mask


def create_splits():
    # Get training images and train labels.
    train_images = [i for i in glob("./training_set/*.png") if "Annotation" not in i]
    train_labels = [i for i in glob("./training_set/*.png") if "Annotation" in i]

    # Write training files.
    with open("./train_data.txt", "w", encoding="utf-8") as f:
        for i, j in zip(train_images[:-99], train_labels[:-99]):
            f.write(f"{i[2:]} {j[2:]}\n")
    
    # Write validation files.
    with open("./val_data.txt", "w", encoding="utf-8") as f:
        for i, j in zip(train_images[-99:], train_labels[-99:]):
            f.write(f"{i[2:]} {j[2:]}\n")

# create_splits()