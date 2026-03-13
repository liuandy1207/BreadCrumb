################################################################################
#                              Primary Model
################################################################################

# key imports
import torch

# data loading imports
import os     # to manage files
import json   # to read COCO annotations
from torch.utils.data import Dataset    # to make a custom dataset
import torchvision.transforms as transforms     # to process images
import matplotlib.pyplot as plt         # to get images
from torch.utils.data import DataLoader     # to make data loaders

################################################################################
#                              Data Loading
################################################################################

# custom dataset class to be able to use data loaders
# note: data was exported from Roboflow in COCO
class COCODataset(Dataset):
    def __init__(self, path, grid_size=8, num_classes=5):
        self.path = path
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            # normalize between [-1, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        # open COCO .json file
        with open(os.path.join(path, "_annotations.coco.json")) as f:
            coco = json.load(f)     # load as dictionary
        self.images = coco["images"]    # get image data
        self.annotations = coco["annotations"]  # get bounding boxes

        # build fast look-up table
        self.ann_map = {}
        for ann in self.annotations:
            img_id = ann["image_id"]     # find binding box parent image
            # add annotation to parent image
            if img_id not in self.ann_map:
                self.ann_map[img_id] = []
            self.ann_map[img_id].append(ann)

    # stuff required by Dataset
    def __len__(self):
        return len(self.images)

    def __get_item__(self, idx):
        img_info = self.images[idx]     # get image data
        img_id = img_info["id"]         # get unique id
        img_path = os.path.join(self.path, img_info["file_name"]) # get img path
        image = plt.imread(img_path)    # load image as RGB
        image = self.transform(image)   # convert to tensor

        # make empty grid target tensor, for storing correct classifications
        label = torch.zeros(
            self.grid_size,
            self.grid_size,
            1 + self.num_classes
        )

        if img_id in self.ann_map:
            for ann in self.ann_map[img_id]:
                x, y, w, h = ann["bbox"]
                # find center to see which grid in belongs to
                xc = (x + w/2) / img_info["width"]
                yc = (y + h/2) / img_info["height"]
                gx = int(xc * self.grid_size)
                gy = int(yc * self.grid_size)
                # clamp to valid values
                gx = min(gx,self.grid_size-1)
                gy = min(gy,self.grid_size-1)

                cls = ann["category_id"]    # get class
                label[gy,gx,0] = 1  # first index indicates objectness
                label[gy,gx,1+cls] = 1 # later indexes are one-hot encoding

        return image, label

def get_data_loaders(batch_size, grid_size=8, num_classes=5):
    classes = ["resistor","led","push button","logic gate","capacitor"]
    train_dataset = COCODataset(
        folder="dataset/train",
        grid_size=grid_size,
        num_classes=num_classes
    )

    val_dataset = COCODataset(
        folder="dataset/val",
        grid_size=grid_size,
        num_classes=num_classes
    )

    test_dataset = COCODataset(
        folder="dataset/test",
        grid_size=grid_size,
        num_classes=num_classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader, classes















    














