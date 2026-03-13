################################################################################
#                              Primary Model
################################################################################

# key imports
import torch

# data loading imports


from torch.utils.data import DataLoader
import torchvision.transforms as transforms


################################################################################
#                              Data Loading
################################################################################

# custom dataset class to be able to use data loaders
# note: data was exported from Roboflow in COCO
class 















# helper functions
def get_data_loaders(batch_size, grid_size=8, num_classes=5):
    # tensor transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),
        # normalize between [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = BreadboardCOCODataset(
        folder="dataset/train",
        grid_size=grid_size,
        num_classes=num_classes
    )
