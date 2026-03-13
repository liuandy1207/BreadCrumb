################################################################################
#                              Primary Model
################################################################################

# key imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models     # for transfer learning

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

    def __getitem__(self, idx):
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

                cls = ann["category_id"] - 1   # get class, temp solution for empty class
                label[gy,gx,0] = 1  # first index indicates objectness
                label[gy,gx,1+cls] = 1 # later indexes are one-hot encoding

        return image, label

def get_data_loaders(batch_size, grid_size=8, num_classes=5):
    classes = ["resistor","led","push button","logic gate","capacitor"]
    train_dataset = COCODataset(
        path="dataset/train",
        grid_size=grid_size,
        num_classes=num_classes
    )

    val_dataset = COCODataset(
        path="dataset/valid",
        grid_size=grid_size,
        num_classes=num_classes
    )

    test_dataset = COCODataset(
        path="dataset/test",
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

################################################################################
#                                 Network
################################################################################

class Primary(nn.Module):
    def __init__(self, grid_size=8, num_classes=5):
        super(Primary, self).__init__()
        self.name = "Primary"
        self.grid_size = grid_size
        self.num_classes = num_classes

        transfer = models.resnet18(pretrained=True)
        # remove final classifer layer, cuz we don't need those classes
        self.features = nn.Sequential(*list(transfer.children())[:-1])
        # freeze early layers
        for param in self.features.parameters():
            param.requires_grad = False

        # summarize into grids
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        # classify
        self.head = nn.Conv2d(512, 1 + num_classes, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1) # rearrange into BLWC, easier
        objectness = torch.sigmoid(x[..., 0:1])
        class_probs = F.softmax(x[..., 1:], dim=-1)
        return torch.cat([objectness, class_probs], dim=-1)


################################################################################
#                                Train Model
################################################################################

def train_model(net, train_loader, val_loader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        net.train()
        train_loss = 0
        train_error = 0 # sum of grid level

        for images, targets, in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = net(images)

            # note to self, probably better to use the logit loss versions
            obj_loss = F.binary_cross_entropy(
                preds[..., 0],
                targets[..., 0]
            )
            # note to self: code optimize by only doing this when objectness
            # class loss
            cls_loss = F.cross_entropy(
                preds[...,1:],
                targets[...,1:]
            )
            loss = obj_loss + cls_loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_obj = (preds[..., 0] > 0.5).float()
            pred_class = preds[...,1:].argmax(dim=-1)
            true_obj = targets[...,0]
            true_class = targets[..., 1:].argmax(dim=-1)
            obj_acc = (pred_obj == true_obj).float().mean()
            cls_acc = (pred_class[true_obj==1] == true_class[true_obj==1]).float().mean()
            train_error += 1 - (0.5*obj_acc + 0.5*cls_acc)

        train_loss /= len(train_loader)
        train_error /= len(train_loader)
    
    # evaluate
        net.eval()
        val_loss = 0
        val_error = 0

        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            preds = net(images)

            obj_loss = F.binary_cross_entropy(
                preds[...,0],
                targets[...,0]
            )

            cls_loss = F.cross_entropy(
                preds[...,1:],
                targets[...,1:]
            )

            loss = obj_loss + cls_loss
            val_loss += loss.item()

            pred_obj = (preds[..., 0] > 0.5).float()
            pred_class = preds[...,1:].argmax(dim=-1)
            true_obj = targets[...,0]
            true_class = targets[..., 1:].argmax(dim=-1)
            obj_acc = (pred_obj == true_obj).float().mean()
            cls_acc = (pred_class[true_obj==1] == true_class[true_obj==1]).float().mean()
            val_error += 1 - (0.5*obj_acc + 0.5*cls_acc)

        val_loss /= len(train_loader)
        val_error /= len(train_loader)
        print(f"Epoch {epoch+1}")
        print("Train Loss:", train_loss)
        print("Train Error:", train_error.item())
        print("Val Loss:", val_loss)
        print("Val Error:", val_error.item())
        print("-------------")


######
import torch
import torch.nn.functional as F

def sanity_check(model, dataset, num_images=2, epochs=50, lr=1e-2, device=None):
    """
    Train the model on a small subset to check if it can overfit.
    
    Args:
        model: your Primary model
        dataset: PyTorch Dataset (COCODataset)
        num_images: number of images to use for sanity check
        epochs: number of epochs to train
        lr: learning rate
        device: "cuda" or "cpu"
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # small subset loader
    subset = torch.utils.data.Subset(dataset, range(num_images))
    loader = torch.utils.data.DataLoader(subset, batch_size=num_images, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        total_loss = 0
        total_obj = 0
        correct_obj = 0
        total_cls = 0
        correct_cls = 0

        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)

            # BCE loss
            obj_loss = F.binary_cross_entropy(preds[...,0], targets[...,0])
            cls_loss = F.cross_entropy(preds[...,1:], targets[...,1:])
            loss = obj_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # objectness accuracy
            pred_obj = (preds[...,0] > 0.5).float()
            correct_obj += (pred_obj == targets[...,0]).sum().item()
            total_obj += targets[...,0].numel()

            # class accuracy (only where object exists)
            mask = targets[...,0] == 1
            if mask.sum() > 0:
                pred_cls = torch.argmax(preds[...,1:], dim=-1)
                true_cls = torch.argmax(targets[...,1:], dim=-1)
                correct_cls += (pred_cls[mask] == true_cls[mask]).sum().item()
                total_cls += mask.sum().item()

        print(
            f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, "
            f"Obj Acc={correct_obj/total_obj:.4f}, "
            f"Cls Acc={correct_cls/total_cls if total_cls>0 else 0:.4f}"
        )



################################################################################
#                                main()
################################################################################

if __name__ == "__main__":
    torch.manual_seed(1)
    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=8, grid_size=8)

    train_dataset = COCODataset("dataset/train", grid_size=8, num_classes=5)

    primary = Primary(grid_size = 8)
    sanity_check(primary, train_dataset, num_images=20, epochs=50, lr=1e-2, device=None)
    # train_model(primary, train_loader, val_loader, epochs=10, lr = 1e-3)











    














