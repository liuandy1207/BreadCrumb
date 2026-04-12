################################################################################
#                                BreadCrumb
#
# pipeline: 
# 1. start with images of a breadboard with circuit components on it
# 2. create a window (patch) that slides across the image,
#    if the centre of a component is in the window,
#    then classify if confident enough, making bounding boxes basically
# 3. apply non-maxing supression (NMS) to collapse bounding boxes into the
#    the most confident inferences
# 4. count and return the number of bounding boxes remaining
################################################################################

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

import json     # for accessing the COCO annotations
import os       # for accessing files


import glob

# for graphing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

################################################################################
#                       Hyperparameters, kinda
################################################################################

PATCH_SIZE  = 48   # size of the bounding boxes
STRIDE = 16         # how much the window moves around
CONF_THRESH = 0.99  # minimum confidence to count as a detection
IOU_THRESH = 0.0001     # amount of overlap for collapsing

# obtained from loader function instead of hardcoded
CLASSES = None
NUM_CLASSES = None

# real hyperparameters
BATCH_SIZE = 32

################################################################################
#                                  Device
################################################################################

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # for mac
    if torch.cuda.is_available():   
        return torch.device("cuda") # for nvidia GPUs
    return torch.device("cpu")

################################################################################
#                               Load Data
################################################################################

def load_category_map(json_path):
    # note: i accidentally left a category "electric components" in my
    #       Roboflow export, so we ignore index 0 and use it as "background"
    with open(json_path) as f:
        coco = json.load(f)

    # skip the top-level supercategory (supercategory == "none")
    cats = [c for c in coco["categories"] if c["supercategory"] != "none"]
    cats.sort(key=lambda c: c["id"])    # sort to build list in order
    max_id = max(c["id"] for c in cats) # get the highest id

    # build a list of classes, starting with ["background"] at 0
    classes = ["background"] + [""] * max_id
    for c in cats:
        classes[c["id"]] = c["name"]

    return classes, len(classes)

def find_annotations_json(split_path):
    return os.path.join(split_path, "_annotations.coco.json")

################################################################################
#                            Custom Dataset
################################################################################

class PatchDataset(Dataset): # to turn image data into patches
    def __init__(self, path, classes, patch_size=PATCH_SIZE, stride=STRIDE):
        self.patch_size = patch_size
        self.classes = classes

        self.transform = transforms.Compose([
            transforms.ToPILImage(),        # convert back into an image
            transforms.Resize((patch_size, patch_size)),  # so we can resize
            transforms.ToTensor(),          # back to a tensor
            transforms.Normalize([0.485, 0.456, 0.406],   # so we can normalize
                                  [0.229, 0.224, 0.225]),
        ])

        # get json annotations
        ann_json = find_annotations_json(path)
        with open(ann_json) as f:
            coco = json.load(f)

        # Datasets need an annotation map, so build that
        ann_map = {}
        for ann in coco["annotations"]:
            ann_map.setdefault(ann["image_id"], []).append(ann)

        self.patches = []

        for img_info in coco["images"]:
            img_path = os.path.join(path, img_info["file_name"])
            image = plt.imread(img_path)
            image = image[..., :3]  # drop alpha channel for rgba image
            H, W  = image.shape[:2]
            anns  = ann_map.get(img_info["id"], [])

            # calculate centres of bounding boxes to use for checking
            # if a component inside, do it this way so edges dont count
            centres = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                centres.append((x + w/2, y + h/2, ann["category_id"]))

            # slide across both dims, stepping by stride
            for y0 in range(0, H - patch_size + 1, stride):
                for x0 in range(0, W - patch_size + 1, stride):
                    x1, y1 = x0 + patch_size, y0 + patch_size
                    label  = 0      # default is zero for background

                    # if a component centre is in there, then
                    for cx, cy, cls_id in centres:
                        if x0 <= cx < x1 and y0 <= cy < y1:
                            label = cls_id
                            break       # to prevent assigning multiple

                    self.patches.append((image[y0:y1, x0:x1], label))

    # other functions a Dataset needs
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, label = self.patches[idx]
        return self.transform(patch), label

def get_data_loaders(classes, batch_size=BATCH_SIZE):
    train_ds = PatchDataset("dataset/train", classes)
    val_ds   = PatchDataset("dataset/valid", classes)
    test_ds  = PatchDataset("dataset/test",  classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=True, num_workers=0)
    return train_loader, val_loader, test_loader

################################################################################
#                               Models :3
################################################################################

class PatchClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PatchClassifier, self).__init__()
        self.name = "PatchClassifier"
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # apparently this parameter is the newer way to do it since
        #   pretrained=True is depracated
        for p in base.parameters():
            p.requires_grad = False     # freeze parameters

        # get rid of the last classificaiton layer of the pretrained bro
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.drop = nn.Dropout(p=0.3)
        self.fc1  = nn.Linear(512, 256)
        self.fc2  = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x) 
        x = F.relu(x) 
        x = self.drop(x) 
        x = self.fc2(x)  
        return x

class BaselineANN(nn.Module):
    def __init__(self, patch_size, num_classes):
        super(BaselineANN, self).__init__()
        self.name = "BaselineANN"
        self.fc1 = nn.Linear(patch_size*patch_size*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

################################################################################
#                                 Training
################################################################################

def train_model(model, train_loader, val_loader, num_classes,
                epochs=15, lr=1e-3, save_path="best_model.pt"):

    device = get_device()       # for hardware accel.
    model.to(device)

    # had an issue with accuracy going really high despite terrible predictions
    # and it was because most patches are background, so it kept guessing that
    class_weights = torch.ones(num_classes, device=device)
    class_weights[0] = 0.2   # need to make guessing background less important
    class_weights[1:]= 3.0   # and guessing everything else more important!
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    # initialize arrays
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            logits = model(patches)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total += labels.size(0)
        scheduler.step()    # updates learning rate for Adam

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():       # dont compute grads for validation
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                logits = model(patches)
                v_loss += criterion(logits, labels).item()
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        t_loss /= len(train_loader)
        v_loss /= len(val_loader)
        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        print(f"Epoch {epoch+1:>2}/{epochs}  "
              f"train loss {t_loss:.4f} acc {t_acc:.3f}  |  "
              f"val loss {v_loss:.4f} acc {v_acc:.3f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f"              saved {save_path}")

    # plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(train_losses, label="Train loss")
    ax1.plot(val_losses,   label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{model.name} - Loss"); ax1.legend()
    ax2.plot(train_accs, label="Train acc")
    ax2.plot(val_accs,   label="Val acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model.name} - Accuracy"); ax2.legend()
    plt.tight_layout()
    fname = f"training_curves_{model.name}.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"Saved {fname}")

    return train_losses, val_losses, train_accs, val_accs


def evaluate(model, loader):
    device = get_device()
    model.eval(); model.to(device)
    correct = total = 0
    with torch.no_grad():
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            preds = model(patches).argmax(1)
            components = labels > 0
            correct += (preds[components] == labels[components]).sum().item()
            total += components.sum().item()
    return correct / total

################################################################################
#     NMS, a little different from other ones bc i only keep one (well, try to)
################################################################################

def merge_nms(boxes, scores, iou_thresh=IOU_THRESH):
    if len(boxes) == 0:
        return [], []

    boxes  = np.array(boxes,  dtype=float)
    scores = np.array(scores, dtype=float)

    # convert from [x,y,w,h] to [x1,y1,x2,y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]      # sort by confidence
    # track used boxes, to not double count
    used = np.zeros(len(boxes), dtype=bool)  

    merged_boxes  = []
    merged_scores = []

    while order.size > 0:
        i = order[0]        # pick highest confidence remaining box
        if used[i]:
            order = order[1:]   # if used, skip
            continue

        # computer overlap between box i and all others in order
        xx1 = np.maximum(x1[i], x1[order])    # by getting intersections
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])
        # get intersection area
        inter = (np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1))
        
        # intersection over union, basically overlap
        iou = inter / (areas[i] + areas[order] - inter + 1e-6)

        # cluster = everything that overlaps box i by >= iou_thresh
        cluster_mask = iou >= iou_thresh    # mask over order
        cluster_idx  = order[cluster_mask]  

        # keep the most confident
        # note to self:maybe look into combining the boxes like spacially too
        best = cluster_idx[scores[cluster_idx].argmax()]
        merged_boxes.append(boxes[best].tolist())
        merged_scores.append(float(scores[best]))

        used[cluster_idx] = True     # mark used clusters as used
        order = order[~cluster_mask] # remove image in cluster from being reused
    return merged_boxes, merged_scores

################################################################################
#                               COUNTING
################################################################################

def predict_counts(model, image_path, classes,
                   patch_size=PATCH_SIZE, stride=STRIDE,
                   conf_thresh=CONF_THRESH, save_vis=None):
    device = get_device()
    model.eval(); model.to(device)
    num_classes = len(classes)

    # same thing as before lol
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])

    image = plt.imread(image_path)
    image = image[..., :3]

    H, W = image.shape[:2]
    raw_boxes = {c: [] for c in range(1, num_classes)}
    raw_scores = {c: [] for c in range(1, num_classes)}

    with torch.no_grad():
        for y0 in range(0, H - patch_size + 1, stride):
            for x0 in range(0, W - patch_size + 1, stride):
                # get a patch
                patch  = image[y0:y0+patch_size, x0:x0+patch_size]
                # preprocess patch and add batch dimension
                tensor = transform(patch).unsqueeze(0).to(device)   
                probs = F.softmax(model(tensor)[0], dim=0)
                cls = probs.argmax().item()
                conf = probs[cls].item()
                # record detections that are not background and are confident
                if cls > 0 and conf >= conf_thresh:
                    raw_boxes[cls].append([x0, y0, patch_size, patch_size])
                    raw_scores[cls].append(conf)

    # initialize counts
    counts = {classes[c]: 0 for c in range(1, num_classes)}
    kept = []
    # for every class (except bg, do merge_nms)
    for cls in range(1, num_classes):
        m_boxes, m_scores = merge_nms(raw_boxes[cls], raw_scores[cls])
        counts[classes[cls]] = len(m_boxes)
        for box, score in zip(m_boxes, m_scores):
            kept.append((box, classes[cls], score))

    # visualization of the bounding boxes
    if save_vis:
        colors = ["#E8593C", "#3B8BD4", "#1D9E75",
                  "#BA7517", "#A32D2D", "#7B4FBF"]  # one color per class
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image)
        for (x0, y0, pw, ph), cls_name, conf in kept:
            ci = classes.index(cls_name) - 1  # color selector, zero indexed
            col = colors[ci % len(colors)]     # look up the actual colour string
            # draw rectangle
            rect = mpatches.Rectangle((x0, y0), pw, ph, linewidth=2, 
                                      edgecolor=col, facecolor="none")
            ax.add_patch(rect)
            # label box
            ax.text(x0, y0 - 4, f"{cls_name} {conf:.2f}", color=col, fontsize=16,
                    bbox=dict(facecolor="white", alpha=0.6, pad=2))
        
        title = "  |  ".join(f"{k}: {v}" for k, v in counts.items())
        ax.set_title(title, fontsize=20)
        ax.axis("off")  # hide ticks
        plt.tight_layout()
        plt.savefig(save_vis, dpi=150); plt.close()
        print("saved pic")
    return counts

################################################################################
#                                  main
################################################################################

if __name__ == "__main__":
    torch.manual_seed(1)
    device = get_device()

    # get annotations
    ann_json = find_annotations_json("dataset/train")
    classes, num_classes = load_category_map(ann_json)
    print(f"\nClasses ({num_classes}): {classes}")

    # build loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        classes, batch_size=BATCH_SIZE)

    # run baseline
    print("\n###### Baseline ANN #######")
    baseline = BaselineANN(patch_size=PATCH_SIZE, num_classes=num_classes)
    train_model(baseline, train_loader, val_loader, num_classes,
                epochs=15, lr=1e-3, save_path="baseline_model.pt")
    baseline.load_state_dict(torch.load("baseline_model.pt", map_location=device))
    baseline_acc = evaluate(baseline, test_loader)
    print(f"Test accuracy: {baseline_acc:.3f}")

    # run primary
    print("\n######## PatchClassifier #######")
    model = PatchClassifier(num_classes=num_classes)
    train_model(model, train_loader, val_loader, num_classes,
                epochs=15, lr=1e-3, save_path="best_model.pt")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model_acc = evaluate(model, test_loader)
    print(f"Test accuracy: {model_acc:.3f}")

    # print results
    print("\n results summary")
    print(f"  BaselineANN      {baseline_acc:.3f}")
    print(f"  PatchClassifier  {model_acc:.3f}")
    print(f"  Improvement      +{model_acc - baseline_acc:.3f}")

    # get a picture to show what it be looking like when it working
    test_images = glob.glob("dataset/test/*.jpg")
    if test_images:
        print(f"\nRunning demo on {test_images[0]}...")
        counts = predict_counts(model, test_images[0], classes,
                                save_vis="demo_prediction.png")
        print("Detected components:")
        for cls_name, n in counts.items():
            if n > 0:
                print(f"  {cls_name:>15}: {n}")
    else:
        print("\nNo test images found for demo.")


# notes to self
# could improve speed by not keeping background patches
# could improve bounding boxes by merging coords of high confidence ones
# i forgor