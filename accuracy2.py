################################################################################
# new test set eval
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

################################################################################
# hyperparameters
################################################################################

PATCH_SIZE = 48
STRIDE = 16
CONF_THRESH = 0.99
IOU_THRESH = 0.0001

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

################################################################################
# models
################################################################################

class PatchClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PatchClassifier, self).__init__()
        self.name = "PatchClassifier"
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in base.parameters():
            p.requires_grad = False
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

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
# load annotations
################################################################################

def load_category_map(json_path):
    with open(json_path) as f:
        coco = json.load(f)
    cats = [c for c in coco["categories"] if c["supercategory"] != "none"]
    cats.sort(key=lambda c: c["id"])
    max_id = max(c["id"] for c in cats)
    classes = ["background"] + [""] * max_id
    for c in cats:
        classes[c["id"]] = c["name"]
    return classes, len(classes)

def load_ground_truth_split(split_path):
    ann_json = os.path.join(split_path, "_annotations.coco.json")
    with open(ann_json) as f:
        coco = json.load(f)
    
    old_gt = {}
    new_gt = {}
    
    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_name = img_info["file_name"]
        counts = defaultdict(int)
        for ann in coco["annotations"]:
            if ann["image_id"] == img_id:
                counts[ann["category_id"]] += 1
        
        # Check if filename starts with IMG_302 or IMG_303
        if img_name.startswith("IMG_302") or img_name.startswith("IMG_303"):
            new_gt[img_name] = counts
        else:
            old_gt[img_name] = counts
    
    return old_gt, new_gt

################################################################################
# NMS
################################################################################

def merge_nms(boxes, scores, iou_thresh=IOU_THRESH):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    used = np.zeros(len(boxes), dtype=bool)
    merged_boxes = []
    merged_scores = []
    while order.size > 0:
        i = order[0]
        if used[i]:
            order = order[1:]
            continue
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order] - inter + 1e-6)
        cluster_mask = iou >= iou_thresh
        cluster_idx = order[cluster_mask]
        best = cluster_idx[scores[cluster_idx].argmax()]
        merged_boxes.append(boxes[best].tolist())
        merged_scores.append(float(scores[best]))
        used[cluster_idx] = True
        order = order[~cluster_mask]
    return merged_boxes, merged_scores

################################################################################
# prediction
################################################################################

def predict_counts(model, image_path, classes, patch_size=PATCH_SIZE,
                   stride=STRIDE, conf_thresh=CONF_THRESH):
    device = get_device()
    model.eval()
    model.to(device)
    num_classes = len(classes)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image = plt.imread(image_path)
    image = image[..., :3]
    H, W = image.shape[:2]
    raw_boxes = {c: [] for c in range(1, num_classes)}
    raw_scores = {c: [] for c in range(1, num_classes)}
    
    with torch.no_grad():
        for y0 in range(0, H - patch_size + 1, stride):
            for x0 in range(0, W - patch_size + 1, stride):
                patch = image[y0:y0+patch_size, x0:x0+patch_size]
                tensor = transform(patch).unsqueeze(0).to(device)
                probs = F.softmax(model(tensor)[0], dim=0)
                cls = probs.argmax().item()
                conf = probs[cls].item()
                if cls > 0 and conf >= conf_thresh:
                    raw_boxes[cls].append([x0, y0, patch_size, patch_size])
                    raw_scores[cls].append(conf)
    
    counts = {c: 0 for c in range(1, num_classes)}
    for cls in range(1, num_classes):
        m_boxes, m_scores = merge_nms(raw_boxes[cls], raw_scores[cls])
        counts[cls] = len(m_boxes)
    
    return counts

################################################################################
# evaluation
################################################################################

def evaluate_model_split(model, split_path, classes, gt_dict, split_name):
    num_classes = len(classes)
    
    per_class_errors = defaultdict(list)
    total_gt = defaultdict(int)
    total_pred = defaultdict(int)
    
    for img_name, gt_count_dict in gt_dict.items():
        img_path = os.path.join(split_path, img_name)
        if not os.path.exists(img_path):
            continue
        
        pred_counts = predict_counts(model, img_path, classes)
        
        for cls in range(1, num_classes):
            gt = gt_count_dict.get(cls, 0)
            pred = pred_counts.get(cls, 0)
            error = abs(pred - gt)
            per_class_errors[classes[cls]].append(error)
            total_gt[classes[cls]] += gt
            total_pred[classes[cls]] += pred
    
    class_mae = {}
    for cls_name in classes[1:]:
        if cls_name and per_class_errors[cls_name]:
            class_mae[cls_name] = np.mean(per_class_errors[cls_name])
    
    return class_mae, total_gt, total_pred

################################################################################
# plotting
################################################################################

def plot_test_set(baseline_results, primary_results, classes, set_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    class_names = [c for c in classes[1:] if c]
    
    # per-class MAE
    baseline_mae = [baseline_results['mae'].get(c, 0) for c in class_names]
    primary_mae = [primary_results['mae'].get(c, 0) for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    ax1.bar(x - width/2, baseline_mae, width, label='Baseline', color='#E8593C')
    ax1.bar(x + width/2, primary_mae, width, label='Primary', color='#3B8BD4')
    ax1.set_ylabel('MAE')
    ax1.set_title(f'Per-Class MAE - {set_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    
    # total counts
    gt_counts = [baseline_results['gt'].get(c, 0) for c in class_names]
    baseline_pred = [baseline_results['pred'].get(c, 0) for c in class_names]
    primary_pred = [primary_results['pred'].get(c, 0) for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    ax2.bar(x - width, gt_counts, width, label='Ground Truth', color='#1D9E75')
    ax2.bar(x, baseline_pred, width, label='Baseline', color='#E8593C')
    ax2.bar(x + width, primary_pred, width, label='Primary', color='#3B8BD4')
    ax2.set_ylabel('Total Count')
    ax2.set_title(f'Total Counts - {set_name}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    filename = f'{set_name.lower().replace(" ", "_")}_results.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")

################################################################################
# main
################################################################################

if __name__ == "__main__":
    device = get_device()
    
    ann_json = os.path.join("dataset/train", "_annotations.coco.json")
    classes, num_classes = load_category_map(ann_json)
    print(f"Classes: {classes}")
    
    # load ground truth split by old/new
    old_gt, new_gt = load_ground_truth_split("dataset/test")
    print(f"\nOld test set: {len(old_gt)} images")
    print(f"New test set: {len(new_gt)} images")
    
    # load models
    print("\nLoading models...")
    baseline = BaselineANN(patch_size=PATCH_SIZE, num_classes=num_classes)
    baseline.load_state_dict(torch.load("baseline_model.pt", map_location=device))
    
    primary = PatchClassifier(num_classes=num_classes)
    primary.load_state_dict(torch.load("best_model.pt", map_location=device))
    
    # evaluate baseline
    print("\n" + "="*60)
    print("BASELINE MODEL")
    print("="*60)
    
    print("\nEvaluating on original test set...")
    b_old_mae, b_old_gt, b_old_pred = evaluate_model_split(
        baseline, "dataset/test", classes, old_gt, "old")
    
    print("Evaluating on new test set...")
    b_new_mae, b_new_gt, b_new_pred = evaluate_model_split(
        baseline, "dataset/test", classes, new_gt, "new")
    
    print("\nOriginal Test Set:")
    print(f"{'Class':<15} {'MAE':<8} {'GT Total':<10} {'Pred Total':<10}")
    print("-"*45)
    for cls in classes[1:]:
        if cls:
            print(f"{cls:<15} {b_old_mae.get(cls, 0):<8.2f} "
                  f"{b_old_gt.get(cls, 0):<10} {b_old_pred.get(cls, 0):<10}")
    
    print("\nNew Test Set:")
    print(f"{'Class':<15} {'MAE':<8} {'GT Total':<10} {'Pred Total':<10}")
    print("-"*45)
    for cls in classes[1:]:
        if cls:
            print(f"{cls:<15} {b_new_mae.get(cls, 0):<8.2f} "
                  f"{b_new_gt.get(cls, 0):<10} {b_new_pred.get(cls, 0):<10}")
    
    # evaluate primary
    print("\n" + "="*60)
    print("PRIMARY MODEL")
    print("="*60)
    
    print("\nEvaluating on original test set...")
    p_old_mae, p_old_gt, p_old_pred = evaluate_model_split(
        primary, "dataset/test", classes, old_gt, "old")
    
    print("Evaluating on new test set...")
    p_new_mae, p_new_gt, p_new_pred = evaluate_model_split(
        primary, "dataset/test", classes, new_gt, "new")
    
    print("\nOriginal Test Set:")
    print(f"{'Class':<15} {'MAE':<8} {'GT Total':<10} {'Pred Total':<10}")
    print("-"*45)
    for cls in classes[1:]:
        if cls:
            print(f"{cls:<15} {p_old_mae.get(cls, 0):<8.2f} "
                  f"{p_old_gt.get(cls, 0):<10} {p_old_pred.get(cls, 0):<10}")
    
    print("\nNew Test Set:")
    print(f"{'Class':<15} {'MAE':<8} {'GT Total':<10} {'Pred Total':<10}")
    print("-"*45)
    for cls in classes[1:]:
        if cls:
            print(f"{cls:<15} {p_new_mae.get(cls, 0):<8.2f} "
                  f"{p_new_gt.get(cls, 0):<10} {p_new_pred.get(cls, 0):<10}")
    
    # plot comparisons
    baseline_old = {'mae': b_old_mae, 'gt': b_old_gt, 'pred': b_old_pred}
    baseline_new = {'mae': b_new_mae, 'gt': b_new_gt, 'pred': b_new_pred}
    primary_old = {'mae': p_old_mae, 'gt': p_old_gt, 'pred': p_old_pred}
    primary_new = {'mae': p_new_mae, 'gt': p_new_gt, 'pred': p_new_pred}
    
    plot_test_set(baseline_old, primary_old, classes, "Original Test Set")
    plot_test_set(baseline_new, primary_new, classes, "New Test Set")
    
    print("\nDone!")