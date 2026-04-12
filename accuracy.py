################################################################################
# eval tesst set
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def load_ground_truth(split_path):
    ann_json = os.path.join(split_path, "_annotations.coco.json")
    with open(ann_json) as f:
        coco = json.load(f)
    
    gt_counts = {}
    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_name = img_info["file_name"]
        counts = defaultdict(int)
        for ann in coco["annotations"]:
            if ann["image_id"] == img_id:
                counts[ann["category_id"]] += 1
        gt_counts[img_name] = counts
    
    return gt_counts, coco

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

def evaluate_model(model, split_path, classes):
    gt_counts, coco = load_ground_truth(split_path)
    num_classes = len(classes)
    
    per_image_errors = []
    per_class_errors = defaultdict(list)
    total_abs_error = 0
    total_gt_objects = 0
    
    for img_name, gt_count_dict in gt_counts.items():
        img_path = os.path.join(split_path, img_name)
        if not os.path.exists(img_path):
            continue
        
        pred_counts = predict_counts(model, img_path, classes)
        
        img_error = 0
        for cls in range(1, num_classes):
            gt = gt_count_dict.get(cls, 0)
            pred = pred_counts.get(cls, 0)
            error = abs(pred - gt)
            img_error += error
            per_class_errors[classes[cls]].append((gt, pred, error))
            total_abs_error += error
            total_gt_objects += gt
        
        per_image_errors.append(img_error)
    
    mae = np.mean(per_image_errors)
    
    class_metrics = {}
    for cls_name in classes[1:]:
        if cls_name and per_class_errors[cls_name]:
            errors = [e[2] for e in per_class_errors[cls_name]]
            gts = [e[0] for e in per_class_errors[cls_name]]
            preds = [e[1] for e in per_class_errors[cls_name]]
            class_metrics[cls_name] = {
                'mae': np.mean(errors),
                'total_gt': sum(gts),
                'total_pred': sum(preds),
                'abs_diff': abs(sum(preds) - sum(gts))
            }
    
    return {
        'mae': mae,
        'per_image_errors': per_image_errors,
        'per_class_errors': per_class_errors,
        'class_metrics': class_metrics,
        'total_abs_error': total_abs_error,
        'total_gt_objects': total_gt_objects
    }

################################################################################
# plotting
################################################################################

def plot_comparison(baseline_results, primary_results, classes):
    fig = plt.figure(figsize=(14, 10))
    
    # MAE comparison
    ax1 = plt.subplot(2, 2, 1)
    models = ['Baseline', 'Primary']
    maes = [baseline_results['mae'], primary_results['mae']]
    bars = ax1.bar(models, maes, color=['#E8593C', '#3B8BD4'])
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Overall MAE per Image')
    for bar, mae in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mae:.2f}', ha='center', va='bottom', fontsize=10)
    
    # per-class MAE
    ax2 = plt.subplot(2, 2, 2)
    class_names = [c for c in classes[1:] if c]
    baseline_class_mae = [baseline_results['class_metrics'].get(c, {}).get('mae', 0)
                          for c in class_names]
    primary_class_mae = [primary_results['class_metrics'].get(c, {}).get('mae', 0)
                         for c in class_names]
    x = np.arange(len(class_names))
    width = 0.35
    ax2.bar(x - width/2, baseline_class_mae, width, label='Baseline', color='#E8593C')
    ax2.bar(x + width/2, primary_class_mae, width, label='Primary', color='#3B8BD4')
    ax2.set_ylabel('MAE')
    ax2.set_title('Per-Class MAE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    
    # error distribution
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(baseline_results['per_image_errors'], bins=15, alpha=0.6,
             label='Baseline', color='#E8593C')
    ax3.hist(primary_results['per_image_errors'], bins=15, alpha=0.6,
             label='Primary', color='#3B8BD4')
    ax3.set_xlabel('Absolute Error per Image')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    
    # total count accuracy
    ax4 = plt.subplot(2, 2, 4)
    baseline_totals = [(m['total_gt'], m['total_pred'])
                       for m in baseline_results['class_metrics'].values()]
    primary_totals = [(m['total_gt'], m['total_pred'])
                      for m in primary_results['class_metrics'].values()]
    
    class_names_plot = list(baseline_results['class_metrics'].keys())
    gt_counts = [baseline_results['class_metrics'][c]['total_gt']
                 for c in class_names_plot]
    baseline_pred = [baseline_results['class_metrics'][c]['total_pred']
                     for c in class_names_plot]
    primary_pred = [primary_results['class_metrics'][c]['total_pred']
                    for c in class_names_plot]
    
    x = np.arange(len(class_names_plot))
    width = 0.25
    ax4.bar(x - width, gt_counts, width, label='Ground Truth', color='#1D9E75')
    ax4.bar(x, baseline_pred, width, label='Baseline', color='#E8593C')
    ax4.bar(x + width, primary_pred, width, label='Primary', color='#3B8BD4')
    ax4.set_ylabel('Total Count')
    ax4.set_title('Total Counts Across Test Set')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names_plot, rotation=45, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    plt.close()
    print("Saved evaluation_results.png")

################################################################################
# main
################################################################################

if __name__ == "__main__":
    device = get_device()
    
    ann_json = os.path.join("dataset/train", "_annotations.coco.json")
    classes, num_classes = load_category_map(ann_json)
    print(f"Classes: {classes}")
    
    # load models
    print("\nLoading models...")
    baseline = BaselineANN(patch_size=PATCH_SIZE, num_classes=num_classes)
    baseline.load_state_dict(torch.load("baseline_model.pt", map_location=device))
    
    primary = PatchClassifier(num_classes=num_classes)
    primary.load_state_dict(torch.load("best_model.pt", map_location=device))
    
    # evaluate on test set
    print("\nEvaluating baseline model...")
    baseline_results = evaluate_model(baseline, "dataset/test", classes)
    
    print("Evaluating primary model...")
    primary_results = evaluate_model(primary, "dataset/test", classes)
    
    # print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nBaseline MAE: {baseline_results['mae']:.3f}")
    print(f"Primary MAE:  {primary_results['mae']:.3f}")
    print(f"Improvement:  {baseline_results['mae'] - primary_results['mae']:.3f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<15} {'GT Total':<10} {'Baseline':<12} {'Primary':<12} {'B MAE':<8} {'P MAE':<8}")
    print("-"*75)
    for cls_name in classes[1:]:
        if cls_name and cls_name in baseline_results['class_metrics']:
            b = baseline_results['class_metrics'][cls_name]
            p = primary_results['class_metrics'][cls_name]
            print(f"{cls_name:<15} {b['total_gt']:<10} {b['total_pred']:<12} "
                  f"{p['total_pred']:<12} {b['mae']:<8.2f} {p['mae']:<8.2f}")
    
    plot_comparison(baseline_results, primary_results, classes)
    print("\nDone!")