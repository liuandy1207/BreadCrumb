import torch
import glob
from primary3 import (
    get_device, find_annotations_json, load_category_map,
    BaselineANN, PatchClassifier,
    predict_counts, PATCH_SIZE, NUM_CLASSES
)

device = get_device()

# load class map from training annotations
ann_json = find_annotations_json("dataset/train")
classes, num_classes = load_category_map(ann_json)
print(f"Classes: {classes}")

# pick the first test image
test_images = glob.glob("dataset/test/*.jpg")
if not test_images:
    print("No test images found in dataset/test/")
    exit()

image_path = test_images[0]
print(f"Running on: {image_path}")

baseline = BaselineANN(patch_size=PATCH_SIZE, num_classes=num_classes)
baseline.load_state_dict(torch.load("baseline_model.pt", map_location=device))
baseline.eval()

counts_baseline = predict_counts(baseline, image_path, classes,
                                 save_vis="demo_baseline.png")
print("\nBaselineANN detections:")
for cls_name, n in counts_baseline.items():
    if n > 0:
        print(f"  {cls_name:>15}: {n}")
print("saved demo_baseline.png")

model = PatchClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

counts_model = predict_counts(model, image_path, classes,
                              save_vis="demo_patchclassifier.png")
print("\nPatchClassifier detections:")
for cls_name, n in counts_model.items():
    if n > 0:
        print(f"  {cls_name:>15}: {n}")
print("saved demo_patchclassifier.png")