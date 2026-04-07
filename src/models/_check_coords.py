import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import json

# Load OCR results
with open('../../output/v4/shirobako_res.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
res = data[0] if isinstance(data, list) else data
polys = res.get('dt_polys', [])
texts = res.get('rec_texts', [])

# Centers of OCR lines
ocr_centers = []
for poly in polys:
    pts = np.array(poly)
    ocr_centers.append((pts[:,0].mean(), pts[:,1].mean()))

img = Image.open('../../data/raw/images/shirobako.jpg')
orig_w, orig_h = img.size  # 595, 842

# Load YOLO
model_path = hf_hub_download(repo_id="ogkalu/comic-speech-bubble-detector-yolov8m", filename="comic-speech-bubble-detector.pt")
model = YOLO(model_path)

# Run YOLO on original (non-rotated) image
results = model('../../data/raw/images/shirobako.jpg', verbose=False)
r = results[0]
orig_bubbles = []
for i in range(len(r.boxes)):
    if r.boxes.conf[i] >= 0.5:
        orig_bubbles.append(r.boxes.xyxy[i].tolist())

# Try different transforms to match OCR coordinate space
# PaddleOCR angle=90 -> rotated to 842x595

# Transform A: 90 CCW -> (x,y) -> (y, W-1-x)
def transform_ccw(box):
    x1,y1,x2,y2 = box
    return (y1, orig_w-1-x2, y2, orig_w-1-x1)

# Transform B: 90 CW -> (x,y) -> (H-1-y, x)
def transform_cw(box):
    x1,y1,x2,y2 = box
    return (orig_h-1-y2, x1, orig_h-1-y1, x2)

for name, transform in [("CCW", transform_ccw), ("CW", transform_cw)]:
    bubbles = [transform(b) for b in orig_bubbles]
    matched = 0
    for cx, cy in ocr_centers:
        for bx1, by1, bx2, by2 in bubbles:
            mn_x, mx_x = min(bx1,bx2), max(bx1,bx2)
            mn_y, mx_y = min(by1,by2), max(by1,by2)
            if mn_x <= cx <= mx_x and mn_y <= cy <= mx_y:
                matched += 1
                break
    print(f"{name}: {matched}/{len(polys)} OCR lines matched")
    if matched > 0:
        print(f"  Sample bubbles (transformed):")
        for b in bubbles[:3]:
            print(f"    ({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f})")

# Also try: just run on CCW rotated image
img_ccw = img.transpose(Image.ROTATE_90)
img_ccw.save('../../output/v4/_test_rotated_ccw.jpg')
results_ccw = model('../../output/v4/_test_rotated_ccw.jpg', verbose=False)
r_ccw = results_ccw[0]
bubbles_ccw = []
for i in range(len(r_ccw.boxes)):
    if r_ccw.boxes.conf[i] >= 0.5:
        bubbles_ccw.append(r_ccw.boxes.xyxy[i].tolist())

matched_ccw = 0
for cx, cy in ocr_centers:
    for bx1, by1, bx2, by2 in bubbles_ccw:
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            matched_ccw += 1
            break
print(f"\nYOLO on CCW rotated: {len(bubbles_ccw)} bubbles, {matched_ccw}/{len(polys)} OCR lines matched")
