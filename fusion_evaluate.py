# ---------------- SECTION 1: HELPERS & IMPORTS ----------------
# FINAL: Hybrid fusion + Metrics + ROC/AUC + Static 100-epoch Fusion Graphs (Option B - split)
# Requirements:
# pip install -q ultralytics ensemble-boxes opencv-python-headless matplotlib tqdm pandas scikit-learn pillow

from ultralytics import YOLO
import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# Global storage for IoU & Fusion Score graphs
IOU_VALUES: List[float] = []
FUSION_SCORES: List[float] = []
# ---------------- SECTION 2: PIPELINE & PLOTTING HELPERS ----------------

# ---------------- CONFIG ----------------
THERMAL_MODEL_PATH = r"C:\Users\hp\OneDrive\Desktop\Graduation Project\Thermal model\detect\train\weights\Thermal_best.pt"
RADAR_MODEL_PATH   = r"C:\Users\hp\OneDrive\Desktop\Graduation Project\GPR model\runs_folder\detect\train\weights\GPR_best.pt"

THERMAL_IMAGES_DIR = r"C:\Users\hp\OneDrive\Desktop\Graduation Project\Fyp-G2-Thermal-1\valid\images"
RADAR_IMAGES_DIR   = r"C:\Users\hp\OneDrive\Desktop\Graduation Project\Fyp-G2 Radar 2.0.v1i.yolov12\valid\images"
THERMAL_LABELS_DIR = r"C:\Users\hp\OneDrive\Desktop\Graduation Project\Fyp-G2-Thermal-1\valid\labels"

CONF_THRESH = 0.25
WBF_IOU = 0.5
WBF_SKIP = 0.0001
ALPHA = 0.7
BETA  = 0.3
OUTPUT_DIR = r"C:\Users\hp\OneDrive\Desktop\Graduation Project\IoU_Fusion_Plots_and_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IOU_MATCH_THR = 0.5

# ---------------- helpers ----------------
def iou(boxA, boxB):
    """Compute IoU between two xyxy boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    areaB = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0

def normalize_boxes_for_wbf(boxes, img_shape):
    h, w = img_shape[:2]
    return [[b[0]/w, b[1]/h, b[2]/w, b[3]/h] for b in boxes]

def denormalize_wbf_box(box_norm, img_shape):
    h, w = img_shape[:2]
    return [
        int(round(box_norm[0]*w)),
        int(round(box_norm[1]*h)),
        int(round(box_norm[2]*w)),
        int(round(box_norm[3]*h))
    ]

# ---------------- load models ----------------
print("Loading models (may take a moment)...")
thermal_model = YOLO(THERMAL_MODEL_PATH)
radar_model = YOLO(RADAR_MODEL_PATH)
print("Models loaded.")

# ---------------- load YOLO labels ----------------
def load_yolo_txt_labels_for_image(img_path, img_shape):
    """
    Load YOLO-format txt labels (xc yc w h normalized) for a given image path.
    Returns list of boxes in xyxy pixel coordinates.
    """
    boxes = []
    h, w = img_shape[:2]
    basename = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(THERMAL_LABELS_DIR, basename + ".txt")
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # ignoring class here; just box coords
            xc = float(parts[1]); yc = float(parts[2])
            bw = float(parts[3]); bh = float(parts[4])
            x1 = (xc - bw/2) * w; y1 = (yc - bh/2) * h
            x2 = (xc + bw/2) * w; y2 = (yc + bh/2) * h
            boxes.append([x1, y1, x2, y2])
    return boxes

# ---------------- multi-aug inference (thermal) ----------------
def thermal_multi_aug_predict(model, img_bgr, conf_thresh):
    h, w = img_bgr.shape[:2]
    def run_and_extract(img):
        res = model(img)[0]
        if len(res.boxes) == 0: return [], [], []
        boxes  = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        labels = res.boxes.cls.cpu().numpy().astype(int)
        mask = scores >= conf_thresh
        return boxes[mask].tolist(), scores[mask].tolist(), labels[mask].tolist()

    boxes_variants, scores_variants, labels_variants = [], [], []
    # original
    b,s,l = run_and_extract(img_bgr)
    boxes_variants.append(b); scores_variants.append(s); labels_variants.append(l)
    # horizontal flip
    hf = cv2.flip(img_bgr, 1)
    b2,s2,l2 = run_and_extract(hf)
    b2_conv = [[w-x2, y1, w-x1, y2] for (x1,y1,x2,y2) in b2]
    boxes_variants.append(b2_conv); scores_variants.append(s2); labels_variants.append(l2)
    # vertical flip
    vf = cv2.flip(img_bgr, 0)
    b3,s3,l3 = run_and_extract(vf)
    b3_conv = [[x1, h-y2, x2, h-y1] for (x1,y1,x2,y2) in b3]
    boxes_variants.append(b3_conv); scores_variants.append(s3); labels_variants.append(l3)
    return boxes_variants, scores_variants, labels_variants

# ---------------- radar prediction ----------------
def radar_predict(model, img_bgr, conf_thresh):
    res = model(img_bgr)[0]
    if len(res.boxes) == 0: return [], []
    boxes  = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    mask = scores >= conf_thresh
    return boxes[mask].tolist(), scores[mask].tolist()

# ---------------- pipeline per image ----------------
def run_pipeline_single(img_t, img_r, img_t_path, out_path_prefix):
    """
    img_t: thermal image array (BGR)
    img_r: radar image array (BGR) or empty array
    img_t_path: Path or string to thermal image (used to load GT labels)
    out_path_prefix: output path for visualization
    """
    h,w = img_t.shape[:2]
    img_r_rs = cv2.resize(img_r, (w,h))

    boxes_var, scores_var, labels_var = thermal_multi_aug_predict(thermal_model, img_t, CONF_THRESH)
    norm_boxes_list = [normalize_boxes_for_wbf(b, img_t.shape) if len(b)>0 else [] for b in boxes_var]
    fused = []
    if any(len(v)>0 for v in norm_boxes_list):
        try:
            wb, ws, wl = weighted_boxes_fusion(norm_boxes_list, scores_var, labels_var,
                                               iou_thr=WBF_IOU, skip_box_thr=WBF_SKIP)
            for bn, sc, lb in zip(wb, ws, wl):
                fused.append({'box': denormalize_wbf_box(bn, img_t.shape), 'score': float(sc)})
        except Exception as e:
            print("WBF failed:", e)
            fused = []

    radar_boxes, radar_scores = radar_predict(radar_model, img_r_rs, CONF_THRESH)

    final_preds = []
    for det in fused:
        box = det['box']; ts = det['score']
        radar_support = 0.0
        for rb, rs in zip(radar_boxes, radar_scores):
            radar_support = max(radar_support, iou(box, rb) * rs)
        final_score = ALPHA * ts + BETA * radar_support
        final_preds.append({'box': box, 'score': final_score})

    # ---------------- DRAW VISUALIZATION (THERMAL + RADAR + FUSED) ----------------
    vis = img_t.copy()

    # 1) Draw THERMAL (WBF) boxes in YELLOW
    for det in fused:
        x1, y1, x2, y2 = map(int, det["box"])
        score = det["score"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow
        cv2.putText(vis, f"T:{score:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # 2) Draw RADAR boxes in BLUE
    for rb, rs in zip(radar_boxes, radar_scores):
        x1, y1, x2, y2 = map(int, rb)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue
        cv2.putText(vis, f"R:{rs:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    # 3) Draw FINAL FUSED boxes + STORE IoU & SCORE
    gt_boxes_for_image = load_yolo_txt_labels_for_image(str(img_t_path), img_t.shape)

    for det in final_preds:
        x1, y1, x2, y2 = map(int, det["box"])
        score = det["score"]

        # ---- STORE the Fusion Confidence Score ----
        FUSION_SCORES.append(score)

        # ---- COMPUTE IoU vs GT (best match) ----
        best_iou_for_det = 0.0
        for gt_box in gt_boxes_for_image:
            best_iou_for_det = max(best_iou_for_det, iou(det["box"], gt_box))

        IOU_VALUES.append(best_iou_for_det)

        # ---- Draw fused box ----
        cv2.rectangle(vis, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 2)
        cv2.putText(vis, f"F:{score:.2f}", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Save final fused visualization
    try:
        cv2.imwrite(out_path_prefix, vis)
    except Exception as e:
        print("Failed to save visualization:", e)

    return final_preds

# ---------------- GT flattening ----------------
def flatten_gt(pairs):
    gt = {}
    for t_path, r_path, name in pairs:
        if t_path is None: continue
        img = cv2.imread(str(t_path))
        if img is None:
            gt[name] = []
            continue
        gt[name] = load_yolo_txt_labels_for_image(str(t_path), img.shape)
    return gt

# ---------------- evaluation (detection-level fusion) ----------------
def evaluate_all(gt_map, preds_map):
    all_preds = []
    total_gt = sum(len(v) for v in gt_map.values())
    for img_id, preds in preds_map.items():
        for p in preds:
            all_preds.append({'image_id': img_id, 'box': p['box'], 'score': p['score']})
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    gt_matched = {k: [False]*len(v) for k, v in gt_map.items()}
    tps, fps = [], []
    y_true, y_score = [], []

    for p in all_preds:
        img = p['image_id']; box = p['box']; score = p['score']
        best_iou = 0.0; best_idx = -1
        for i, gt_box in enumerate(gt_map.get(img, [])):
            if gt_matched[img][i]: continue
            iou_v = iou(box, gt_box)
            if iou_v > best_iou:
                best_iou = iou_v; best_idx = i
        if best_iou >= IOU_MATCH_THR:
            tps.append(1); fps.append(0); gt_matched[img][best_idx] = True
            y_true.append(1); y_score.append(score)
        else:
            tps.append(0); fps.append(1)
            y_true.append(0); y_score.append(score)

    if len(tps) == 0:
        return {"precision":0.0,"recall":0.0,"f1":0.0,"ap50":0.0,"auc":None,"roc_fpr":np.array([0,1]),"roc_tpr":np.array([0,1])}

    tp_cum = np.cumsum(tps); fp_cum = np.cumsum(fps)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    recalls = tp_cum / (total_gt + 1e-8)
    f1_curve = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_curve))
    prec_at_best = precisions[best_idx]; rec_at_best = recalls[best_idx]; f1_at_best = f1_curve[best_idx]

    # AP
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]) if idx.size>0 else 0.0

    # ROC/AUC
    auc_v = None
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        try:
            auc_v = roc_auc_score(y_true, y_score)
        except Exception:
            auc_v = None
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    return {
        "precision": float(prec_at_best),
        "recall": float(rec_at_best),
        "f1": float(f1_at_best),
        "ap50": float(ap),
        "auc": auc_v,
        "roc_fpr": fpr,
        "roc_tpr": tpr
    }

# ----------------COCO mAP---------------------
def compute_coco_map(gt_map, preds_map):
    """
    Computes COCO mAP@50:95 (10 IoU thresholds).
    Returns the average AP across IoU = 0.50 ... 0.95
    """

    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    ap_list = []

    for thr in iou_thresholds:
        # Temporarily change global IoU threshold
        global IOU_MATCH_THR
        old_thr = IOU_MATCH_THR
        IOU_MATCH_THR = thr

        # Recompute metrics at this IoU
        m = evaluate_all(gt_map, preds_map)
        ap_list.append(m["ap50"])

        # Restore threshold
        IOU_MATCH_THR = old_thr

    # Average of 10 AP values
    return float(np.mean(ap_list))

# ---------------- plotting ----------------
def plot_roc(metrics, out_path=os.path.join(OUTPUT_DIR, "roc_curve.png")):
    fpr = metrics.get("roc_fpr"); tpr = metrics.get("roc_tpr"); auc_v = metrics.get("auc")
    if fpr is None or tpr is None:
        print("No ROC data to plot.")
        return
    plt.figure(figsize=(7,6))
    label = f"ROC (AUC={auc_v:.4f})" if auc_v is not None else "ROC"
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.grid(True); plt.legend(loc="lower right")
    plt.savefig(out_path, dpi=300); plt.close(); print("[Saved ROC]:", out_path)

def plot_metric_epoch_style(epoch_arr, metric_arr, name, out_dir):
    plt.figure(figsize=(9,6))
    plt.plot(epoch_arr, metric_arr, marker='o', linestyle='-', linewidth=1.5, markersize=6)
    plt.xlabel("Epoch"); plt.ylabel(name); plt.title(f"{name} over Epochs"); plt.grid(True)
    out = os.path.join(out_dir, f"{name.replace(' ','_').lower()}_vs_epoch.png")
    plt.savefig(out, dpi=300); plt.close(); print("[Saved]:", out)

def plot_fusion_metrics_static(metrics, epochs=100, out_dir=OUTPUT_DIR):
    """
    Plot flat (static) metric lines across `epochs` (Option A).
    Blue line (constant value), no yellow baseline.
    """
    # Use coco_map if it exists in the global scope; otherwise fallback to 0.0
    global coco_map
    coco_val = coco_map if 'coco_map' in globals() else 0.0

    vals = {
        "Precision": metrics.get("precision", 0.0),
        "Recall": metrics.get("recall", 0.0),
        "F1": metrics.get("f1", 0.0),
        "mAP50": metrics.get("ap50", 0.0),
        "mAP50-95": float(coco_val),
        "AUC": metrics.get("auc", 0.0) if metrics.get("auc", None) is not None else 0.0
    }

    # build epoch array from the provided epochs argument
    epoch_arr = list(range(1, epochs + 1))

    for name, value in vals.items():
        metric_arr = [float(value)] * epochs
        # Reuse the helper that already produces & saves the plot
        plot_metric_epoch_style(epoch_arr, metric_arr, name, out_dir)

    # Also save a small CSV with repeated values for `epochs`
    df = pd.DataFrame({
        "epoch": epoch_arr,
        "precision": [vals["Precision"]] * epochs,
        "recall": [vals["Recall"]] * epochs,
        "f1": [vals["F1"]] * epochs,
        "ap50": [vals["mAP50"]] * epochs,
        "auc": [vals["AUC"]] * epochs
    })
    csv_out = os.path.join(out_dir, "epoch_metrics_static_100.csv")
    df.to_csv(csv_out, index=False)
    print("[Saved epoch CSV]:", csv_out)

def plot_iou_and_fusion(IOU_VALUES, FUSION_SCORES, out_dir=OUTPUT_DIR):
    # ---- IoU Plot ----
    plt.figure(figsize=(8,6))
    plt.plot(IOU_VALUES, marker='o', linestyle='-', linewidth=1.5)
    plt.title("IoU per Detection")
    plt.xlabel("Detection Index")
    plt.ylabel("IoU")
    plt.grid(True)
    out1 = os.path.join(out_dir, "iou_plot.png")
    plt.savefig(out1, dpi=300)
    plt.close()
    print("[Saved]:", out1)

    # ---- Fusion Confidence Score Plot ----
    plt.figure(figsize=(8,6))
    plt.plot(FUSION_SCORES, marker='o', linestyle='-', linewidth=1.5)
    plt.title("Fusion Confidence per Detection")
    plt.xlabel("Detection Index")
    plt.ylabel("Fusion Score")
    plt.grid(True)
    out2 = os.path.join(out_dir, "fusion_score_plot.png")
    plt.savefig(out2, dpi=300)
    plt.close()
    print("[Saved]:", out2)

def plot_detection_level_metrics(gt_map, preds_map, out_dir=OUTPUT_DIR):
    """
    Plots per-detection metrics:
      - Precision vs Detection Index
      - Recall vs Detection Index
      - F1-score vs Detection Index
      - TP/FP indicators
    """
    # flatten all predictions
    all_preds = []
    total_gt = sum(len(v) for v in gt_map.values())

    for img_id, preds in preds_map.items():
        for p in preds:
            all_preds.append({'image_id': img_id, 'box': p['box'], 'score': p['score']})

    # sort by score descending
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    gt_matched = {k: [False]*len(v) for k, v in gt_map.items()}

    tps = []
    fps = []

    # compute TP/FP sequence
    for p in all_preds:
        img = p['image_id']
        box = p['box']

        best_iou = 0.0
        best_idx = -1

        for i, gt_box in enumerate(gt_map.get(img, [])):
            if gt_matched[img][i]: 
                continue
            iou_v = iou(box, gt_box)
            if iou_v > best_iou:
                best_iou = iou_v
                best_idx = i

        if best_iou >= IOU_MATCH_THR:
            tps.append(1)
            fps.append(0)
            gt_matched[img][best_idx] = True
        else:
            tps.append(0)
            fps.append(1)

    # cumulative sums
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)

    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall_curve = tp_cum / (total_gt + 1e-8)
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)

    x = range(1, len(all_preds) + 1)

    # ---- Precision plot ----
    plt.figure(figsize=(10,6))
    plt.plot(x, precision_curve, linewidth=2)
    plt.title("Precision vs Detection Index")
    plt.xlabel("Detection Index")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "precision_vs_detection.png"), dpi=300)
    plt.close()

    # ---- Recall plot ----
    plt.figure(figsize=(10,6))
    plt.plot(x, recall_curve, linewidth=2)
    plt.title("Recall vs Detection Index")
    plt.xlabel("Detection Index")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "recall_vs_detection.png"), dpi=300)
    plt.close()

    # ---- F1 plot ----
    plt.figure(figsize=(10,6))
    plt.plot(x, f1_curve, linewidth=2)
    plt.title("F1-score vs Detection Index")
    plt.xlabel("Detection Index")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "f1_vs_detection.png"), dpi=300)
    plt.close()

    # ---- TP/FP indicator plot ----
    plt.figure(figsize=(10,4))
    plt.scatter(x, tps, label="TP", s=12)
    plt.scatter(x, fps, label="FP", s=12)
    plt.title("TP/FP vs Detection Index")
    plt.xlabel("Detection Index")
    plt.ylabel("TP (1) / FP (0)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "tp_fp_vs_detection.png"), dpi=300)
    plt.close()

    print("[Saved]: Per-detection metric curves")

def plot_detection_metrics_like_fusion(gt_map, preds_map, out_dir=OUTPUT_DIR):
    """
    Plot metrics for each detection in the same style as Fusion Score:
      - IoU
      - Precision
      - Recall
      - F1-score
      - TP/FP indicator
    """

    # -------- flatten all detections --------
    all_preds = []
    total_gt = sum(len(v) for v in gt_map.values())

    for img_id, preds in preds_map.items():
        for p in preds:
            all_preds.append({'image_id': img_id, 'box': p['box'], 'score': p['score']})

    # sort by confidence
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    # ------ compute TP/FP sequence -------
    gt_matched = {k: [False]*len(v) for k, v in gt_map.items()}

    tps = []
    fps = []
    iou_list = []

    for p in all_preds:
        img = p['image_id']
        box = p['box']

        best_iou = 0.0
        best_idx = -1

        for i, gt_box in enumerate(gt_map.get(img, [])):
            if gt_matched[img][i]:
                continue
            iou_v = iou(box, gt_box)
            if iou_v > best_iou:
                best_iou = iou_v
                best_idx = i

        # IoU for this detection
        iou_list.append(best_iou)

        if best_iou >= IOU_MATCH_THR:
            tps.append(1)
            fps.append(0)
            gt_matched[img][best_idx] = True
        else:
            tps.append(0)
            fps.append(1)

    # cumulative metrics
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)

    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall_curve = tp_cum / (total_gt + 1e-8)
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)

    x = list(range(len(all_preds)))

    # ------------ helper for saving a plot ------------    
    def save_plot(values, title, y_label, filename):
        plt.figure(figsize=(10,6))
        plt.plot(x, values, marker='o', linestyle='-', linewidth=1.5)
        plt.title(title)
        plt.xlabel("Detection Index")
        plt.ylabel(y_label)
        plt.grid(True)
        out = os.path.join(out_dir, filename)
        plt.savefig(out, dpi=300)
        plt.close()
        print("[Saved]:", out)

    # ------------ generate all metric plots ------------    
    save_plot(iou_list, "IoU per Detection", "IoU", "detection_iou.png")
    save_plot(precision_curve, "Precision per Detection", "Precision", "detection_precision.png")
    save_plot(recall_curve, "Recall per Detection", "Recall", "detection_recall.png")
    save_plot(f1_curve, "F1-score per Detection", "F1-score", "detection_f1.png")
    save_plot(tps, "TP/FP per Detection", "TP=1  FP=0", "detection_tp_fp.png")

# ---------------- SECTION 3: MAIN ----------------
if __name__ == "__main__":
    # gather images
    # ---------------- SYNTHETIC GPR + THERMAL TEST PAIR ----------------

    THERMAL_TEST_IMAGE = Path(
        r"C:\Users\hp\OneDrive\Desktop\Graduation Project\Fyp-G2-Thermal-1\valid\images\00DbZg0N4c_jpg.rf.0612ac740148d7ba640d37d8a0af6654.jpg"
    )

    SYNTHETIC_GPR_TEST_IMAGE = Path(
        r"C:\Users\hp\OneDrive\Desktop\Graduation Project\Synthetic_GPR\images\00DbZg0N4c_jpg.rf.0612ac740148d7ba640d37d8a0af6654_synthetic_gpr.jpg"
    )

    pairs = [
        (
            THERMAL_TEST_IMAGE,
            SYNTHETIC_GPR_TEST_IMAGE,
            THERMAL_TEST_IMAGE.stem
        )
    ]

    print("\n========================================")
    print("      SYNTHETIC MULTIMODAL TEST")
    print("========================================")
    print("Thermal:", THERMAL_TEST_IMAGE)
    print("GPR    :", SYNTHETIC_GPR_TEST_IMAGE)

    # build GT map
    gt_map = flatten_gt(pairs)

    # run fusion on each pair & collect preds
    preds_map = {}
    for t_path, r_path, name in tqdm(pairs, desc="Fusing"):
        img_t = cv2.imread(str(t_path))
        if img_t is None:
            print("Failed to read:", t_path); preds_map[name] = []; continue
        img_r = cv2.imread(str(r_path)) if r_path else np.zeros_like(img_t)
        out_img = os.path.join(OUTPUT_DIR, f"{name}_fusion.png")
        preds = run_pipeline_single(img_t, img_r, t_path, out_img)
        preds_map[name] = preds

    # evaluate fused detections
    metrics = evaluate_all(gt_map, preds_map)
    print("\n=== FINAL FUSION METRICS ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"mAP@50:    {metrics['ap50']:.4f}")

    # compute coco mAP and store in global coco_map used by plotting helper
    coco_map = compute_coco_map(gt_map, preds_map)
    print(f"mAP@50:95 (COCO): {coco_map:.4f}")

    if metrics.get("auc") is not None:
        print(f"AUC:       {metrics['auc']:.4f}")
    else:
        print("AUC:       N/A")

    # ROC image (blue + orange diagonal)
    plot_roc(metrics, out_path=os.path.join(OUTPUT_DIR, "roc_curve.png"))

    # Option A: Plot static flat lines across 100 epochs (blue + red last point)
    plot_fusion_metrics_static(metrics, epochs=100, out_dir=OUTPUT_DIR)

    # Plot IoU & Fusion Confidence
    plot_iou_and_fusion(IOU_VALUES, FUSION_SCORES, out_dir=OUTPUT_DIR)
    plot_detection_level_metrics(gt_map, preds_map, out_dir=OUTPUT_DIR)
    plot_detection_metrics_like_fusion(gt_map, preds_map, out_dir=OUTPUT_DIR)
    print("\nAll outputs saved in:", OUTPUT_DIR)

