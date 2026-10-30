# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:15:00 2026

Training Script for Pre-binned Pushbroom HSI Data (233 Bands)
using Single-Band Grayscale Preview, GT Mask, and Predicted Classification Map.

@author: TSC
"""
import json
import joblib
import cv2
import numpy as np
import spectral as spy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_coco_masks(json_path: str, height: int, width: int) -> tuple[np.ndarray, dict[int, str]]:
    """Generates a 2D label mask from COCO polygon annotations."""
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    label_mask = np.zeros((height, width), dtype=np.int32)
    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}

    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        for seg in ann['segmentation']:
            pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(label_mask, [pts], color=cat_id)

    return label_mask, category_names


def map_mask_to_bgr(mask_2d: np.ndarray) -> np.ndarray:
    """Maps 2D class ID mask to a 3D BGR color visualization image."""
    height, width = mask_2d.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    class_color_map = {
        0: [0, 0, 0],         # Background (Black)
        1: [0, 0, 255],       # Red
        2: [255, 255, 255],   # White
        3: [255, 0, 0]        # Blue
    }

    for cat_id, color in class_color_map.items():
        color_mask[mask_2d == cat_id] = color

    return color_mask


def visualize_all_maps(
    model: RandomForestClassifier,
    pca: PCA,
    datacube: np.ndarray,
    gt_mask: np.ndarray,
    selected_band: int = 50
) -> None:
    """Executes full-cube prediction and displays Single Band Grayscale, GT Mask, and Classification Map."""
    height, width = datacube.shape[:2]
    num_bands = datacube.shape[2] if datacube.ndim == 3 else 1
    print("\nRunning full-image inference for Grayscale vs GT vs Predicted Map...")

    # 1. ดึงภาพ Grayscale มา 1 Layer แล้ว Normalize เป็น uint8 (0-255)
    band_idx = min(selected_band, num_bands - 1)
    gray_layer = datacube[:, :, band_idx] if datacube.ndim == 3 else np.squeeze(datacube)
    gray_layer_norm = cv2.normalize(gray_layer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # แปลง Grayscale 2D เป็น 3 Channel เพื่อนำไป hstack รวมกับภาพสีอื่นได้
    gray_3ch = cv2.cvtColor(gray_layer_norm, cv2.COLOR_GRAY2BGR)

    # 2. Reshape & Transform PCA (233 -> 15 Components)
    X_full = datacube.reshape(height * width, num_bands)
    X_pca_full = pca.transform(X_full)

    # 3. Predict ทุกพิกเซล
    y_pred_full = model.predict(X_pca_full)
    pred_mask_2d = y_pred_full.reshape(height, width)

    # 4. แปลง Mask เป็นสี BGR
    gt_color = map_mask_to_bgr(gt_mask)
    pred_color = map_mask_to_bgr(pred_mask_2d)

    # 5. รวมภาพ 3 หน้าต่าง Side-by-side (Single Band Gray | GT Mask | Predicted Mask)
    triptych = np.hstack([gray_3ch, gt_color, pred_color])

    cv2.putText(triptych, f"1. Grayscale Layer (Band {band_idx})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(triptych, "2. Ground Truth Mask", (width + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(triptych, "3. Predicted Classification Map", (width * 2 + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Pushbroom HSI Inspection: Grayscale Layer vs GT vs Predicted Map", triptych)
    print("Visualization complete. Press any key on window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_spectral_dataset(
    hdr_path: str,
    json_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """Extracts spectra, labels, datacube, GT mask directly from 233-band dataset."""
    img_obj = spy.open_image(hdr_path)
    datacube = img_obj.load()
    height, width = datacube.shape[:2]
    num_bands = datacube.shape[2] if datacube.ndim == 3 else 1

    label_mask, category_names = load_coco_masks(json_path, height, width)

    X_all = datacube.reshape(-1, num_bands)
    y_all = label_mask.reshape(-1)

    valid_indices = np.where(y_all > 0)[0]
    X = X_all[valid_indices]
    y = y_all[valid_indices]

    return X, y, datacube, label_mask, category_names


def train_rf_from_coco(
    hdr_path: str = 'dataset/dataset.bip.hdr',
    json_path: str = 'dataset/dataset.bip.json',
    model_output: str = 'rf_coco_model.joblib',
    n_components: int = 15
) -> None:
    """Trains PCA + RF model directly on 233-band data and displays maps."""
    print("Extracting 233-band spectral pixels using COCO annotations...")
    X_raw, y, datacube, gt_mask, category_names = extract_spectral_dataset(hdr_path, json_path)

    print(f"Total Annotated Pixels: {X_raw.shape[0]} across {X_raw.shape[1]} Bands")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Fitting PCA (Reducing features from {X_raw.shape[1]} to {n_components} components)...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)

    print("Training High-Speed Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    target_names = [category_names[cid] for cid in sorted(category_names.keys())]
    print("\nClassification Evaluation Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    artifact = {
        'model': model,
        'pca': pca,
        'category_names': category_names
    }
    joblib.dump(artifact, model_output)
    print(f"Model and PCA artifacts saved to '{model_output}'")

    visualize_all_maps(model, pca, datacube, gt_mask, selected_band=50)


if __name__ == '__main__':
    train_rf_from_coco(
        hdr_path='dataset/dataset.bip.hdr',
        json_path='dataset/dataset.bip.json',
        model_output='rf_coco_model.joblib'
    )
