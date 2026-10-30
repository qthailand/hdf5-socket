# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:55:00 2026

Smart Feature Selection (SelectKBest + Scaler + SGDClassifier) for High Accuracy.

@author: TSC
"""
import json
import joblib
import cv2
import numpy as np
import spectral as spy
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_coco_masks(json_path: str, height: int, width: int) -> tuple[np.ndarray, dict[int, str]]:
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


def train_smart_bands_sgd(num_selected_bands: int = 30):
    hdr_path = 'dataset/dataset.bip.hdr'
    json_path = 'dataset/dataset.bip.json'
    model_output = 'rf_coco_model.joblib'

    print("Loading Pre-binned Datacube...")
    img_obj = spy.open_image(hdr_path)
    datacube = img_obj.load()
    height, width, num_bands = datacube.shape

    label_mask, category_names = load_coco_masks(json_path, height, width)

    X_all = datacube.reshape(-1, num_bands)
    y_all = label_mask.reshape(-1)

    valid_indices = np.where(y_all > 0)[0]
    X_raw_233 = X_all[valid_indices].astype(np.float32)
    y = y_all[valid_indices]

    # Train / Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw_233, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Feature Selection ด้วย SelectKBest (ANOVA F-value) เลือกกระจายแบนด์เด่นจริง
    print(f"\nSelecting Best {num_selected_bands} Discriminative Bands using SelectKBest...")
    selector = SelectKBest(score_func=f_classif, k=num_selected_bands)
    X_train_selected = selector.fit_transform(X_train_raw, y_train)
    X_test_selected = selector.transform(X_test_raw)

    selected_bands = np.where(selector.get_support())[0]
    print(f"Selected Band Indices ({len(selected_bands)} bands): {selected_bands.tolist()}")

    # 2. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # 3. Train SGDClassifier
    print("\nTraining High-Speed SGDClassifier Model...")
    model = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-4,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    target_names = [category_names[cid]
                    for cid in sorted(category_names.keys()) if cid in np.unique(y_test)]

    print("\nClassification Evaluation Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save Artifact (คำนวณ Weights รวม Scaler สำหรับ Direct Matrix Mult)
    # W_scaled = W / scale, b_scaled = b - sum(W * mean / scale)
    W_orig = model.coef_                               # (n_classes, k_bands)
    b_orig = model.intercept_                          # (n_classes,)
    scale = scaler.scale_                              # (k_bands,)
    mean = scaler.mean_                                # (k_bands,)

    W_fused = (W_orig / scale).T                       # (k_bands, n_classes)
    b_fused = b_orig - np.dot(W_orig, (mean / scale))  # (n_classes,)

    artifact = {
        'model': model,
        'W_fused': np.ascontiguousarray(W_fused, dtype=np.float32),
        'b_fused': np.ascontiguousarray(b_fused, dtype=np.float32),
        'selected_bands': selected_bands,
        'category_names': category_names,
        'classes': model.classes_,
        # Lets the simulator feed the model by wavelength and reject a mismatching calibration.
        'band_wavelengths': np.asarray(img_obj.bands.centers, dtype=np.float32),
        'n_bands': int(num_bands),
        'training_header': hdr_path,
    }
    joblib.dump(artifact, model_output)
    print(f"\nSuccessfully trained and saved optimized model artifact to '{model_output}'")


if __name__ == '__main__':
    train_smart_bands_sgd(num_selected_bands=30)
