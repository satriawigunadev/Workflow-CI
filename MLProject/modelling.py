
"""
Script: modelling.py
Deskripsi: Baseline Model Training dengan auto-export artefak fisik ke target folder.
"""

import os
import argparse
import shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

def load_preprocessed_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train_scaled.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_scaled.npy"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.flatten()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.flatten()
    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # Deteksi Jalur Data
    DATA_DIR = "default_of_credit_card_clients_preprocessing"
    TARGET_EXPORT_DIR = "saved_model"
    
    if not os.path.exists(DATA_DIR) and os.path.exists("MLProject"):
        DATA_DIR = "MLProject/default_of_credit_card_clients_preprocessing"
        TARGET_EXPORT_DIR = "MLProject/saved_model"

    X_train, X_test, y_train, y_test = load_preprocessed_data(DATA_DIR)
    
    mlflow.set_experiment("Credit_Card_Default_Baseline")
    
    # Aktifkan autologging MLflow untuk mencatat parameter, metrik, dan model secara otomatis
    mlflow.autolog()
    
    with mlflow.start_run(run_name="RF_Base_CI_Execution"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # Tetap mencatat metrik evaluasi test set secara manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Hapus folder ekspor lama jika ada, lalu simpan model segar ke target folder fisik
        if os.path.exists(TARGET_EXPORT_DIR):
            shutil.rmtree(TARGET_EXPORT_DIR)
            
        mlflow.sklearn.save_model(sk_model=model, path=TARGET_EXPORT_DIR)
        print(f"[SUCCESS] Model fisik dan MLmodel sukses diekspor langsung ke: {TARGET_EXPORT_DIR}")

if __name__ == "__main__":
    main()