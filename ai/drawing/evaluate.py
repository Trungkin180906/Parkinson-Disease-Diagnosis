import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
def main():

    model_path  = os.path.join(CURRENT_DIR, 'drawing_model.pkl')
    scaler_path = os.path.join(CURRENT_DIR, 'drawing_scaler.pkl')
    X_path      = os.path.join(CURRENT_DIR, 'X_drawing.npy')
    y_path      = os.path.join(CURRENT_DIR, 'y_drawing.npy')
    for p in [model_path, scaler_path, X_path, y_path]:
        if not os.path.exists(p):
            print(f" Thiếu file: {p}")
            print("   Hãy chạy preprocess.py rồi train.py trước!")
            return
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X      = np.load(X_path)
    y      = np.load(y_path)
    print(f" Tải xong dữ liệu: {len(X)} mẫu")

    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    # Chuẩn hóa dữ liệu test bằng scaler đã train
    X_test_sc = scaler.transform(X_test)

    y_pred       = model.predict(X_test_sc)
    y_pred_proba = model.predict_proba(X_test_sc)[:, 1]  # Xác suất là Parkinson
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 55)
    print(" KẾT QUẢ ĐÁNH GIÁ CHI TIẾT - DRAWING ANALYSIS MODEL")
    print("=" * 55)
    print(f" Accuracy  : {acc * 100:.2f}%")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Khỏe mạnh (0)', 'Parkinson (1)']))
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Drawing Analysis Model — Evaluation Results', fontsize=14, fontweight='bold')
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Khỏe mạnh', 'Parkinson'])
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Confusion Matrix')
  
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc     = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Guess')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')
    plt.tight_layout()
    # Lưu biểu đồ ra file ảnh
    save_path = os.path.join(CURRENT_DIR, 'evaluation_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 AUC Score : {roc_auc:.4f}")
    print(f" Biểu đồ đã lưu: evaluation_result.png")
    plt.show()
if __name__ == "__main__":
    main()
