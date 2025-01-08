import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from A.breast_classification import BreastClassification
from B.blood_classification import BloodClassification

def reset_seeds(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def plot_confusion_matrices(y_true_rf, y_pred_rf,
                            y_true_svm, y_pred_svm,
                            y_true_res, y_pred_res,
                            class_names=None,
                            save_path="A/confusion_matrices.png"):
    """
    Plot confusion matrices of three models (RF, SVM, ResNet) in one figure (1 x 3),
    with "abs value + percentage" in each cell, colored by percentage.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot a single confusion matrix
    def plot_cm_on_ax(ax, y_true, y_pred, title, add_colorbar=False):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Shade according to the percentage
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # Set class names as labels
        if class_names is not None:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right")

        # Add abs value + percentage text in each cell
        n_rows, n_cols = cm.shape
        for i in range(n_rows):
            for j in range(n_cols):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                absolute = cm[i, j]
                perc = 100.0 * cm_norm[i, j]
                text_str = f"{absolute}\n{perc:.1f}%"
                ax.text(j, i, text_str, ha="center", va="center", color=color)

        # Add colorbar
        if add_colorbar:
            fig = ax.get_figure()
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Proportion (%)', rotation=-90, va="bottom")

    plot_cm_on_ax(axes[0], y_true_rf,  y_pred_rf,  title="RandomForest")
    plot_cm_on_ax(axes[1], y_true_svm, y_pred_svm, title="SVM")
    plot_cm_on_ax(axes[2], y_true_res, y_pred_res, title="ResNet-18", add_colorbar=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def breast_methods_tests():
    bc = BreastClassification(download=True)

    # RF search space
    rf_param_grid = {
        'n_estimators': [20, 50, 100],
        'max_depth': [None, 4, 5, 8],
        'min_samples_split': [3, 5, 7],
        'min_samples_leaf': [1, 2, 4],
    }
    rf_model, rf_test_preds, rf_test_acc = bc.train_random_forest(param_grid=rf_param_grid)

    # SVM search space
    svm_param_grid = {
        'C': [3, 5, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm_model, svm_test_preds, svm_test_acc = bc.train_svm(param_grid=svm_param_grid)

    # 3) ResNet-18 search space
    lr_candidates = [1e-3, 1e-4]
    batch_candidates = [32, 64, 128]
    epochs_candidates = [10, 20, 30]
    resnet_model, res_test_preds, res_test_acc = bc.train_resnet(
        lr_candidates=lr_candidates,
        batch_candidates=batch_candidates,
        epochs_candidates=epochs_candidates,
        final_train_plot_path="A/resnet_loss_curve_breast.png"
    )
    print("===== Final Comparison =====")
    print(f"RandomForest Test Acc: {rf_test_acc*100:.2f}%")
    print(f"SVM         Test Acc: {svm_test_acc*100:.2f}%")
    print(f"ResNet-18   Test Acc: {res_test_acc*100:.2f}%")
    class_names = ["malignant", "benign"]

    # Plot confusion matrices
    y_test = bc.y_test  # True labels
    plot_confusion_matrices(
        y_true_rf=y_test,  y_pred_rf=rf_test_preds,
        y_true_svm=y_test, y_pred_svm=svm_test_preds,
        y_true_res=y_test, y_pred_res=res_test_preds,
        class_names=class_names,
        save_path="A/confusion_matrices_breast.png"
    )

def blood_methods_tests():
    bc = BloodClassification(download=True)

    # RF search space
    rf_param_grid = {
        'n_estimators': [300, 400], 
        'max_depth': [None, 15, 20],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
    }
    rf_model, rf_test_preds, rf_test_acc = bc.train_random_forest(param_grid=rf_param_grid)

    # SVM search space
    svm_param_grid = {
        'C': [5, 10, 15],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm_model, svm_test_preds, svm_test_acc = bc.train_svm(param_grid=svm_param_grid)

    # ResNet-18 search space
    lr_candidates = [1e-3, 1e-4]
    batch_candidates = [32, 64, 128]
    epochs_candidates = [20, 30]
    resnet_model, res_test_preds, res_test_acc = bc.train_resnet(
        lr_candidates=lr_candidates,
        batch_candidates=batch_candidates,
        epochs_candidates=epochs_candidates,
        final_train_plot_path="B/resnet_loss_curve_blood.png"
    )

    print("===== Final Comparison =====")
    print(f"RandomForest Test Acc: {rf_test_acc*100:.2f}%")
    print(f"SVM         Test Acc: {svm_test_acc*100:.2f}%")
    print(f"ResNet-18   Test Acc: {res_test_acc*100:.2f}%")
    class_names = [
        'basophil', 'eosinophil', 'erythroblast', 'immature granulocytes',
        'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
    ]

    # Plot confusion matrices
    y_test = bc.y_test  # True labels
    plot_confusion_matrices(
        y_true_rf=y_test,  y_pred_rf=rf_test_preds,
        y_true_svm=y_test, y_pred_svm=svm_test_preds,
        y_true_res=y_test, y_pred_res=res_test_preds,
        class_names=class_names,
        save_path="B/confusion_matrices_blood.png"
    )
    
if __name__ == "__main__":
    reset_seeds()
    breast_methods_tests()
    
    reset_seeds()
    blood_methods_tests()