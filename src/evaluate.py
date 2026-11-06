import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.config import FIGURE_DIR
from src.utils import create_project_dirs


@torch.no_grad()
def evaluate_model(
    model,
    data,
    device,
    mask_name="test",
    num_classes=None
):
    model = model.to(device)
    data = data.to(device)

    model.eval()

    logits = model(data.x, data.edge_index)

    mask = getattr(data, f"{mask_name}_mask")

    y_true = data.y[mask].cpu().numpy()
    y_pred = logits[mask].argmax(dim=1).cpu().numpy()

    if num_classes is None:
        num_classes = int(data.y.max().item()) + 1

    labels = np.arange(num_classes)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        )
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    return metrics, cm


def plot_confusion_matrix(cm, model_name):
    create_project_dirs()

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / f"{model_name.lower()}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)