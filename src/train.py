import pandas as pd
import torch
import torch.nn as nn

from src.config import CHECKPOINT_DIR, METRICS_DIR
from src.utils import (
    create_project_dirs,
    plot_training_history
)


def train_model(
    model,
    data,
    model_name,
    device,
    epochs=300,
    learning_rate=0.01,
    weight_decay=5e-4,
    patience=50
):
    create_project_dirs()

    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    history = []

    checkpoint_path = CHECKPOINT_DIR / f"{model_name.lower()}.pth"

    for epoch in range(1, epochs + 1):
        model.train()

        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)

        train_loss = criterion(
            logits[data.train_mask],
            data.y[data.train_mask]
        )

        train_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            logits = model(data.x, data.edge_index)

            val_loss = criterion(
                logits[data.val_mask],
                data.y[data.val_mask]
            )

            train_pred = logits[data.train_mask].argmax(dim=1)
            val_pred = logits[data.val_mask].argmax(dim=1)

            train_acc = (
                train_pred == data.y[data.train_mask]
            ).float().mean().item()

            val_acc = (
                val_pred == data.y[data.val_mask]
            ).float().mean().item()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            torch.save(best_state, checkpoint_path)

        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"{model_name} | "
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"{model_name}: Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    history = pd.DataFrame(history)

    history.to_csv(
        METRICS_DIR / f"{model_name.lower()}_history.csv",
        index=False
    )

    plot_training_history(history, model_name)

    print(f"\n{model_name} training completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return model, history