from __future__ import annotations
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from data_utils import (
    fit_standard_scaler,
    generate_base_dataset,
    generate_circle,
    generate_linear_blobs,
    generate_xor,
    stratified_train_test_split,
    transform_standard_scaler,
)
from metrics_utils import accuracy, f1_score, precision, recall, roc_auc_manual, roc_curve_manual
from perceptron import Perceptron

RESULTS_DIR = "results"


def prepare_base_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = generate_base_dataset()
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=0.3,
        seed=42,
    )

    mean, std = fit_standard_scaler(X_train)
    X_train_scaled = transform_standard_scaler(X_train, mean, std)
    X_test_scaled = transform_standard_scaler(X_test, mean, std)
    return X_train_scaled, X_test_scaled, y_train, y_test


def save_csv(filename: str, rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сохранено: {path}")


def plot_loss(
        model: Perceptron,
        filename: str,
        title: str,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(model.train_losses, label="train loss")
    if model.val_losses:
        plt.plot(model.val_losses, label="validation loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"График сохранён: {path}")


def plot_decision_boundary(
        model: Perceptron,
        X: np.ndarray,
        y: np.ndarray,
        filename: str,
        title: str,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
    y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probabilities = model.predict_proba(grid).reshape(xx.shape)
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, probabilities, levels=30, alpha=0.35)
    plt.contour(xx, yy, probabilities, levels=[0.5], linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=35)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"График сохранён: {path}")


def plot_metric_lines(
        rows: list[dict],
        x_key: str,
        y_key: str,
        filename: str,
        title: str,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    x_values = [row[x_key] for row in rows]
    y_values = [row[y_key] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"График сохранён: {path}")


def plot_roc_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        filename: str,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve_manual(y_true, y_score)
    auc = roc_auc_manual(y_true, y_score)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, marker=".", label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"График сохранён: {path}")


def plot_errors(
        model: Perceptron,
        X: np.ndarray,
        y: np.ndarray,
        filename: str,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    y_pred = model.predict(X)
    errors = y_pred != y
    plt.figure(figsize=(7, 6))
    plt.scatter(X[~errors, 0], X[~errors, 1], c=y[~errors], edgecolors="k", s=35)
    plt.scatter(X[errors, 0], X[errors, 1], marker="x", s=80, label="Ошибки")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title("Ошибочно классифицированные точки")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"График сохранён: {path}")


def run_base_training() -> dict:
    X_train, X_test, y_train, y_test = prepare_base_data()
    model = Perceptron(
        learning_rate=0.1,
        epochs=100,
        batch_size=32,
        init_type="small_random",
        seed=42,
    )

    model.fit(X_train, y_train, X_test, y_test)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    result = {
        "experiment": "base_training",
        "learning_rate": 0.1,
        "epochs": 100,
        "batch_size": 32,
        "train_accuracy": accuracy(y_train, y_train_pred),
        "test_accuracy": accuracy(y_test, y_test_pred),
    }

    plot_loss(
        model,
        "base_loss.png",
        "Изменение функции потерь при базовом обучении",
    )

    plot_decision_boundary(
        model,
        X_test,
        y_test,
        "base_decision_boundary.png",
        "Разделяющая граница на тестовой выборке",
    )
    return result


def experiment_learning_rate() -> list[dict]:
    X_train, X_test, y_train, y_test = prepare_base_data()
    learning_rates = [0.001, 0.01, 0.5, 1.0]
    rows = []
    for lr in learning_rates:
        model = Perceptron(
            learning_rate=lr,
            epochs=100,
            batch_size=32,
            init_type="small_random",
            seed=42,
        )
        model.fit(X_train, y_train, X_test, y_test)
        y_test_pred = model.predict(X_test)
        rows.append(
            {
                "learning_rate": lr,
                "final_train_loss": model.train_losses[-1],
                "final_val_loss": model.val_losses[-1],
                "test_accuracy": accuracy(y_test, y_test_pred),
            }
        )
        plot_loss(
            model,
            f"loss_learning_rate_{lr}.png",
            f"Loss при learning_rate={lr}",
        )
    save_csv("learning_rate_experiment.csv", rows)
    return rows


def experiment_batch_size() -> list[dict]:
    X_train, X_test, y_train, y_test = prepare_base_data()
    batch_sizes = [1, 16, 64, 256]
    rows = []
    for batch_size in batch_sizes:
        model = Perceptron(
            learning_rate=0.1,
            epochs=100,
            batch_size=batch_size,
            init_type="small_random",
            seed=42,
        )
        model.fit(X_train, y_train, X_test, y_test)
        y_test_pred = model.predict(X_test)
        rows.append(
            {
                "batch_size": batch_size,
                "final_train_loss": model.train_losses[-1],
                "final_val_loss": model.val_losses[-1],
                "test_accuracy": accuracy(y_test, y_test_pred),
            }
        )
        plot_loss(
            model,
            f"loss_batch_size_{batch_size}.png",
            f"Loss при batch_size={batch_size}",
        )
    save_csv("batch_size_experiment.csv", rows)
    return rows


def experiment_initialization() -> list[dict]:
    X_train, X_test, y_train, y_test = prepare_base_data()
    init_types = ["zero", "small_random", "large_random"]
    rows = []
    for init_type in init_types:
        model = Perceptron(
            learning_rate=0.1,
            epochs=100,
            batch_size=32,
            init_type=init_type,
            seed=42,
        )
        model.fit(X_train, y_train, X_test, y_test)
        y_test_pred = model.predict(X_test)
        rows.append(
            {
                "init_type": init_type,
                "final_train_loss": model.train_losses[-1],
                "final_val_loss": model.val_losses[-1],
                "test_accuracy": accuracy(y_test, y_test_pred),
            }
        )
        plot_loss(
            model,
            f"loss_init_{init_type}.png",
            f"Loss при инициализации: {init_type}",
        )
    save_csv("initialization_experiment.csv", rows)
    return rows


def experiment_synthetic_datasets() -> list[dict]:
    datasets = {
        "linear_blobs": generate_linear_blobs(seed=42),
        "xor": generate_xor(seed=42),
        "circle": generate_circle(seed=42),
    }
    rows = []
    for dataset_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X,
            y,
            test_size=0.3,
            seed=42,
        )
        mean, std = fit_standard_scaler(X_train)
        X_train = transform_standard_scaler(X_train, mean, std)
        X_test = transform_standard_scaler(X_test, mean, std)
        model = Perceptron(
            learning_rate=0.1,
            epochs=100,
            batch_size=32,
            init_type="small_random",
            seed=42,
        )
        model.fit(X_train, y_train, X_test, y_test)
        y_test_pred = model.predict(X_test)
        rows.append(
            {
                "dataset": dataset_name,
                "test_accuracy": accuracy(y_test, y_test_pred),
                "final_val_loss": model.val_losses[-1],
            }
        )
        plot_decision_boundary(
            model,
            X_test,
            y_test,
            f"decision_boundary_{dataset_name}.png",
            f"Разделяющая граница: {dataset_name}",
        )
    save_csv("synthetic_datasets_experiment.csv", rows)
    return rows


def experiment_regularization() -> list[dict]:
    X_train, X_test, y_train, y_test = prepare_base_data()
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
    rows = []
    for l2_lambda in lambdas:
        model = Perceptron(
            learning_rate=0.1,
            epochs=100,
            batch_size=32,
            init_type="small_random",
            l2_lambda=l2_lambda,
            seed=42,
        )
        model.fit(X_train, y_train, X_test, y_test)
        y_test_pred = model.predict(X_test)
        rows.append(
            {
                "l2_lambda": l2_lambda,
                "weights_norm": float(np.linalg.norm(model.w)),
                "final_val_loss": model.val_losses[-1],
                "test_accuracy": accuracy(y_test, y_test_pred),
            }
        )
    save_csv("regularization_experiment.csv", rows)
    return rows


def experiment_metrics_and_errors() -> dict:
    X_train, X_test, y_train, y_test = prepare_base_data()
    model = Perceptron(
        learning_rate=0.1,
        epochs=100,
        batch_size=32,
        init_type="small_random",
        seed=42,
    )
    model.fit(X_train, y_train, X_test, y_test)
    y_score = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    result = {
        "accuracy": accuracy(y_test, y_pred),
        "precision": precision(y_test, y_pred),
        "recall": recall(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_manual(y_test, y_score),
    }
    plot_roc_curve(y_test, y_score, "roc_curve.png")
    plot_errors(model, X_test, y_test, "classification_errors.png")
    save_csv("metrics_experiment.csv", [result])
    return result


def experiment_momentum() -> list[dict]:
    X_train, X_test, y_train, y_test = prepare_base_data()
    betas = [0.0, 0.5, 0.9, 0.99]
    rows = []
    for beta in betas:
        model = Perceptron(
            learning_rate=0.1,
            epochs=100,
            batch_size=32,
            init_type="small_random",
            momentum=beta,
            seed=42,
        )
        model.fit(X_train, y_train, X_test, y_test)
        y_test_pred = model.predict(X_test)
        rows.append(
            {
                "momentum_beta": beta,
                "final_train_loss": model.train_losses[-1],
                "final_val_loss": model.val_losses[-1],
                "test_accuracy": accuracy(y_test, y_test_pred),
            }
        )
        plot_loss(
            model,
            f"loss_momentum_{beta}.png",
            f"Loss при momentum={beta}",
        )
    save_csv("momentum_experiment.csv", rows)
    return rows


def run_all_experiments() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base_result = run_base_training()
    save_csv("base_training.csv", [base_result])
    learning_rate_rows = experiment_learning_rate()
    batch_size_rows = experiment_batch_size()
    initialization_rows = experiment_initialization()
    synthetic_rows = experiment_synthetic_datasets()
    regularization_rows = experiment_regularization()
    metrics_result = experiment_metrics_and_errors()
    momentum_rows = experiment_momentum()
    print("\nБазовое обучение:")
    print(base_result)
    print("\nЭксперимент со скоростью обучения:")
    print(learning_rate_rows)
    print("\nЭксперимент с размером батча:")
    print(batch_size_rows)
    print("\nЭксперимент с инициализацией:")
    print(initialization_rows)
    print("\nСинтетические датасеты:")
    print(synthetic_rows)
    print("\nРегуляризация:")
    print(regularization_rows)
    print("\nМетрики качества:")
    print(metrics_result)
    print("\nMomentum:")
    print(momentum_rows)
