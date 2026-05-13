from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification


def generate_base_dataset(
        n_samples: int = 500,
        n_features: int = 2,
        n_redundant: int = 0,
        n_informative: int = 2,
        n_clusters_per_class: int = 1,
        random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=n_redundant,
        n_informative=n_informative,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state,
    )
    return X.astype(float), y.astype(int)


def stratified_train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for class_value in np.unique(y):
        class_indices = np.where(y == class_value)[0]
        rng.shuffle(class_indices)
        n_test = int(len(class_indices) * test_size)
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def fit_standard_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    return mean, std


def transform_standard_scaler(
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
) -> np.ndarray:
    return (X - mean) / std


def generate_linear_blobs(
        n_samples: int = 500,
        noise: float = 0.5,
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2
    class_0 = rng.normal(loc=(-2, -2), scale=noise, size=(n_half, 2))
    class_1 = rng.normal(loc=(2, 2), scale=noise, size=(n_samples - n_half, 2))
    X = np.vstack([class_0, class_1])
    y = np.array([0] * n_half + [1] * (n_samples - n_half))
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def generate_xor(
        n_samples: int = 500,
        noise: float = 0.2,
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    X += rng.normal(0.0, noise, size=X.shape)
    return X, y


def generate_circle(
        n_samples: int = 500,
        noise: float = 0.1,
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n_samples, 2))
    radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = (radius > 0.8).astype(int)
    X += rng.normal(0.0, noise, size=X.shape)
    return X, y
