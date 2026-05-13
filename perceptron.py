from __future__ import annotations
import numpy as np

EPS = 1e-12


class Perceptron:
    def __init__(
            self,
            learning_rate: float = 0.1,
            epochs: int = 100,
            batch_size: int = 32,
            init_type: str = "small_random",
            loss_type: str = "cross_entropy",
            l2_lambda: float = 0.0,
            momentum: float = 0.0,
            seed: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_type = init_type
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda
        self.momentum = momentum
        self.seed = seed
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def initialize_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        if self.init_type == "zero":
            self.w = np.zeros(n_features)
            self.b = 0.0
        elif self.init_type == "small_random":
            self.w = rng.normal(0.0, 0.01, size=n_features)
            self.b = 0.0
        elif self.init_type == "large_random":
            self.w = rng.normal(0.0, 10.0, size=n_features)
            self.b = 0.0
        else:
            raise ValueError(f"Неизвестный тип инициализации: {self.init_type}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Модель ещё не обучена")
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.w is None:
            raise ValueError("Модель ещё не обучена")
        if self.loss_type == "cross_entropy":
            y_pred = np.clip(y_pred, EPS, 1.0 - EPS)
            loss = -np.mean(
                y_true * np.log(y_pred)
                + (1.0 - y_true) * np.log(1.0 - y_pred)
            )

        elif self.loss_type == "hinge":
            y_signed = np.where(y_true == 1, 1, -1)
            scores = 2.0 * y_pred - 1.0
            loss = np.mean(np.maximum(0.0, 1.0 - y_signed * scores))

        else:
            raise ValueError(f"Неизвестная функция потерь: {self.loss_type}")

        l2_penalty = 0.5 * self.l2_lambda * np.sum(self.w ** 2)
        return float(loss + l2_penalty)

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred_labels: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred_labels))

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
    ) -> "Perceptron":
        n_samples, n_features = X_train.shape
        self.initialize_weights(n_features)
        if self.w is None:
            raise ValueError("Ошибка инициализации весов")

        rng = np.random.default_rng(self.seed)
        velocity_w = np.zeros_like(self.w)
        velocity_b = 0.0

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = self.forward(X_batch)
                error = y_pred - y_batch
                grad_w = X_batch.T @ error / len(X_batch)
                grad_b = float(np.mean(error))

                if self.l2_lambda > 0:
                    grad_w += self.l2_lambda * self.w
                if self.momentum > 0:
                    velocity_w = self.momentum * velocity_w - self.learning_rate * grad_w
                    velocity_b = self.momentum * velocity_b - self.learning_rate * grad_b
                    self.w += velocity_w
                    self.b += velocity_b
                else:
                    self.w -= self.learning_rate * grad_w
                    self.b -= self.learning_rate * grad_b

            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_pred)
            train_labels = self.predict(X_train)
            train_acc = self.accuracy_score(y_train, train_labels)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_labels = self.predict(X_val)
                val_acc = self.accuracy_score(y_val, val_labels)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
