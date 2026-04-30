"""
Base federated learning client: PyTorch MLP + weight serialization helpers.
"""
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FedMLP(nn.Module):
    def __init__(self, input_dim: int = 20, hidden1: int = 64,
                 hidden2: int = 32, output_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------

class BaseClient:
    def __init__(self, client_id: str, client_name: str,
                 learning_rate: float = 0.001, local_epochs: int = 5,
                 batch_size: int = 32):
        self.client_id    = client_id
        self.client_name  = client_name
        self.local_epochs = local_epochs
        self.batch_size   = batch_size

        self.model     = FedMLP()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> OrderedDict:
        """Return model weights as an OrderedDict of numpy arrays."""
        return OrderedDict(
            {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        )

    def set_weights(self, weights_dict: OrderedDict) -> None:
        """Load weights from an OrderedDict of numpy arrays."""
        state = OrderedDict(
            {k: torch.tensor(v) for k, v in weights_dict.items()}
        )
        self.model.load_state_dict(state)

    def serialize_weights(self) -> str:
        """Serialize weights to a JSON string for Snowflake VARIANT storage."""
        return json.dumps(
            {k: v.tolist() for k, v in self.get_weights().items()}
        )

    @staticmethod
    def deserialize_weights(json_str: str) -> OrderedDict:
        """Deserialize a JSON string back to an OrderedDict of numpy arrays."""
        raw = json.loads(json_str) if isinstance(json_str, str) else json_str
        return OrderedDict({k: np.array(v) for k, v in raw.items()})

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _make_dataloader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def train_local(self, dataloader: DataLoader) -> dict:
        """Run local_epochs of training. Returns loss, accuracy, num_samples."""
        self.model.train()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(self.local_epochs):
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss    += loss.item() * len(y_batch)
                total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total_samples += len(y_batch)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            "loss":        avg_loss,
            "accuracy":    accuracy,
            "num_samples": len(dataloader.dataset),
        }

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on a dataloader without updating weights."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss    = 0.0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits        = self.model(X_batch)
                loss          = self.criterion(logits, y_batch)
                total_loss   += loss.item() * len(y_batch)
                total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total_samples += len(y_batch)

        return {
            "loss":     total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
