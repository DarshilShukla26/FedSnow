"""Device client — 50/50 balanced (IoT sensor readings)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clients.base_client import BaseClient
from config import LEARNING_RATE, LOCAL_EPOCHS, BATCH_SIZE, DATA_DIR


class DeviceClient(BaseClient):
    def __init__(self):
        super().__init__(
            client_id="client_device",
            client_name="Device",
            learning_rate=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
        )
        self._load_data()

    def _load_data(self):
        path = os.path.join(DATA_DIR, "client_device.csv")
        df   = pd.read_csv(path)
        self.X = df.drop("label", axis=1).values.astype(np.float32)
        self.y = df["label"].values.astype(np.int64)

    def get_dataloader(self):
        return self._make_dataloader(self.X, self.y)
