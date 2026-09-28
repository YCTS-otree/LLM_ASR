"""Application-owned resident model. Loading/inference run off the UI thread."""
import threading
from engines import Engine


class ModelHost:
    def __init__(self, factory=Engine):
        self.factory = factory
        self.lock = threading.RLock()
        self.engine = None
        self.options = None

    def load(self, device, precision, config):
        options = (device, precision, config)
        with self.lock:
            if self.engine is not None and options == self.options:
                return self.engine
            # Explicit device/precision changes require reloading. Do not hold two
            # large models concurrently; a failed reload can be retried next start.
            if self.engine is not None:
                self.engine.close()
            self.engine = None
            self.options = None
            self.engine = self.factory(device, precision, config=config)
            self.options = options
            return self.engine

    def close(self):
        with self.lock:
            if self.engine is not None:
                self.engine.close()
            self.engine = None
            self.options = None
