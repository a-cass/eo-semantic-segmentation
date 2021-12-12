from tensorflow.keras.callbacks import Callback
import time

class TrainingTimer(Callback):
    """Times the model's training process.
    """

    def __init__(self):
        self.logs = {}

    def on_train_begin(self, logs=None):
        self.logs['start_time'] = time.time()

    def on_train_end(self, logs=None):
        self.logs['stop_time'] = time.time()
        self.logs['runtime'] = self.logs['stop_time'] - self.logs['start_time']

    def runtime_seconds(self):
        return self.logs.get('runtime', 0.)

    def runtime_minutes(self):
        return self.runtime_seconds() / 60

    def runtime_hours(self):
        return self.runtime_seconds() / 3600
