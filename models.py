import tensorflow
import numpy as np


class BasicModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        pass

    def get_action(self, state):
        return np.array([1.0, 0.0, 0.0])

        self.model.predict()
