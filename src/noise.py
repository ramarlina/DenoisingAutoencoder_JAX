import numpy as np
        
class GaussianNoise():
    def __init__(self, mu, sd):
        self.mu = mu
        self.sd = sd

    def __call__(self, X):
        return X + np.random.normal(self.mu, self.sd)
