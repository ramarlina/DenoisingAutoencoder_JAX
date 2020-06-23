import numpy as np
import jax
import jax.numpy as jnp

def Activation(fname):
    if fname == "tanh":
        return jnp.tanh 

class AutoEncoder():   
    def __init__(self, n_input, n_hidden, lr=0.01, activation="tanh"):
        self.W = np.random.randn(n_input,n_hidden)*np.sqrt(1/n_input) 
        self.activation = activation
        self.lr = lr  

    def learn(self, X, loss, corruption=Identity()):
        e = loss(self.W, corruption(X), X, self.forward)  
        grad = loss.grad_fn(self.W, X, X, self.forward) 
        self.W -= self.lr * np.array(grad)
        return float(e) 

    def forward(self, W, X):
        H = Activation(self.activation)(jnp.dot(X,W))
        H = (H > np.random.uniform(-1, 1, H.shape)).astype(np.float32)
        X_ = Activation(self.activation)(jnp.dot(H, jnp.transpose(W,(1,0))))
        return X_

    def encode(self, X):
        return Activation(self.activation)(X.dot(self.W))

    def decode(self, X):
        return Activation(self.activation)(X.dot(self.W.T))
 
