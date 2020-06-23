import jax.numpy as jnp
import jax

class MaxSE():
    def __init__(self):
        self.grad_fn = jax.grad(self.__call__, argnums=0)

    def __call__(self, W, x, y, forward):  
        p = forward(W, x)
        e = jnp.max((y - p)**2)
        return e
