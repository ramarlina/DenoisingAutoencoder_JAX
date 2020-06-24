import sys
sys.path.append("../src")

import yfinance as yf  
import numpy as np

from model import AutoEncoder
from noise import GaussianNoise
from loss import MaxSE
from tqdm import tqdm
from utils import rolling_window

stock_data = yf.download('CFR', 
                      start='2012-01-01', 
                      end='2020-05-31', 
                      progress=False) 

closing_prices = stock_data["Close"].rolling(window=3).mean()[3:].values 

# initialization
window = 5

dA_model = AutoEncoder(window, 4)  
loss = MaxSE()

y = closing_prices.copy()

# training
for i in range(5):
    e = []
    T = rolling_window(y.copy()[:1000], window)

    np.random.shuffle(T)
    for t in tqdm(T): 
        t -= t.min()
        t /= t.max()
        t -= 0.5
        t *= 2
        e.append(dA_model.learn(t, loss, noise=GaussianNoise(0,0.001)))

    print(np.mean(e)) 

# encoding
prices = [i.copy() for i in np.array_split(y, len(y)//window) if len(i) == window]
p = []
T = []
r = []
for i in prices:
    i -= i.min()
    i /= i.max()
    i  = 2 * (i - 0.5)
    T.append(i)
    H = dA_model.encode(i)
    H = (H > np.random.uniform(-1,1,H.shape)).astype("f")
    r.append(dA_model.decode(H))
    p.append(H)

p = np.array(p)

# mapping binary representations to unique values
cid = p.dot(np.power(2, range(p.shape[1])).reshape(p.shape[1],))
unique_vals = np.unique(cid.flatten())
val_map = dict((v, i) for i, v in enumerate(unique_vals))
cid = np.array([val_map[i] for i in cid])
unique_vals = np.array(list(val_map.values()))
len(unique_vals)