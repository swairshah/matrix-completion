import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def clean_data(m, n, r):
    """ 
    Generate mxn matrix with rank r
    """
    U = np.random.randn(m, r)
    V = np.random.randn(n ,r)
    return U, V, U @ V.T

def noisy_data(m, n, r, sigma = 0.5):
    """
    mxn matrix with rank r,
    and added gaussian noise
    """
    U, V, X = clean_data(m, n, r)
    noise = np.random.randn(m, n) * sigma
    X += noise
    return U, V, X

def show(X):
    im = Image.fromarray(X, 'LA')
    plt.imshow(im)
    plt.show()
