import numpy as np
from utils import fft

x = np.random.random(5)
transform = fft.DFT_slow(x)

print "Input"
print x

print "Transform"
print transform