import numpy as np
from utils import fft

x = np.random.random(4)
transform = fft.DFT_slow(x)
fast_transform = fft.FFT(x)

print "Input"
print x

print "Transform"
print transform

print "Fast Transform"
print fast_transform

print np.allclose(fast_transform, np.fft.fft(x))