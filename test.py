import numpy as np
from utils import fft

raw_data = np.random.randint(0, 255, (3,100))

#raw_data.resize(50*50*3)
#print raw_data.shape

data = raw_data.reshape(3, 10, 10)
print data.shape

# x = np.random.random(4)
# transform = fft.DFT_slow(x)
# fast_transform = fft.FFT(x)

# print "Input"
# print x

# print "Transform"
# print transform

# print "Fast Transform"
# print fast_transform

# print np.allclose(fast_transform, np.fft.fft(x))