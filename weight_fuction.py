import numpy as np
from matplotlib import pyplot

x = np.arange(0.01,1,0.01)
# weight = np.exp(1/score)
# weight = 1/x
# weight = np.power(1/score,1/score)
# weight = np.exp2(1/score)
# weight = 1 / x ** 4
weight = (-np.log(x-0.07))**4
pyplot.figure()
pyplot.plot(x,weight)
pyplot.show()
