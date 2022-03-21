import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 8])
xpoints = np.arange(10, 15)
ypoints = np.cumprod(xpoints)

print(xpoints)
print(ypoints)

plt.plot(xpoints, ypoints)
plt.show()
