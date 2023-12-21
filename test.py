import numpy as np
import matplotlib.pyplot as plt
data = np.load("vector_matrix.npy")
plt.figure(figsize=(8, 6)) 
plt.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto')
plt.colorbar()  # Add a colorbar for reference
plt.title('Matrix Visualization')
plt.show()