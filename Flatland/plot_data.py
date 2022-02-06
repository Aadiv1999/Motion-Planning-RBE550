import matplotlib.pyplot as plt
import numpy as np

plot_data = np.load("plot_data.npy")

for i in range(len(plot_data[0])):
    plt.plot(plot_data[:,0]*100, plot_data[:,i], marker='o', linewidth=2, markersize=7)

plt.grid(True)
plt.xlabel("Obstacle Density (%)")
plt.ylabel("# taken to reach goal")
plt.legend(["BFS", "Dijkstra", "DFS"],loc='upper right')
plt.show()