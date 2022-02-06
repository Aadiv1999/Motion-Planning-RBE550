from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

plot_data = np.load("plot_data.npy")
labels = ["BFS", "DFS", "Dijkstra's", "Random"]
for i in range(1,len(plot_data[0])):
    plt.plot(plot_data[:,0]*100, plot_data[:,i], marker='o', linewidth=2, markersize=7, label=labels[i-1])

plt.grid(True)
plt.xlabel("Obstacle Density (%)")
plt.ylabel("# taken to reach goal")
plt.legend(loc='upper right')
plt.show()