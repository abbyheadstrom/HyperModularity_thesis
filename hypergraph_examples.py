import matplotlib.pyplot as plt
import xgi



#hyperedges = [[1, 2, 3], [2, 3, 4], [4, 5, 6, 1], [5, 6, 7], [7, 8, 9], [8, 9, 1], [3, 6], [6, 9]]
# hyperedges = [[1, 2, 3], [3, 4, 5], [3, 6], [6, 7, 8, 9], [1, 4, 10, 11, 12], [1, 4], [6, 7, 3]]
# hyperedges = [[1, 2, 3, 4], [3, 4, 5, 6], [6, 7, 8], [8, 9, 10, 11, 12], [5, 9], [2, 7, 11]]
hyperedges = [[1, 2, 3, 4, 5], [5, 6, 7, 8], [8, 9, 10], [10, 11, 12, 13], [3, 9, 13], [6, 11], [1, 6, 12]]
H = xgi.Hypergraph(hyperedges)


pos = xgi.barycenter_spring_layout(H, seed=1)
xgi.draw(H, pos=pos)

plt.show()

