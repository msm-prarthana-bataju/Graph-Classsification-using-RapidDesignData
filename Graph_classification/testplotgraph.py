import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml('C:/Users/Prarthana_Bataju/Desktop/Graph_Representation_Research/data/3DStepDataClassification/FCBaseModel_Graph/01_Shaft/2014-1-117-SFJ12-76.graphml')

pos = nx.spring_layout(G)
nx.draw_spring(G, with_labels=False, node_size=80, node_color='blue')
plt.show()
plt.savefig("output_file_path")




