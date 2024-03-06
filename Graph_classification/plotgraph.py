import os
import networkx as nx
import matplotlib.pyplot as plt
# Define the list of folders
folders = "D:/3DStepGraphClassification/GraphData/01_Shaft"
# Create an empty graph
# G = nx.Graph()
# Get the list of files in the folder
file_list = os.listdir(folders)
# Loop over the folders
for file in file_list:
    # Get the GraphML file path
    graphml_path = os.path.join(file)
    # Load the graph from the file
    G = nx.read_graphml(folders +"/" + graphml_path)
    
    # Add the nodes and edges to the main graph
    # G.add_nodes_from(subgraph.nodes(data=True))
    # G.add_edges_from(subgraph.edges(data=True))

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_spring(G, with_labels=False, node_size=20, node_color='blue')

    filenameonly = os.path.splitext(os.path.basename(file))[0]

    # Set the output file path
    output_file_path = 'D:/3DStepGraphClassification/GraphPlot/01_Shaft/' + filenameonly +".png"
    # plt.show()
    # Save the graph to the output file
    plt.savefig(output_file_path)
  

