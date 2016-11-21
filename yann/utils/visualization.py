import matplotlib.pyplot as plt
import networkx as nx

def draw_network(graph, filename = 'network.pdf', show = False):
    """
    This is a simple wrapper to the networkx_draw.
    Args:
        filename: what file to save down as.
        show: will display the graph on a window. 
    Nots:
        Takes any format that networkx matplotlib plotter takes.
    """
    pos=nx.spectral_layout(graph)
    labels = {}
    for node in graph.nodes(): 
        labels[node] = node
    nx.draw(    graph,
                pos = pos,
                node_size = 1000,
                node_color = 'g',
                node_shape = 'o',
                width = 1.0,
                labels = labels )
    plt.savefig(filename)
    if show == True:
        plt.show()
