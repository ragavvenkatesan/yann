import matplotlib.pyplot as plt
import networkx as nx

def _search_list(list, search):
    """
    internal searching of the list
    """
    present = False
    for sublist in list:
        if search in sublist:
            present = True 
            break 
    return present

def draw_network(graph, filename = 'network.pdf', show = False):
    """
    This is a simple wrapper to the networkx_draw.
    Args:
        graph: Supply a networkx graph object. NNs are all DiGraphs. 
        filename: what file to save down as.
        show: will display the graph on a window. 
    Notes:
        Takes any format that networkx plotter takes. This is not ready to be used. Still buggy 
        sometimes.
        Rudra is working on developing this further internally.
    """

    labels = {}
    node_size = []
    node_list = []
    shells = []

    for node in graph.nodes(): 
        labels [ node ] = node    
        node_size.append ( len(node) * 1000 )
        node_list.append ( node ) 
        succ_list = []        
        for succ in graph.successors(node):
            succ_list = []            
            if not _search_list(shells, succ) is True: 
                succ_list.append(succ) 
        if not len(succ_list) == 0:
            shells.append(succ_list) 

    pos=nx.spectral_layout(graph,shells)
    nx.draw_networkx_nodes( G = graph, 
                            pos = pos,
                            node_list = node_list,
                            node_size = node_size,
                            node_color = 'g',
                            node_shape = 'o' )
    nx.draw_networkx_edges(G = graph, pos = pos)
    nx.draw_networkx_labels(G = graph, pos = pos , 
                            labels = labels ,font_family='sans-serif',
                            font_size=7)
    plt.savefig(filename)

    if show == True:
        plt.show()
