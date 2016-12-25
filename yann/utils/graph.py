from networkx.drawing.nx_agraph import to_agraph

def draw_network(graph, filename = 'network.png', show = False, verbose = 2 ):
    """
    This is a simple wrapper to the networkx_draw.

    Args:
        graph: Supply a networkx graph object. NNs are all DiGraphs. 
        filename: what file to save down as. Will add '.png' to the end.
        verbose: Do I even have to talk about this ?
        
    Notes:
        Takes any format that networkx plotter takes. This is not ready to be used. Still buggy 
        sometimes.
        Rudra is working on developing this further internally.
        This is slow at the moment.
    """
    if verbose >=2:
        print ".. Saving the network down as an image"

    # convert from networkx -> pydot
    if verbose >=3 :
        print "... Coverting to dot"
    """
    cgraph=nx.complete_graph(net.graph)
    temp_agraph=nx.to_agraph(cgraph)
    agraph=nx.from_agraph(temp_agraph)
    """
    agraph = to_agraph(graph)
    # Need to work on putting a box around nodes of the same layer.
    agraph.layout(prog = 'dot')
    agraph.draw(filename)
