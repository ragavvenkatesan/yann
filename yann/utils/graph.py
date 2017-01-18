from networkx.drawing.nx_pydot import to_pydot

def draw_network(graph, filename = 'network.pdf', show = False, verbose = 2 ):
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

    dot = to_pydot(graph)
    dot.set_node_defaults(style="filled", fillcolor="grey")
    dot.set_edge_defaults(color="blue", arrowhead="vee", weight="0")    
    if verbose >=3 :
        print "... Writing down"
    dot.write_png(filename)    
