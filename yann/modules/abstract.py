"""
Todo:
    * Something is wrong with the d3viz visualizer for html printing. The path is weird.
    * Visualizer module needs to make use of mathplotlib and print online graphs of outputs of cost
      and possibly display first layer filters for CNNs
    * Datastream should include fuel interface and also needs interface for COCO, PASCAL and 
      IMAGENET. Also consider migrating to hd5 for larger datasets ? Should also be able to create
      datasets from images in python. Right now its a roundabout way of going via matlab.
    
"""

class module(object):
    """
    Prototype for what a layer should look like. Every layer should inherit from this.
    """
    def __init__(self, id, type, verbose = 2):
        self.id = id
        self.type = type
        # Every layer must have these four properties.
        if verbose >= 3:
            print "... Initializing a new module " + self.id + " of type " + self.type        


if __name__ == '__main__':
    pass              