"""
Notes:
    This code contains one method that explains how to build a 
    logistic regression classifier for the MNIST dataset using
    the yann toolbox.

    For a more interactive tutorial refer the notebook at 
    yann/pantry/tutorials/notebooks/Logistic Regression.ipynb
"""

from yann.network import network
from yann.utils.graph import draw_network    

def log_reg ( dataset ):            
    """
    This function is a demo example of logistic regression.  

    """
    # Create the yann network class with empty layers.
    net = network()    

    # Setup the datastream module and add it to network.
    dataset_params  = { "dataset"   :  dataset,
                        "svm"       :  False, 
                        "n_classes" : 10 }                       
    net.add_module ( type = 'datastream',  params = dataset_params )

    # Create an input layer that feeds from the datastream modele.
    net.add_layer ( type = "input",  datastream_origin = 'data')

    # Create a logistic regression layer.
    # Creates a softmax layer.
    net.add_layer ( type = "classifier", num_classes = 10 )

    # Create an objective layer.
    # Default is negative log likelihood.
    # What ever the objective is, is always minimized. 
    net.add_layer ( type = "objective" )

    # Cook the network.
    net.cook()

    # See how the network looks like.
    net.pretty_print()

    # Train the network.
    net.train()      

    # Test for acccuracy.
    net.test()


## Boiler Plate ## 
if __name__ == '__main__':
    dataset = None  
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_dataset':
            from yann.special.datasets import cook_mnist  
            data = cook_mnist (verbose = 3)
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"
    
    if dataset is None:
        print " creating a new dataset to run through"
        from yann.special.datasets import cook_mnist  
        data = cook_mnist (verbose = 3)
        dataset = data.dataset_location()

    log_reg ( dataset ) 