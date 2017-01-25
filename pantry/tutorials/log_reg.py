from yann.network import network
from yann.utils.graph import draw_network    

def log_reg ( dataset ):            
    """
    This function is a demo example of logistic regression.  

    """
    dataset_params  = { "dataset"   :  dataset,
                        "svm"       :  False, 
                        "n_classes" : 10 }
    net = network()                       
    net.add_module ( type = 'datastream',  params = dataset_params )
    net.add_layer ( type = "input",  datastream_origin = 'data')
    net.add_layer ( type = "classifier", num_classes = 10 )
    net.add_layer ( type = "objective" )
    net.cook()
    net.pretty_print()
    net.train()      
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