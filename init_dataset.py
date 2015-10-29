#!/usr/bin/python
from samosa.dataset import setup_dataset
import os
import gzip
import cPickle

# this loads up the data_params from a folder and sets up the initial databatch.         
def reset ( dataset, data_params ):
    import pdb
    pdb.set_trace()
    os.remove(dataset + '/data_params.pkl.gz')
    f = gzip.open(dataset +  '/data_params.pkl.gz', 'wb')
    cPickle.dump(new_data_params, f, protocol=2)
    f.close()			

def setup( data_params, outs ):
        
    dataset = setup_dataset (data_params = data_params , outs = outs, preprocess_params = preprocess_params) # do this only once per dataset.
                   
    ## Boiler Plate ## 
if __name__ == '__main__':
              
    data_params = {
                   "type"               : 'mat',                                   
                   "loc"                : '../colored_mnist/',                                          
                   "batch_size"         : 500,                                     
                   "load_batches"       : 1, 
                   "batches2train"      : 100,                                      
                   "batches2test"       : 20,                                      
                   "batches2validate"   : 20,                                        
                   "height"             : 28,                                       
                   "width"              : 28,                                       
                   "channels"           : 3                                       
                  }
                  
    preprocess_params = { 
                            "normalize"     : False,
                            "GCN"           : False,
                            "ZCA"           : False,
                            "gray"          : False,
                        }
                  
    setup( data_params = data_params, outs = 8 )
    
    """
    # If you want to modify data_params. 
    dataset = "_datasets/_dataset_81761"
    new_data_params = {
                    "type"               : 'base',                                   
                    "loc"                : dataset,                                          
                    "batch_size"         : 80,                                    
                    "load_batches"       : 1,
                    "batches2train"      : 34,                                      
                    "batches2test"       : 19,                                     
                    "batches2validate"   : 4,                                       
                    "height"             : 128,                                      
                    "width"              : 128,                                       
                    "channels"           : 3,
                    "multi_load"		 : True,
                    "n_train_batches"	 : 1,
                    "n_test_batches"	 : 1,
                    "n_valid_batches"	 : 1  					                                        
                    }
                    
    reset( dataset = dataset, data_params = new_data_params)
    
    """