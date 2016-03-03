#!/usr/bin/python
from dataset import setup_dataset
import os
import cPickle

# this loads up the data_params from a folder and sets up the initial databatch.         
def reset ( dataset, data_params ):
    import pdb
    pdb.set_trace()
    os.remove(dataset + '/data_params.pkl')
    f = open(dataset +  '/data_params.pkl', 'wb')
    cPickle.dump(new_data_params, f, protocol=2)
    f.close()			

def setup( data_params, outs ):
        
    dataset = setup_dataset (data_params = data_params , outs = outs, preprocess_params = preprocess_params) # do this only once per dataset.
                   
    ## Boiler Plate ## 
if __name__ == '__main__':
    
    data_params = {
                   "type"               : 'skdata',                                   
                   "loc"                : 'caltech101',                                          
                   "batch_size"         : 36,                                     
                   "load_batches"       : 1, 
                   "batches2train"      : 127,                                      
                   "batches2test"       : 63,                                      
                   "batches2validate"   : 64,                                        
                   "height"             : 32,                                       
                   "width"              : 32,                                       
                   "channels"           : 3                                       
                  }
                      
    # parameters relating to preprocessing.
    preprocess_params = { 
                            "normalize"     : True,
                            "GCN"           : False,
                            "ZCA"           : False,
                            "gray"          : False,
                        }
          
    # run and it will create a directory with a random name. 
    # Ensure that _datasets directory exist in the directory from which this is being called from.
    setup( data_params = data_params, outs = 102 )
    
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