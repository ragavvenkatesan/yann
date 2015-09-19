#!/usr/bin/python
from samosa.dataset import setup_dataset
def main( data_params, outs ):
              
    dataset = setup_dataset (data_params = data_params , outs = outs, preprocess_params = preprocess_params) # do this only once per dataset.
                   
    ## Boiler Plate ## 
if __name__ == '__main__':
            
    data_params = {
                   "type"               : 'skdata',                                   
                   "loc"                : 'cifar10',                                          
                   "batch_size"         : 100,                                     
                   "load_batches"       : 1, 
                   "batches2train"      : 400,                                      
                   "batches2test"       : 100,                                      
                   "batches2validate"   : 100,                                        
                   "height"             : 32,                                       
                   "width"              : 32,                                       
                   "channels"           : 3                                        
                  }
                  
    preprocess_params = { 
                            "normalize"     : True,
                            "GCN"           : True,
                            "ZCA"           : True,
                            "gray"          : True,
                        }
                  
    main( data_params = data_params, outs = 10 )