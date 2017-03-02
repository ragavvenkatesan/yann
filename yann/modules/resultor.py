import os
from abstract import module

class resultor(module):
    """
    Resultor of the network saves down resultor. The initilizer initializes the directories
    for storing results.

    Args:
        verbose:  Similar to any 3-level verbose in the toolbox.
        resultor_init_args: ``resultor_init_args`` is a dictionary of the form

            .. code-block:: none

                resultor_init_args    =    {
                    "root"      : "<root directory to save stuff inside>",
                    "results"   : "<results_file_name>.txt",
                    "errors"    : "<error_file_name>.txt",
                    "costs"     : "<cost_file_name>.txt",
                    "confusion" : "<confusion_file_name>.txt",
                    "learning_rate" : "<learning_rate_file_name>.txt"
                    "momentum"  : <momentum_file_name>.txt
                    "visualize" : <bool>
                    "id"        : id of the resultor
                                }

            While the filenames are optional, ``root`` must be provided. If a particular file is
            not provided, that value will not be saved.

    Returns:
        yann.modules.resultor: A resultor object

    """
    def __init__( self, resultor_init_args, verbose = 1):
        if "id" in resultor_init_args.keys():
            id = resultor_init_args["id"]
        else:
            id = '-1'
        super(resultor,self).__init__(id = id, type = 'resultor')

        if verbose >= 3:
            print "... Creating resultor directories"

        if not "root" in resultor_init_args.keys():
            resultor_init_args["root"] = "."

        if not "results" in resultor_init_args.keys():
            resultor_init_args["results"] = "results.txt"

        if not "errors" in resultor_init_args.keys():
            resultor_init_args["errors"] = "errors.txt"

        if not "costs" in resultor_init_args.keys():
            resultor_init_args["costs"] = "costs.txt"

        if not "confusion" in resultor_init_args.keys():
            resultor_init_args["confusion"] = "confusion.txt"

        if not "learning_rate" in resultor_init_args.keys():
            resultor_init_args["learning_rate"] = "learning_rate.txt"

        if not "momentum" in resultor_init_args.keys():
            resultor_init_args["momentum"] = "momentum.txt"

        if not "viualize" in resultor_init_args.keys():
            resultor_init_args["visualize"] = True

        for item, value in resultor_init_args.iteritems():
            if item == "root":
                self.root                   = value
            elif item == "results":
                self.results_file           = value
            elif item == "errors":
                self.error_file             = value
            elif item == "costs":
                self.cost_file              = value
            elif item == "confusion":
                self.confusion_file         = value
            elif item == "learning_rate":
                self.learning_rate          = value
            elif item == "momentum":
                self.momentum               = value


        if not hasattr(self, 'root'): raise Exception('root variable has not been provided. \
                                            Without a root folder, no save can be performed')
        if not os.path.exists(self.root):
            if verbose >= 3:
                print "... Creating a root directory for save files"
            os.makedirs(self.root)
        
        for file in [self.results_file, self.error_file, self.cost_file, self.confusion_file,
                     self.learning_rate, self.momentum]:
            f = open(self.root + "/" + file, 'w')
            f.close()

        if verbose >= 3:
            print ( "... Resultor is initiliazed" )

    def process_results(    self,
                            cost,
                            lr,
                            mom,                        
                            verbose = 2 ):
        """
        This method will print results and also write them down in the appropriate files.

        Args:
            cost: Cost, is a float
            lr: Learning Rate, is a float
            mom: Momentum, is a float.
        """
        print ( ".. Cost                : " + str(cost) )
        if verbose >= 3:
            print ( "... Learning Rate       : " + str(lr) )
            print ( "... Momentum            : " + str(mom) )

        f = open(self.root + "/" + self.cost_file, 'a')
        f.write(str(cost))
        f.write('\n')
        f.close()

        f = open(self.root + "/" + self.learning_rate, 'a')
        f.write(str(lr))
        f.write('\n')
        f.close()

        f = open(self.root + "/" + self.momentum, 'a')
        f.write(str(mom))
        f.write('\n')        
        f.close()    

    def plot (self, verbose = 2):
        """
        This method will (should) plot all the values in the files.
        """
        print "TBD"

    def update_plot (self, verbose = 2):
        """
        This method should update the open plots with costs and other values.
        """
        print "TBD"                    