import os
from abstract import module
import matplotlib.pyplot as plt
import numpy

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
                    "learning_rate" : "<learning_rate_file_name>.txt"
                    "momentum"  : <momentum_file_name>.txt
                    "visualize" : <bool>
                    "id"        : id of the resultor
                                }

            While the filenames are optional, ``root`` must be provided. If a particular file is
            not provided, that value will not be saved.

    Returns:
        yann.modules.resultor: A resultor object

    TODO:
        Remove the input file names, assume file names as default. 

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
            resultor_init_args["confusion"] = "confusion.eps"

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
        
        for file in [self.results_file, self.error_file, self.cost_file,
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

    def update_plot (self, verbose = 2):
        """
        TODO:

            This method should update the open plots with costs and other values. Ideally, a browser
            based system should be implemented, such as using mpl3d or using bokeh. This system
            should open opne browser where it should update realtime the cost of training, validation
            and testing accuracies per epoch, display the visualizations of filters, some indication
            of the weight of gradient trained, confusion matrices, learning rate and momentum plots
            etc. 
        """
        print "TBD"                    
    
    def print_confusion (self, epoch=0, train = None, valid = None, test = None, verbose = 2):
        """
        This method will print the confusion matrix down in files. 

        Args:
            epoch: This is used merely to create a directory for each epoch so that there is a copy.
            train: training confusion matrix as gained by the validate method.
            valid: validation confusion amtrix as gained by the validate method.
            test: testing confusion matrix as gained by the test method.
            verbose: As usual.
        """
        if verbose >=3:
            print ("... Printing confusion matrix")
        if not os.path.exists(self.root + '/confusion'):
            if verbose >= 3:
                print "... Creating a root directory for saving confusions"
            os.makedirs(self.root + '/confusion')

        location = self.root + '/confusion' + '/epoch_' + str(epoch)        
        if not os.path.exists( location ):
            if verbose >=3 :
                print "... Making the epoch directory"
            os.makedirs (location)

        if verbose >=3 :
            print ("... Saving down the confusion matrix")

        if not train is None:
            self._store_confusion_img (confusion = train,
                              filename = location + '/train_confusion.eps',
                              verbose = 2)
        if not valid is None:
            self._store_confusion_img (confusion = valid,
                              filename = location + '/valid_confusion.eps',
                              verbose = 2)

        if not test is None:
            self._store_confusion_img (confusion = test,
                              filename = location + '/test_confusion.eps',
                              verbose = 2)

    def _store_confusion_img (self, confusion, filename, verbose = 2):
        """
        Convert a normalized confusion matrix into an image and save it down.

        Args:
            confusion: confusion matrix.
            filename: save the image at the location as a file.
            verbose: as usual.
        """
        corrects = numpy.trace(confusion)       
        total_samples = numpy.sum(confusion)
        accuracy = 100 * corrects / float(total_samples)
        if verbose >= 3:
            print ("... Saving the file down")
        confusion = confusion / confusion.sum(axis = 1)[:,None]
        fig = plt.figure(figsize=(4, 4), dpi=1200)
        plt.matshow(confusion)
        for (i, j), z in numpy.ndenumerate(confusion):
            plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=10, color = 'm') 

        plt.title('Accuracy: ' + str(int(corrects)) + '/' + str(int(total_samples)) + \
                                                ' = ' + str(round(accuracy,2)) + '%')
        plt.set_cmap('GnBu')
        plt.colorbar()
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.savefig(filename)
        plt.close('all')