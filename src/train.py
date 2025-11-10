'''
How to run/execute this it (make sure you ran preprocess.py first to build vocab and clean data, 
                            you can see how to run it at the top comment of the program in preprocess.py)

'''


# Adding other 2 program files I write with their functions for this 
from .utils import seed_everything, make_loader, compute_metrics, load_json
from .models import RNNClassifier