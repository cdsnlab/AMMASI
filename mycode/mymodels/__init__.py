from mymodels.neighbor_attention import *
from mymodels.neighbor import *
from mymodels.basic import *
from mymodels.naive import *
# from mymodels.rgat import *

import numpy as np
import tensorflow as tf
import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname) 

