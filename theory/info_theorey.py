

# ------------------------ Code --------------------------------------
# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

import numpy as np
import math as np

PI = 3.1415926
E = 2.71828


"""
I find out a easier way to address them.
"""


def renyi_entropy_1(rank, dimension, scale):
    sigma2 = scale/dimension
    renyi = 1/(1e-10)*np.log(sigma2*rank*dimension)
    return renyi


def renyi_entropy_2(rank, dimension, scale, distribution_type="gaussian"):
    sigma2 = scale/dimension

    if distribution_type == "gaussian":
        renyi = rank*sigma2*sigma2*(rank+2)*dimension
    else:
        renyi = sigma2*sigma2*(4*rank+5*rank*rank)/5*dimension
    renyi = 1/1*np.log(renyi)
    return renyi


# def renyi_entropy(eigenvalues, alpha=1):
#     """
#     renyi entropy computation
#     """

#     if alpha == 1:
#         return shannon_entropy(eigenvalues)
#     else:
#         scale = 1/(1-alpha)
#         assert len(eigenvalues) != 0
#         res = 0.
#         for i, x in enumerate(eigenvalues):
#             res += x**alpha
#         res = math.log(res)
#         return res


# def shannon_entropy(eigenvalues):
#     assert len(eigenvalues) != 0

#     res = 0.
#     for i, x in enumerate(eigenvalues):
#         if x == 0:
#             res += 0
#         else:
#             res += x*math.log(x)
#     return -1*res


# def entropy_power(eigenvalues):

#     n = len(eigenvalues)
#     assert n != 0

#     entropy_value = shannon_entropy(eigenvalues)

#     power = 1/(2*PI*E) * math.exp(2/n*entropy_value)

#     return power
