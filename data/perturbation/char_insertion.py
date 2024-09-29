"""
======================================================================
CHAR_INSERTION --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 28 September 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
nltk.download("punkt")

import csv
import sys, getopt

import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, Node
import xml.dom.minidom       
from numpy import double

from random import seed
from random import randint

seed(1)


def return_random_number(begin, end):
    return randint(begin, end)

def perturbeCharInsertion(text:str):
    is_sample_perturbed = False
    sample_tokenized = nltk.word_tokenize(text)

    random_word_index = 0
    random_word_selected = False

    while (random_word_selected != True):
        random_word_index = return_random_number(0, len(sample_tokenized)-1)
        if (len(sample_tokenized[random_word_index]) > 2):
            random_word_selected = True

    print('Selected random word:', sample_tokenized[random_word_index])

    #--------------------------- select a random position

    selected_word = sample_tokenized[random_word_index]

    random_char_index = return_random_number(1, len(selected_word)-2)
    print('Random position:', random_char_index)

    #--------------------------- select a random character

    random_char_code = return_random_number(97, 122)
    print('Random character:', chr(random_char_code))

    temp_word = selected_word[:random_char_index]
    temp_word += chr(random_char_code)
    temp_word += selected_word[random_char_index:]

    perturbed_word = ""
    for i in range(0, len(temp_word)):
        perturbed_word += temp_word[i]

    print('After insertion:', perturbed_word)

    #--------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):

        perturbed_sample += sample_tokenized[i] + ' '

    perturbed_sample += perturbed_word + ' '
    is_sample_perturbed = True

    for i in range(random_word_index+1, len(sample_tokenized)):    
        perturbed_sample += sample_tokenized[i] + ' '

    print('Perturbed sample:', perturbed_sample)

    return perturbed_sample



## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


