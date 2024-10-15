"""
======================================================================
CHAR_SWAPPING ---

Swapping chars in a given text.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 27 September 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

nltk.download("punkt")

from builtins import str

import csv
import sys, getopt

import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, Node
import xml.dom.minidom
from numpy import double

from random import seed
from random import randint

seed(20240927)


def perturbeBySwapping(text: str):
    is_sample_perturbed = False
    sample_tokenized = nltk.word_tokenize(text)

    random_word_index = 0
    random_word_selected = False

    i_break = 0
    while random_word_selected is not True:
        if i_break >= 100:
            print(f">>>TEXT:{text}.")
            return text
            break
        random_word_index = return_random_number(0, len(sample_tokenized) - 1)
        if len(sample_tokenized[random_word_index]) > 2:
            random_word_selected = True
        i_break += 1

    print("Selected random word:", sample_tokenized[random_word_index])

    # --------------------------- select a random position

    selected_word = sample_tokenized[random_word_index]

    random_char_index = return_random_number(0, len(selected_word) - 1)
    print("Random position:", random_char_index)
    print("Char in random position:", selected_word[random_char_index])

    # --------------------------- select an adjacent for swapping

    adjacent_for_swapping = ""

    if random_char_index == 0:
        adjacent_for_swapping = "right"
    elif random_char_index == len(selected_word) - 1:
        adjacent_for_swapping = "left"
    else:
        adjacent = return_random_number(1, 2)
        if adjacent == 1:
            adjacent_for_swapping = "left"
        else:
            adjacent_for_swapping = "right"

    print("Adjacent for swapping:", adjacent_for_swapping)

    # --------------------------- swap the character and the adjacent

    temp_word = swap_characters(selected_word, random_char_index, adjacent_for_swapping)

    perturbed_word = ""
    for i in range(0, len(temp_word)):
        perturbed_word += temp_word[i]

    print("After swapping:", perturbed_word)

    # --------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):
        perturbed_sample += sample_tokenized[i] + " "

    perturbed_sample += perturbed_word + " "
    is_sample_perturbed = True

    for i in range(random_word_index + 1, len(sample_tokenized)):
        perturbed_sample += sample_tokenized[i] + " "

    print("Perturbed sample:", perturbed_sample)

    return perturbed_sample


def return_random_number(begin, end):
    return randint(begin, end)


def swap_characters(input_word, position, adjacent):
    temp_word = ""
    if adjacent == "left":
        if position == 1:
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif position == len(input_word) - 1:
            temp_word = input_word[0 : position - 1]
            temp_word += input_word[position]
            temp_word += input_word[position - 1]
        elif position > 1 and position < len(input_word) - 1:
            temp_word = input_word[0 : position - 1]
            temp_word += input_word[position]
            temp_word += input_word[position - 1]
            temp_word += input_word[position + 1 :]

    elif adjacent == "right":
        if position == 0:
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif position == len(input_word) - 2:
            temp_word = input_word[0:position]
            temp_word += input_word[position + 1]
            temp_word += input_word[position]
        elif position > 0 and position < len(input_word) - 2:
            temp_word = input_word[0:position]
            temp_word += input_word[position + 1]
            temp_word += input_word[position]
            temp_word += input_word[position + 2 :]

    return temp_word


## running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
