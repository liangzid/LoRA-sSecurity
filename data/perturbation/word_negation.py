"""
======================================================================
WORD_NEGATION ---

Refer to https://github.com/mmoradi-iut/NLP-perturbation/blob/main/Word-Negation.py

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

nltk.download("averaged_perceptron_tagger")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("punkt")

import csv
import sys, getopt

import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, Node
import xml.dom.minidom
from numpy import double

from random import seed
from random import randint
import mlconjug3

seed(20240927)


def return_random_number(begin, end):
    return randint(begin, end)


def perturbe_a_sample(text: str):
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(text)
    sample_pos_tag = nltk.pos_tag(sample_tokenized)

    # print(sample_pos_tag)

    Perturbed_sample = ""

    remove_negation = False
    basic_to_third_person = False
    can_change_basic_form = True
    can_change_pp_modal = True
    can_change_future = True
    basic_to_past = False

    # print("----------------")
    # print(sample_pos_tag)
    for i in range(0, len(sample_pos_tag)-1):
        token = sample_pos_tag[i]
        if (
            remove_negation == False
            and basic_to_third_person == False
            and can_change_basic_form == True
            and basic_to_past == False
            and can_change_pp_modal == True
        ):
            if (
                token[0] == "has" and sample_pos_tag[i + 1][1] == "VBN"
            ):  # ----- third person singular present perfect
                verb = token[0]
                verb = verb + " not"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif (
                token[0] == "have" and sample_pos_tag[i + 1][1] == "VBN"
            ):  # ----- present perfect
                verb = token[0]
                verb = verb + " not"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif (
                token[0] == "had" and sample_pos_tag[i + 1][1] == "VBN"
            ):  # ----- past perfect
                verb = token[0]
                verb = verb + " not"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif token[0] == "does" and sample_pos_tag[i + 1][0] in (
                "not",
                "n't",
            ):  # ----- negative, third person present simple
                remove_negation = True
                basic_to_third_person = True
                Perturbed_sample += ""
                is_sample_perturbed = True

            elif token[0] == "do" and sample_pos_tag[i + 1][0] in (
                "not",
                "n't",
            ):  # ----- negative, present simple
                remove_negation = True
                can_change_basic_form = False
                Perturbed_sample += ""
                is_sample_perturbed = True

            elif token[0] == "did" and sample_pos_tag[i + 1][0] in (
                "not",
                "n't",
            ):  # ----- negative, past simple
                remove_negation = True
                can_change_basic_form = True
                basic_to_past = True
                Perturbed_sample += ""
                is_sample_perturbed = True

            elif token[0] == "has" and sample_pos_tag[i + 1][0] in (
                "not",
                "n't",
            ):  # ----- negative, third person present perfect
                remove_negation = True
                can_change_pp_modal = False
                Perturbed_sample += "has" + " "
                is_sample_perturbed = True

            elif token[0] == "have" and sample_pos_tag[i + 1][0] in (
                "not",
                "n't",
            ):  # ----- negative, present perfect
                remove_negation = True
                can_change_pp_modal = False
                Perturbed_sample += "have" + " "
                is_sample_perturbed = True

            elif token[0] == "had" and sample_pos_tag[i + 1][0] in (
                "not",
                "n't",
            ):  # ----- negative, present perfect
                remove_negation = True
                can_change_pp_modal = False
                Perturbed_sample += "had" + " "
                is_sample_perturbed = True

            elif len(token) >= 2:
                if (
                token[1] == "MD"
                and sample_pos_tag[i + 1][0]
                in (
                    "not",
                    "n't",
                )
                        ):  # ----- negative, future and modal verbs
                    verb = token[0]
                    if verb == "ca":
                        verb = "can"
                    remove_negation = True
                    can_change_pp_modal = False
                    Perturbed_sample += verb + " "
                    is_sample_perturbed = True

            elif token[0] in ("is", "are", "was", "were", "am") and sample_pos_tag[
                i + 1
            ][0] in (
                "not",
                "n't",
            ):  # ----- negative, to be present and past, continuous present and past
                verb = token[0]
                remove_negation = True
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif len(token) >= 2:
                if token[1] == "MD":  # ----- future and modal verbs
                    verb = token[0]
                    if verb == "can" or verb == "Can":
                        verb = verb + "not"
                    else:
                        verb = verb + " not"
                    Perturbed_sample += verb + " "
                    is_sample_perturbed = True

            elif token[0] in (
                "is",
                "are",
                "was",
                "were",
                "am",
            ):  # ----- to be present and past, continuous present and past
                verb = token[0]
                verb = verb + " not"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif len(token) >= 2:
                if token[1] == "VBZ":  # ----- third person singular present
                    verb = token[0]
                    length = len(verb)
                    if verb == "has":
                        verb = "does not " + "have"
                    elif verb[length - 3 :] == "oes":
                        verb = "does not " + verb[: length - 2]
                    elif verb[length - 4 :] == "ches":
                        verb = "does not " + verb[: length - 2]
                    elif verb[length - 3 :] == "ses":
                        verb = "does not " + verb[: length - 2]
                    elif verb[length - 4 :] == "shes":
                        verb = "does not " + verb[: length - 2]
                    elif verb[length - 3 :] == "xes":
                        verb = "does not " + verb[: length - 2]
                    elif verb[length - 3 :] == "zes":
                        verb = "does not " + verb[: length - 2]
                    elif verb[length - 3 :] == "ies":
                        verb = "does not " + verb[: length - 3] + "y"
                    else:
                        verb = "does not " + verb[: length - 1]
                    Perturbed_sample += verb + " "
                    is_sample_perturbed = True

            elif len(token) >= 2:
                if token[1] == "VBP":  # ----- basic form present
                    verb = token[0]
                    length = len(verb)
                    verb = "do not " + verb
                    Perturbed_sample += verb + " "
                    is_sample_perturbed = True

            elif len(token) >= 2:
                if token[1] == "VBD":  # ----- past
                    verb = token[0]
                    verb = "did not " + WordNetLemmatizer().lemmatize(verb, "v")
                    Perturbed_sample += verb + " "
                    is_sample_perturbed = True

            else:
                Perturbed_sample += token[0] + " "

        elif remove_negation == True:
            if token[0] in ("not", "n't"):  # ----- removing not after do or does
                Perturbed_sample += ""
                remove_negation = False

        elif basic_to_third_person == True and can_change_basic_form == True:
            if len(token) >=2:
                if token[1] == "VB":  # ----- converting basic form to third person singular
                    verb = token[0]
                    length = len(verb)
                    if verb == "have":
                        verb = "has"
                    elif verb == "go":
                        verb = "goes"
                    elif verb[length - 2 :] == "ch":
                        verb = verb + "es"
                    elif verb[length - 1 :] == "s":
                        verb = verb + "es"
                    elif verb[length - 2 :] == "sh":
                        verb = verb + "es"
                    elif verb[length - 1 :] == "x":
                        verb = verb + "es"
                    elif verb[length - 1 :] == "z":
                        verb = verb + "es"
                    elif verb[length - 1 :] == "y":
                        verb = verb[: length - 1] + "ies"
                    else:
                        verb = verb + "s"

                    Perturbed_sample += verb + " "
                    basic_to_third_person = False

        elif can_change_basic_form == False:
            if len(token) >= 2:
                if token[1] == "VB":  # ----- do not change basic form
                    verb = token[0]
                    Perturbed_sample += verb + " "
                    can_change_basic_form = True

        elif can_change_basic_form == True and basic_to_past == True:
            if len(token) >= 2:
                if token[1] == "VB":  # ----- change basic form to past
                    verb = token[0]
                    past_tense = ""

                    default_conjugator = mlconjug3.Conjugator(language="en")
                    past_verb = default_conjugator.conjugate(verb)
                    all_conjugates = past_verb.iterate()

                    for j in range(0, len(all_conjugates)):
                        if all_conjugates[j][1] == "indicative past tense":
                            past_tense = all_conjugates[j][3]

                    Perturbed_sample += past_tense + " "
                    basic_to_past = False

        elif can_change_pp_modal == False:
            if len(token) >= 2:
                if token[1] in ("VBN", "VB"):  # ----- keep past participle or modal or will
                    verb = token[0]
                    Perturbed_sample += verb + " "
                    can_change_pp_modal = True

    # print('Perturbed sample:', Perturbed_sample)
    return Perturbed_sample


