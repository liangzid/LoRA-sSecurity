"""
======================================================================
VARY_INIT_VAR ---

Plot the curve varying the variance of the initialization.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 11 November 2024
======================================================================
"""

# ------------------------ Code --------------------------------------

# normal import
import json

# from typing import List, Tuple, Dict
# import random
# from pprint import pprint as ppp

# from math import exp
# import pickle
# import torch.nn.functional as F
# import torch

# from transformers import AutoModelForCausalLM
# from transformers import AutoModelForSequenceClassification
# from transformers import AutoModelForTokenClassification
# from transformers import AutoTokenizer, AutoConfig, AutoModel

# import sys
# import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# from sklearn import metrics
# import sklearn
from collections import OrderedDict


def parse_json_file(
    pth="../varyvaroverall.json",
):
    with open(pth, "r", encoding="utf8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    res_5times_dict = data[1]

    return res_5times_dict


def main1():
    # x_label_ls = ["2", "1", "1/2", "1/4", "1/8", "1/16", "1/32"]
    # x_label_ls.reverse() # x_key_ls = ["2", "1", "0.5", "0.25", "0.12", "0.06", "0.03"]
    # x_key_ls.reverse()
    # x_ls = [float(xx) for xx in x_key_ls]

    x_label_ls = [
        "2.0", "1.5", "1.0", "0.667", "0.333", "0.1", "0.001", "0.0001"
    ]
    x_key_ls = [
        # "1.2", "1.0", "0.8", "0.6", "0.4", "0.333", "0.2", "0.001",
        "2.0", "1.5", "1.0", "0.667", "0.333", "0.1", "0.001", "0.0001"
    ]
    x_label_ls.reverse()
    x_key_ls.reverse()
    x_ls = [float(xx) for xx in x_key_ls]

    ff__clean_ls = {
        "sst2": [
            [0.9263 for x in range(len(x_ls))],
            [0.9144 for x in range(len(x_ls))],
            [0.9554 for x in range(len(x_ls))],
            [0.9290 for x in range(len(x_ls))],
        ],
        "cola": [
            [0.8287 for x in range(len(x_ls))],
            [0.8540 for x in range(len(x_ls))],
            [0.9079 for x in range(len(x_ls))],
            [0.8798 for x in range(len(x_ls))],
        ],
        "qnli": [
            [0.9100 for x in range(len(x_ls))],
            [0.9247 for x in range(len(x_ls))],
            [0.8949 for x in range(len(x_ls))],
            [0.9094 for x in range(len(x_ls))],
        ],
        "qqp": [
            [0.9063 for x in range(len(x_ls))],
            [0.8673 for x in range(len(x_ls))],
            [0.8805 for x in range(len(x_ls))],
            [0.8737 for x in range(len(x_ls))],
        ],
    }

    ff__poison_ls = {
        "sst2": [
            [0.9245 for x in range(len(x_ls))],
            [0.9127 for x in range(len(x_ls))],
            [0.9423 for x in range(len(x_ls))],
            [0.9271 for x in range(len(x_ls))],
        ],
        "cola": [
            [0.8157 for x in range(len(x_ls))],
            [0.8529 for x in range(len(x_ls))],
            [0.8873 for x in range(len(x_ls))],
            [0.8692 for x in range(len(x_ls))],
        ],
        "qnli": [
            [0.8962 for x in range(len(x_ls))],
            [0.9026 for x in range(len(x_ls))],
            [0.8907 for x in range(len(x_ls))],
            [0.8966 for x in range(len(x_ls))],
        ],
        "qqp": [
            [0.8721 for x in range(len(x_ls))],
            [0.8152 for x in range(len(x_ls))],
            [0.8446 for x in range(len(x_ls))],
            [0.8293 for x in range(len(x_ls))],
        ],
    }

    ff__poison_std_ls = {
        "sst2": [
            [0.0064 for x in range(len(x_ls))],
            [0.0166 for x in range(len(x_ls))],
            [0.0150 for x in range(len(x_ls))],
            [0.0059 for x in range(len(x_ls))],
        ],
        "cola": [
            [0.0079 for x in range(len(x_ls))],
            [0.0177 for x in range(len(x_ls))],
            [0.0328 for x in range(len(x_ls))],
            [0.0086 for x in range(len(x_ls))],
        ],
        "qnli": [
            [0.0066 for x in range(len(x_ls))],
            [0.0083 for x in range(len(x_ls))],
            [0.0116 for x in range(len(x_ls))],
            [0.0069 for x in range(len(x_ls))],
        ],
        "qqp": [
            [0.0016 for x in range(len(x_ls))],
            [0.0135 for x in range(len(x_ls))],
            [0.0251 for x in range(len(x_ls))],
            [0.0054 for x in range(len(x_ls))],
        ],
    }

    print("]]))--> Data loading DONE")

    # overall_data = parse_json_file()
    overall_data = parse_json_file("../varyvarovrrall_1229_latest.json")

    print("]]))--> parse DONE")
    row_ls = ["sst2", "cola", "qnli", "qqp"]

    column_ls = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    method_ls = [
        "0.0",
        "0.3",
    ]

    method_label_dict = {
        "0.0": "LoRA (Clean)",
        "0.3": "LoRA (PR=0.3)",
    }

    fig, axs = plt.subplots(4, 4, figsize=(20, 14))
    print("subplot done.")

    font_size = 21
    a = 0.2
    lw = 1.7
    marker = {
        method_ls[0]: "o",
        method_ls[1]: "s",
        # method_ls[2]: "x",
        "FF (clean)": "o",
        "FF (poison)": "o",
    }
    model_color_dict = {
        method_ls[0]: "#eb3b5a",
        method_ls[1]: "#3867d6",
        # method_ls[2]: "#3867d6",
        "FF (clean)": "#eb3b5a",
        "FF (poison)": "#eb3b5a",
    }
    # model_color_dict2=model_color_dict
    model_color_dict2 = {
        method_ls[0]: "#f78fb3",
        method_ls[1]: "#778beb",
        # method_ls[2]: "#778beb",
        "FF (clean)": "#f78fb3",
        "FF (poison)": "#f78fb3",
    }

    model_line_style = {
        method_ls[0]: "-",
        method_ls[1]: "-.",
        # method_ls[2]: "dotted",
        "FF (clean)": "dashed",
        "FF (poison)": "dashed",
    }
    data = overall_data

    # plt.xscale("log")
    for i_row, row in enumerate(row_ls):
        for i_col, col in enumerate(column_ls):
            for method in method_ls:
                # if method == "0.0":
                #     continue

                # print("data[method]",data[method])
                yls_average = [
                    data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method][
                        "1"
                    ]["mean"][i_col]
                    for x in x_key_ls
                ]
                yls_std = [
                    data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method][
                        "1"
                    ]["std"][i_col]
                    for x in x_key_ls
                ]
                yls_max = [
                    data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method][
                        "1"
                    ]["mean"][i_col]
                    + data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][
                        method
                    ]["1"]["std"][i_col]
                    for x in x_key_ls
                ]
                yls_min = [
                    data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method][
                        "1"
                    ]["mean"][i_col]
                    - data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][
                        method
                    ]["1"]["std"][i_col]
                    for x in x_key_ls
                ]

                yff_poison_max_ls = ff__poison_ls[row][i_col] + \
                    ff__poison_std_ls[row][i_col]
                yff_poison_min_ls = [ff__poison_ls[row][i_col][iii] -
                                     ff__poison_std_ls[row][i_col][iii] for iii in range(len(x_ls))]

                axs[i_row][i_col].plot(
                    x_ls,
                    yls_average,
                    # yls_std,
                    label=method_label_dict[method],
                    linewidth=lw,
                    marker=marker[method],
                    markevery=1,
                    markersize=15,
                    markeredgewidth=lw,
                    markerfacecolor="none",
                    alpha=1.0,
                    linestyle=model_line_style[method],
                    color=model_color_dict[method],
                )
                # label_ff_clean="FF (clean)"
                # axs[i_row][i_col].plot(
                #     x_ls,
                #     ff__clean_ls[row][i_col],
                #     label=label_ff_clean,
                #     linewidth=lw,
                #     marker=marker[label_ff_clean],
                #     markevery=1,
                #     markersize=15,
                #     markeredgewidth=lw,
                #     markerfacecolor="none",
                #     alpha=1.0,
                #     linestyle=model_line_style[label_ff_clean],
                #     color=model_color_dict[label_ff_clean],
                # )

                # label_ff_poison = "FF (poison)"
                # axs[i_row][i_col].plot(
                #     x_ls,
                #     ff__poison_ls[row][i_col],
                #     # ff__poison_std_ls[row][i_col],
                #     label=label_ff_poison,
                #     linewidth=lw,
                #     # marker=marker[label_ff_poison],
                #     # markevery=1,
                #     # markersize=15,
                #     # markeredgewidth=lw,
                #     # markerfacecolor="none",
                #     alpha=1.0,
                #     linestyle=model_line_style[label_ff_poison],
                #     color=model_color_dict[label_ff_poison],
                # )

                axs[i_row][i_col].fill_between(x_ls,
                                               yls_min, yls_max,
                                               alpha=a,
                                               linewidth=0.,
                                               # alpha=1.0,
                                               color=model_color_dict2[method])

            axs[i_row][i_col].set_xlabel(
                "Initialzation Vairance", fontsize=font_size)
            axs[i_row][i_col].set_ylabel(col, fontsize=font_size - 5)
            axs[i_row][i_col].set_xticks(
                x_ls, x_label_ls, rotation=48, size=font_size - 4
            )
            axs[i_row][i_col].tick_params(
                axis="y",
                labelsize=font_size - 6,
                rotation=65,
                width=2,
                length=2,
                pad=0,
                direction="in",
                which="both",
            )
            # axs[i_row][i_col].set_xscale("log")

    font1 = {
        "weight": "normal",
        "size": font_size - 1,
    }

    plt.legend(
        loc=(-2.25, 5.70),
        prop=font1,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./varyvar.pdf", pad_inches=0.1)


def main_varyinit_upa_Acc_plot():

    x_label_ls = [
        "2.0", "1.5", "1.0", "0.667", "0.333", "0.1", "0.001", "0.0001",
    ]
    x_key_ls = x_label_ls
    x_label_ls.reverse()
    x_key_ls.reverse()
    x_ls = [float(xx) for xx in x_key_ls]

    overall_data = parse_json_file("../varyvarovrrall_1229_latest.json")
    # overall_data = parse_json_file("../varyrank_new.json")

    overall_data_islora1 = overall_data

    row_ls = [
        "sst2",
        "cola",
        "qnli",
        "qqp",
    ]
    row_dict = {
        "sst2": "SST-2",
        "cola": "COLA",
        "qnli": "QNLI",
        "qqp": "QQP",
    }
    column_ls = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    method_ls = [
        "0.0",
        "0.3",
    ]

    method_label_dict = {
        "0.0": "LoRA (Clean)",
        "0.3": "LoRA (PR=0.3)",
    }

    fig, axs = plt.subplots(1, 4, figsize=(20, 3.85))

    font_size = 21
    a = 0.2
    lw = 1.7
    marker = {
        method_ls[0]: "D",
        method_ls[1]: "s",
    }
    model_color_dict = {
        method_ls[0]: "#C72323",
        method_ls[1]: "#326497",
    }
    # model_color_dict2=model_color_dict
    model_color_dict2 = {
        method_ls[0]: "#FBB7AD",
        method_ls[1]: "#84B2D3",
    }

    model_line_style = {
        method_ls[0]: "dotted",
        method_ls[1]: "-.",
    }
    data = overall_data

    for i_row, row in enumerate(row_ls):
        i_col = 0
        col = "Accuracy"
        for method in method_ls:

            if method == "0":
                data = overall_data
            else:
                data = overall_data_islora1

            yls_average = [
                data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["mean"][i_col]
                for x in x_key_ls
            ]
            yls_std = [
                data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["std"][i_col]
                for x in x_key_ls
            ]
            yls_max = [
                data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["mean"][i_col]
                +
                data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["std"][i_col]
                for x in x_key_ls
            ]
            yls_min = [
                data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["mean"][i_col]
                -
                data[row][x]["y"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["std"][i_col]
                for x in x_key_ls
            ]

            axs[i_row].plot(
                x_ls,
                yls_average,
                # yls_std,
                label=method_label_dict[method],
                linewidth=lw,
                marker=marker[method],
                markevery=1,
                markersize=15,
                markeredgewidth=lw,
                # markerfacecolor="none",
                alpha=1.0,
                linestyle=model_line_style[method],
                color=model_color_dict[method],
            )

            axs[i_row].fill_between(x_ls,
                                    yls_min, yls_max,
                                    alpha=a,
                                    linewidth=0.2,
                                    # alpha=1.0,
                                    color=model_color_dict2[method])

        axs[i_row].set_xlabel(
            "Initialization Variance", fontsize=font_size)
        axs[i_row].set_title(
            row_dict[row], fontsize=font_size)
        axs[i_row].set_ylabel(col, fontsize=font_size - 5)
        axs[i_row].set_xticks(
            [0.1, 0.333, 0.667, 1.0, 1.5, 2.0,],
            [0.1, 0.333, 0.667, 1.0, 1.5, 2.0,],
            rotation=40,
            size=font_size - 4
        )

        axs[i_row].tick_params(
            axis="y",
            labelsize=font_size - 6,
            rotation=65,
            width=2,
            length=2,
            pad=0,
            direction="in",
            which="both",
        )
        # axs[i_row][i_col].set_xscale("log")
        # axs[i_row].set_xscale("log")

    font1 = {
        "weight": "normal",
        "size": font_size - 1,
    }

    plt.legend(
        loc=(-2.25, -0.7),
        prop=font1,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.9)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./varyinit_poison_acc1x4.pdf", pad_inches=0.1)
    print("Save done.")


def main2backdoor():
    x_label_ls = [
        "2.0", "1.5", "1.0", "0.667", "0.333", "0.1", "0.001",  # "0.0001"
    ]
    x_key_ls = [
        # "1.2", "1.0", "0.8", "0.6", "0.4", "0.333", "0.2", "0.001",
        "2.0", "1.5", "1.0", "0.667", "0.333", "0.1", "0.001",  # "0.0001"
    ]
    x_label_ls.reverse()
    x_key_ls.reverse()
    x_ls = [float(xx) for xx in x_key_ls]

    print("]]))--> Data loading DONE")

    overall_data = parse_json_file("../varyvarbackdoor.json",)
    data_poison = parse_json_file("../varyvaroverall.json",)

    print("]]))--> parse DONE")
    row_ls = ["sst2", "cola", "qnli", "qqp"]
    column_ls = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    method_ls = [
        "0.0",
        "0.0015",
    ]

    method_label_dict = {
        "0.0": "LoRA (Clean)",
        "0.0015": "LoRA (PR=0.15%)",
    }

    fig, axs = plt.subplots(4, 4, figsize=(20, 14))
    print("subplot done.")

    font_size = 21
    a = 0.2
    lw = 1.7
    marker = {
        method_ls[0]: "o",
        method_ls[1]: "s",
        # method_ls[2]: "x",
        "FF (clean)": "o",
        "FF (poison)": "o",
    }
    model_color_dict = {
        method_ls[0]: "#eb3b5a",
        method_ls[1]: "#3867d6",
        # method_ls[2]: "#3867d6",
        "FF (clean)": "#eb3b5a",
        "FF (poison)": "#eb3b5a",
    }
    # model_color_dict2=model_color_dict
    model_color_dict2 = {
        method_ls[0]: "#f78fb3",
        method_ls[1]: "#778beb",
        # method_ls[2]: "#778beb",
        "FF (clean)": "#f78fb3",
        "FF (poison)": "#f78fb3",
    }

    model_line_style = {
        method_ls[0]: "-",
        method_ls[1]: "-.",
        # method_ls[2]: "dotted",
        "FF (clean)": "dashed",
        "FF (poison)": "dashed",
    }
    data = overall_data

    # plt.xscale("log")
    for i_row, row in enumerate(row_ls):
        for i_col, col in enumerate(column_ls):
            for method in method_ls:
                # if method == "0.0":
                #     data = data_poison
                #     kw="y"
                # else:
                #     data = overall_data
                #     kw="backdoor-simple"
                kw = "backdoor-simple"

                yls_average = [
                    data[row][x][kw]["1.0"]["google-bert/bert-large-uncased"][method][
                        "1"
                    ]["mean"][i_col]
                    for x in x_key_ls
                ]
                yls_std = [
                    data[row][x][kw]["1.0"]["google-bert/bert-large-uncased"][method][
                        "1"
                    ]["std"][i_col]
                    for x in x_key_ls
                ]
                yls_max = [
                    data[row][x][kw]["1.0"]["google-bert/bert-large-uncased"][method][
                        "1"
                    ]["mean"][i_col]
                    + data[row][x][kw]["1.0"]["google-bert/bert-large-uncased"][
                        method
                    ]["1"]["std"][i_col]
                    for x in x_key_ls
                ]
                yls_min = [
                    data[row][x][kw]["1.0"]["google-bert/bert-large-uncased"][method][
                        "1"
                    ]["mean"][i_col]
                    - data[row][x][kw]["1.0"]["google-bert/bert-large-uncased"][
                        method
                    ]["1"]["std"][i_col]
                    for x in x_key_ls
                ]

                axs[i_row][i_col].plot(
                    x_ls,
                    yls_average,
                    # yls_std,
                    label=method_label_dict[method],
                    linewidth=lw,
                    marker=marker[method],
                    markevery=1,
                    markersize=15,
                    markeredgewidth=lw,
                    markerfacecolor="none",
                    alpha=1.0,
                    linestyle=model_line_style[method],
                    color=model_color_dict[method],
                )
                # label_ff_clean="FF (clean)"
                # axs[i_row][i_col].plot(
                #     x_ls,
                #     ff__clean_ls[row][i_col],
                #     label=label_ff_clean,
                #     linewidth=lw,
                #     marker=marker[label_ff_clean],
                #     markevery=1,
                #     markersize=15,
                #     markeredgewidth=lw,
                #     markerfacecolor="none",
                #     alpha=1.0,
                #     linestyle=model_line_style[label_ff_clean],
                #     color=model_color_dict[label_ff_clean],
                # )

                # label_ff_poison = "FF (poison)"
                # axs[i_row][i_col].plot(
                #     x_ls,
                #     ff__poison_ls[row][i_col],
                #     # ff__poison_std_ls[row][i_col],
                #     label=label_ff_poison,
                #     linewidth=lw,
                #     # marker=marker[label_ff_poison],
                #     # markevery=1,
                #     # markersize=15,
                #     # markeredgewidth=lw,
                #     # markerfacecolor="none",
                #     alpha=1.0,
                #     linestyle=model_line_style[label_ff_poison],
                #     color=model_color_dict[label_ff_poison],
                # )

                axs[i_row][i_col].fill_between(x_ls,
                                               yls_min, yls_max,
                                               alpha=a,
                                               linewidth=0.,
                                               # alpha=1.0,
                                               color=model_color_dict2[method])

            axs[i_row][i_col].set_xlabel(
                "Initialzation Vairance", fontsize=font_size)
            axs[i_row][i_col].set_ylabel(col, fontsize=font_size - 5)
            axs[i_row][i_col].set_xticks(
                x_ls, x_label_ls, rotation=48, size=font_size - 4
            )
            axs[i_row][i_col].tick_params(
                axis="y",
                labelsize=font_size - 6,
                rotation=65,
                width=2,
                length=2,
                pad=0,
                direction="in",
                which="both",
            )
            # axs[i_row][i_col].set_xscale("log")

    font1 = {
        "weight": "normal",
        "size": font_size - 1,
    }

    plt.legend(
        loc=(-2.25, 5.70),
        prop=font1,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./varyvarbackdoor.pdf", pad_inches=0.1)


def main2_varyinit_bpa_Acc_plot():

    x_label_ls = [
        "2.0", "1.5", "1.0", "0.667", "0.333", "0.1", "0.001", "0.0001",
    ]
    x_key_ls = x_label_ls
    x_label_ls.reverse()
    x_key_ls.reverse()
    x_ls = [float(xx) for xx in x_key_ls]

    overall_data = parse_json_file("../varyvarbackdoor.json",)
    data_poison = parse_json_file("../varyvaroverall.json",)

    row_ls = [
        "sst2",
        "cola",
        "qnli",
        "qqp",
    ]
    row_dict = {
        "sst2": "SST-2",
        "cola": "COLA",
        "qnli": "QNLI",
        "qqp": "QQP",
    }
    column_ls = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    method_ls = [
        "0.0",
        "0.0015",
    ]

    method_label_dict = {
        "0.0": "LoRA (Clean)",
        "0.0015": "LoRA (PR=0.15%)",
    }

    fig, axs = plt.subplots(1, 4, figsize=(20, 3.85))

    font_size = 21
    a = 0.2
    lw = 1.7
    marker = {
        method_ls[0]: "D",
        method_ls[1]: "s",
    }
    model_color_dict = {
        method_ls[0]: "#C72323",
        method_ls[1]: "#326497",
    }
    # model_color_dict2=model_color_dict
    model_color_dict2 = {
        method_ls[0]: "#FBB7AD",
        method_ls[1]: "#84B2D3",
    }

    model_line_style = {
        method_ls[0]: "dotted",
        method_ls[1]: "-.",
    }
    data = overall_data

    for i_row, row in enumerate(row_ls):
        i_col = 0
        col = "Accuracy"
        for method in method_ls:

            # if method == "0":
            #     data = overall_data
            # else:
            #     data = overall_data_islora1
            data = overall_data

            yls_average = [
                data[row][x]["backdoor-simple"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["mean"][i_col]
                for x in x_key_ls
            ]
            yls_std = [
                data[row][x]["backdoor-simple"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["std"][i_col]
                for x in x_key_ls
            ]
            yls_max = [
                data[row][x]["backdoor-simple"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["mean"][i_col]
                +
                data[row][x]["backdoor-simple"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["std"][i_col]
                for x in x_key_ls
            ]
            yls_min = [
                data[row][x]["backdoor-simple"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["mean"][i_col]
                -
                data[row][x]["backdoor-simple"]["1.0"]["google-bert/bert-large-uncased"][method]["1"]["std"][i_col]
                for x in x_key_ls
            ]

            axs[i_row].plot(
                x_ls,
                yls_average,
                # yls_std,
                label=method_label_dict[method],
                linewidth=lw,
                marker=marker[method],
                markevery=1,
                markersize=15,
                markeredgewidth=lw,
                # markerfacecolor="none",
                alpha=1.0,
                linestyle=model_line_style[method],
                color=model_color_dict[method],
            )

            axs[i_row].fill_between(x_ls,
                                    yls_min, yls_max,
                                    alpha=a,
                                    linewidth=0.2,
                                    # alpha=1.0,
                                    color=model_color_dict2[method])

        axs[i_row].set_xlabel(
            "Backdoor Poisoning Rate", fontsize=font_size)
        axs[i_row].set_title(
            row_dict[row], fontsize=font_size)
        axs[i_row].set_ylabel(col, fontsize=font_size - 5)
        axs[i_row].set_xticks(
            [0.1, 0.333, 0.667, 1.0, 1.5, 2.0,],
            [0.1, 0.333, 0.667, 1.0, 1.5, 2.0,],
            # x_ls, x_ls,
            rotation=40,
            size=font_size - 4
        )
        axs[i_row].tick_params(
            axis="y",
            labelsize=font_size - 6,
            rotation=65,
            width=2,
            length=2,
            pad=0,
            direction="in",
            which="both",
        )
        # axs[i_row][i_col].set_xscale("log")
        # axs[i_row].set_xscale("log")

    font1 = {
        "weight": "normal",
        "size": font_size - 1,
    }

    plt.legend(
        loc=(-2.25, -0.7),
        prop=font1,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.9)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./varyinit_backdoor_acc1x4.pdf", pad_inches=0.1)
    print("Save done.")


# running entry
if __name__ == "__main__":
    # main1()
    # main2backdoor()
    main_varyinit_upa_Acc_plot()
    main2_varyinit_bpa_Acc_plot()
    print("EVERYTHING DONE.")
