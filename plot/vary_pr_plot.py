"""
======================================================================
VARY_PR_PLOT ---

Plot the results varying the poisoning rate.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 13 December 2024
======================================================================
"""

# ------------------------ Code --------------------------------------
import json
from matplotlib import pyplot as plt
import matplotlib
from collections import OrderedDict


def main1():
    # x_label_ls = ["2", "1", "1/2", "1/4", "1/8", "1/16", "1/32"]
    # x_label_ls.reverse()
    # x_key_ls = ["2", "1", "0.5", "0.25", "0.12", "0.06", "0.03"]
    # x_key_ls.reverse()
    # x_ls = [float(xx) for xx in x_key_ls]

    # x_ls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,]
    x_ls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,]

    # from collections import OrderedDict
    with open("./vary_pr.json", 'r', encoding='utf8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    overall_data = data[1]

    row_ls = [
        "sst2",
        "cola",
        "qnli",
        # "qqp",
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
        "0",
        "1",
    ]

    method_label_dict = {
        "0": "Full Fine-tuning",
        "1": "LoRA",
    }

    fig, axs = plt.subplots(3, 4, figsize=(20, 10.5))

    font_size = 21
    a = 0.2
    lw = 1.7
    marker = {
        method_ls[0]: "o",
        method_ls[1]: "s",
    }
    model_color_dict = {
        method_ls[0]: "#eb3b5a",
        method_ls[1]: "#3867d6",
    }
    # model_color_dict2=model_color_dict
    model_color_dict2 = {
        method_ls[0]: "#f78fb3",
        method_ls[1]: "#778beb",
    }

    model_line_style = {
        method_ls[0]: "-",
        method_ls[1]: "-.",
    }
    data = overall_data

    for i_row, row in enumerate(row_ls):
        for i_col, col in enumerate(column_ls):
            for method in method_ls:
                # print("data[method]",data[method])
                yls_average = [
                    data[row]["-1"]["y"]["1.0"]["google-bert/bert-large-uncased"][str(
                        x)][method]["mean"][i_col]
                    for x in x_ls
                ]
                yls_std = [
                    data[row]["-1"]["y"]["1.0"]["google-bert/bert-large-uncased"][str(
                        x)][method]["std"][i_col]
                    for x in x_ls
                ]
                yls_max = [
                    data[row]["-1"]["y"]["1.0"]["google-bert/bert-large-uncased"][str(
                        x)][method]["mean"][i_col]
                    +
                    data[row]["-1"]["y"]["1.0"]["google-bert/bert-large-uncased"][str(
                        x)][method]["std"][i_col]
                    for x in x_ls
                ]
                yls_min = [
                    data[row]["-1"]["y"]["1.0"]["google-bert/bert-large-uncased"][str(
                        x)][method]["mean"][i_col]
                    -
                    data[row]["-1"]["y"]["1.0"]["google-bert/bert-large-uncased"][str(
                        x)][method]["std"][i_col]
                    for x in x_ls
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

                axs[i_row][i_col].fill_between(x_ls,
                                               yls_min, yls_max,
                                               alpha=a,
                                               linewidth=0.,
                                               # alpha=1.0,
                                               color=model_color_dict2[method])

            axs[i_row][i_col].set_xlabel(
                "Poisoning Rate", fontsize=font_size)
            axs[i_row][i_col].set_title(
                row_dict[row], fontsize=font_size)
            axs[i_row][i_col].set_ylabel(col, fontsize=font_size - 5)
            axs[i_row][i_col].set_xticks(
                x_ls, x_ls,
                # rotation=48,
                size=font_size - 4
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
        loc=(-2.05, 5.00),
        prop=font1,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.9)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./varypr.pdf", pad_inches=0.1)
    print("Save done.")


if __name__ == "__main__":
    main1()
