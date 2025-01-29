

# ------------------------ Code --------------------------------------
import json
from matplotlib import pyplot as plt
import matplotlib
from collections import OrderedDict


from vary_rank_plot import parse_json_file


def main_plot1():

    fig, axs = plt.subplots(1, 4, figsize=(20, 3.75))

    x_label_ls = ["4", "8", "16", "32", "64", "128", "256", "512"]
    x_key_ls = x_label_ls
    x_ls = [float(xx) for xx in x_key_ls]

    data = parse_json_file("../varyrank_new.json")

    row_ls = ["sst2", "qnli",]
    row_dict = {
        "sst2": "SST-2 (Posioning PR=0.3)",
        "cola": "COLA",
        "qnli": "QNLI (Posioning PR=0.3)",
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
        "0.3": "LoRA (Poisoned)",
    }

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

    for i_row, row in enumerate(row_ls):
        i_col = 0
        col = "Accuracy"
        for method in method_ls:

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
            "Rank", fontsize=font_size-5)
        axs[i_row].set_title(
            row_dict[row], fontsize=font_size-5)
        axs[i_row].set_ylabel(col, fontsize=font_size - 5)
        axs[i_row].set_xticks(
            x_ls, x_ls,
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
        axs[i_row].set_xscale("log")

    font1 = {
        "weight": "normal",
        "size": font_size - 1,
    }

    x_label_ls = ["4", "8", "16", "32", "64", "128", "256", "512"]
    x_key_ls = x_label_ls
    x_ls = [float(xx) for xx in x_key_ls]

    overall_data = parse_json_file("../varyrankonbackdoor.json")

    row_ls = ["sst2", "qnli",
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

    x_label_ls = ["4", "8", "16", "32", "64", "128", "256"]
    x_key_ls = x_label_ls
    x_ls = [float(xx) for xx in x_key_ls]

    data = parse_json_file("../varyrankonbackdoor.json")

    overall_data_islora1 = overall_data

    row_ls = [
        "sst2",
        "qnli",
    ]
    row_dict = {
        "sst2": "SST-2 (Backdoor PR=0.15%)",
        "cola": "COLA",
        "qnli": "QNLI (Backdoor PR=0.15%)",
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
        "0.0015": "LoRA (Poisoned)",
    }

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
        i_row += 2
        i_col = 0
        col = "Accuracy"
        for method in method_ls:

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
            "Rank", fontsize=font_size-5)
        axs[i_row].set_title(
            row_dict[row], fontsize=font_size-5)
        axs[i_row].set_ylabel(col, fontsize=font_size - 5)
        axs[i_row].set_xticks(
            x_ls, x_ls,
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
        axs[i_row].set_xscale("log")

    font1 = {
        "weight": "normal",
        "size": font_size - 1,
    }

    plt.legend(
        loc=(-2.45, -0.7),
        prop=font1,
        ncol=6,
        frameon=False,
        handletextpad=0.0,
        handlelength=1.2,
    )  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.9)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./hyprid_varyrank_acc1x4.pdf", pad_inches=0.1)


if __name__ == "__main__":
    main_plot1()
