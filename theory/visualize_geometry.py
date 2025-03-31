

from matplotlib import pyplot as plt
import numpy as np
import math

from info_theorey import renyi_entropy_1, renyi_entropy_2


def draw_one_heatmap():
    plt.rcParams["font.size"]=12

    N = 40

    ranks = np.geomspace(4, 256, N)
    scales = np.geomspace(0.01, 1.5, N)

    dimension = 1024

    Z_1 = np.zeros((N, N))
    # Z_2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Z_1[i][j] = renyi_entropy_1(ranks[i], dimension, scales[j])
            # Z_2[i][j] = renyi_entropy_2(ranks[i], dimension, scales[j])

    # print(Z_2.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    cmap = "hot"
    xmin = 4
    xmax = 256
    ymin = 0.001
    ymax = 2

    im1 = ax.imshow(
        Z_1,
        extent=[xmin, xmax, ymin, ymax,],
        aspect="auto",
        origin="lower",
        cmap=cmap)
    ax.set_xscale("log")
    fig.colorbar(im1, ax=ax,label="Value")
    # ax.set_yscale("log")

    ax.set_title("$H_1'$")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Scale of Variance (#$/n_{l-1}$)")

    # plt.gca().xaxis.set_major_formatter("{x:.2f}")

    plt.xticks([4, 16, 32, 64, 128, 256,])
    ax.set_yticks([0.01, 1/3, 0.5, 1.0, 1.5, 2.0])
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{t:.2f}" for t in yticks], rotation=60)

    ax.axhline(y=0.333, color="red", linestyle="--",
               linewidth=2, label="Default")

    # im2 = ax2.imshow(Z_2, cmap=cmap)
    # ax2.set_title("$H_2'$")

    # cbar = fig.colorbar(im2, ax=[ax1, ax2])
    # cbar.set_label("Value")

    # ax.legend()
    # plt.tight_layout()
    # plt.rcParams["font.family"]="sans-serif"
    # plt.rcParams["font.sans-serif"]=["Arial"]

    # plt.savefig("visual_r.pdf")
    plt.savefig("visual_r.png")


def draw_two_heatmap():

    N = 40

    ranks = np.geomspace(4, 256, N)
    scales = np.geomspace(0.001, 2, N)

    dimension = 1024

    Z_1 = np.zeros((N, N))
    Z_2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Z_1[i][j] = renyi_entropy_1(ranks[i], dimension, scales[j])
            Z_2[i][j] = renyi_entropy_2(ranks[i], dimension, scales[j])

    # print(Z_2.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    cmap = "plasma"

    im1 = ax1.imshow(Z_1, cmap=cmap)
    ax1.set_title("$H_1'$")
    im2 = ax2.imshow(Z_2, cmap=cmap)
    ax2.set_title("$H_2'$")

    # cbar = fig.colorbar(im2, ax=[ax1, ax2])
    # cbar.set_label("Value")

    # 调整布局
    plt.tight_layout()
    plt.savefig("visual_r.pdf")


if __name__ == "__main__":
    # draw_two_heatmap()
    draw_one_heatmap()
