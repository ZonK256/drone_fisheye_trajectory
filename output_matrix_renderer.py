import numpy as np
import matplotlib.pyplot as plt
import os

from main import wektory2D  # import żeby zachować zgodność z nazwami zmiennych

matrix_files = [f for f in os.listdir() if "matrix" in f and f.endswith(".npy")]
TOTAL_MATRICES = len(matrix_files)
PLOT_STEPS = 4

# matrix = np.zeros((9, 9, 9), dtype=np.int32)

# matrix[4, 4, 0] = 1
# matrix[4, 4, 1] = 1
# matrix[4, 4, 2] = 1
# matrix[4, 4, 3] = 1
# matrix[4, 4, 4] = 1
# matrix[4, 4, 5] = 1
# matrix[4, 4, 6] = 1
# matrix[4, 4, 7] = 1
# matrix[4, 4, 8] = 1


def plot_matricies(matrix, title, filename):
    fig = plt.figure(figsize=(5, 6))
    ax_array = fig.subplots(3, 3, sharex=True, sharey=True)
    Rpix = matrix.shape[0] // 2

    for i, vector in enumerate(wektory2D):
        sub_matrix = matrix[:, :, i]
        for x in range(sub_matrix.shape[0]):
            for y in range(sub_matrix.shape[1]):
                if (x - Rpix) ** 2 + (y - Rpix) ** 2 > Rpix**2:
                    sub_matrix[x, y] = np.nan

        ax_array[i // 3, i % 3].imshow(
            sub_matrix,
            cmap="plasma",
            interpolation="nearest",
        )

        ax_array[i // 3, i % 3].set_title(
            f"[{vector[0]}, {vector[1]}]", fontsize=8, pad=2
        )

    fig.suptitle(title, fontsize=16)

    cbar = fig.colorbar(
        ax_array[2, 2].imshow(matrix[:, :, 8], cmap="plasma"),
        ax=ax_array,
        orientation="horizontal",
    )

    fig.subplots_adjust(wspace=0.1, hspace=0.15, bottom=0.20)
    cbar.ax.set_position([0.05, 0.01, 0.9, 0.05])
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("bottom")

    # fig.show()

    fig.savefig(filename)


def generate_reference_matrix() -> np.ndarray:
    matrix = np.load(matrix_files[0])

    for i in range(1, len(matrix_files)):
        matrix += np.load(matrix_files[i])

        sum_matrix = np.sum(matrix)
        reference_matricies = np.zeros(matrix.shape, dtype=np.float32)
        reference_matricies = matrix / sum_matrix

    print(
        f"Summed up {TOTAL_MATRICES} matrices for reference, total sum: {sum_matrix:_}"
    )
    return reference_matricies


XLIMS = [
    (-1e-9, 1e-9),
    (-1e-8, 1e-8),
    (-1e-9, 1e-9),
    (-1e-8, 1e-8),
    (-2e-6, 2e-6),
    (-1e-8, 1e-8),
    (-1e-9, 1e-9),
    (-1e-8, 1e-8),
    (-1e-9, 1e-9),
]


def plot_histograms(diff_matrices, percentages, title, filename):
    fig, axes = plt.subplots(9, 4, sharey=True, figsize=(12, 18))

    fig.subplots_adjust(hspace=0.5, wspace=0.5, top=0.9)
    for j, percentage in enumerate(percentages):
        axes[0, j].annotate(
            f"{percentage}% matrices",
            xy=(0.5, 1.1),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=15,
            weight="bold",
        )

    for i, vector in enumerate(wektory2D):
        for j, diff_matrix in enumerate(diff_matrices):
            matrix = diff_matrix[:, :, i]
            cleaned_matrix = matrix[~np.isnan(matrix)]
            ax = axes[i, j]
            ax.set_xlim(XLIMS[i])
            ax.hist(cleaned_matrix, bins=20, edgecolor="black")
            if j == 0:
                ax.annotate(
                    f"[{vector[0]}, {vector[1]}]",
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    fontsize=15,
                    ha="right",
                    va="center",
                    rotation=0,
                )
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y / len(cleaned_matrix):.2}")
            )

    fig.suptitle(title, fontsize=20, y=0.95)
    fig.savefig(filename)


if __name__ == "__main__":
    reference_matrices = generate_reference_matrix()
    plot_matricies(reference_matrices, "Reference matrix", "reference_matrix.pdf")

    diff_matrices = []
    percentages = []
    for i in range(PLOT_STEPS, TOTAL_MATRICES, PLOT_STEPS):
        used_matrices_percentage = int(i / TOTAL_MATRICES * 100)
        percentages.append(used_matrices_percentage)

        chunk_matrices = np.load(matrix_files[0])
        for j in range(1, i):
            chunk_matrices += np.load(matrix_files[j])

        sum_chunk_matrices = np.sum(chunk_matrices)
        print(f"Summed up {i} matrices, total sum: {sum_chunk_matrices:_}")

        chunk_matrices = chunk_matrices / sum_chunk_matrices
        plot_matricies(
            chunk_matrices,
            f"{used_matrices_percentage}% matrices summed up",
            f"chunk_matrix_{i}_{TOTAL_MATRICES}.pdf",
        )

        diff_matrix = reference_matrices - chunk_matrices
        diff_matrices.append(diff_matrix)

    plot_histograms(
        diff_matrices,
        percentages,
        "Histograms of Differences",
        "combined_histograms.pdf",
    )

    # plt.show()

# dostępne cmap: 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
