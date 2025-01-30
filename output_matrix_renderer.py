import numpy as np
import matplotlib.pyplot as plt
import os

from main import wektory2D  # import żeby zachować zgodność z nazwami zmiennych

matrix_files = [f for f in os.listdir() if "matrix" in f and f.endswith(".npy")]
print(matrix_files)

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


matrix = np.load(matrix_files[0])

# normalize 9 matrices to the total sum of 1
sum_matrix = np.sum(matrix)
normalized_matricies = np.zeros(matrix.shape, dtype=np.float32)
normalized_matricies = matrix / sum_matrix

Rpix = matrix[:, :, 0].shape[0] // 2

fig = plt.figure()
# plot all 9 matrices in one figure
ax_array = fig.subplots(3, 3, sharex=True, sharey=True)


for i in range(9):
    # if point is outside of the circle, remove it (np.nan)
    # circle is defined by center (Rpix, Rpix) and radius Rpix

    sub_matrix = normalized_matricies[:, :, i]
    for x in range(sub_matrix.shape[0]):
        for y in range(sub_matrix.shape[1]):
            if (x - Rpix) ** 2 + (y - Rpix) ** 2 > Rpix**2:
                sub_matrix[x, y] = np.nan

    # print(i // 3, i % 3, sub_matrix)

    ax_array[i // 3, i % 3].imshow(
        sub_matrix,
        cmap="plasma",
        interpolation="nearest",
    )

# add colorbar to the bottom of the figure
cbar = fig.colorbar(
    ax_array[0, 0].imshow(normalized_matricies[:, :, 0], cmap="plasma"),
    ax=ax_array,
    orientation="horizontal",
)

plt.show()


# dostępne cmap: 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
