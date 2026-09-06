from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from skimage.segmentation import slic, mark_boundaries

from image_utils import (
    build_image_similarity_matrix,
    compute_superpixel_features,
    scaled_sigma_position,
)
from segmentation import baseline_kmeans_segmentation
from spectral import spectral_clustering_from_similarity


def plot_segmentation_results(all_results, k):
    """
    Visualizes BSDS500 results for a selected number of clusters.

    A single human annotation is shown only as a representative visual example;
    the displayed ARI values are mean ± standard deviation across all annotators.
    """
    results = all_results[k]

    _, axes = plt.subplots(
        len(results),
        4,
        figsize=(16, 4 * len(results)),
        squeeze=False,
    )

    for row, (name, result) in enumerate(results):
        axes[row, 0].imshow(result['image'])
        axes[row, 0].set_title(f'{name} — originalna slika')

        axes[row, 1].imshow(result['ground_truth'], cmap='viridis')
        axes[row, 1].set_title(
            f'Primer ground truth-a\n(1 od {len(result["ground_truths"])} anotatora)'
        )

        axes[row, 2].imshow(result['segmented_baseline'], cmap='viridis')
        axes[row, 2].set_title(
            f'Baseline\nARI={result["ari_baseline_mean"]:.3f} '
            f'± {result["ari_baseline_std"]:.3f}'
        )

        axes[row, 3].imshow(result['segmented_spectral'], cmap='viridis')
        axes[row, 3].set_title(
            f'Spektralno\nARI={result["ari_spectral_mean"]:.3f} '
            f'± {result["ari_spectral_std"]:.3f}'
        )

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_ari_comparison(all_results, k):
    """
    Compares mean ARI for both methods, with standard deviation across
    annotators shown as error bars.
    """
    results = all_results[k]

    image_labels = [name for name, _ in results]

    spectral_means = np.array([
        r['ari_spectral_mean'] for _, r in results
    ])
    spectral_stds = np.array([
        r['ari_spectral_std'] for _, r in results
    ])

    baseline_means = np.array([
        r['ari_baseline_mean'] for _, r in results
    ])
    baseline_stds = np.array([
        r['ari_baseline_std'] for _, r in results
    ])

    labels = image_labels
    spectral_vals = spectral_means
    baseline_vals = baseline_means

    spectral_err = spectral_stds
    baseline_err = baseline_stds

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        yerr=baseline_err,
        capsize=4,
        label='Baseline (K-means)',
        color='#7570b3',
        edgecolor='white',
        linewidth=0.8,
    )

    bars2 = ax.bar(
        x + width / 2,
        spectral_vals,
        width,
        yerr=spectral_err,
        capsize=4,
        label='Spektralno klasterovanje',
        color='#1b9e77',
        edgecolor='white',
        linewidth=0.8,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5 if height >= 0 else -14),
                textcoords='offset points',
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=9,
            )

    ax.axhline(
        y=0,
        color='black',
        linewidth=0.8
    )

    ax.set_ylabel('ARI (Adjusted Rand Index)')
    ax.set_title(
        f'Poređenje spektralnog klasterovanja i baseline metode '
        f'na BSDS500 (k={k})'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_segmentations(
    image_path,
    k,
    n_segments=200,
    sigma_color=20,
    position_ratio=0.1,
):
    """
    Runs both segmentation methods on an image and displays them side by side.

    position_ratio defines sigma_position as a fraction of the image diagonal.
    """
    image = io.imread(image_path)

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=10,
        start_label=0,
    )
    colors, positions = compute_superpixel_features(image, segments)
    sigma_position = scaled_sigma_position(image.shape, position_ratio)

    W = build_image_similarity_matrix(
        colors,
        positions,
        sigma_color=sigma_color,
        sigma_position=sigma_position,
    )

    labels_spectral = spectral_clustering_from_similarity(W, k=k, normalized=True)
    labels_baseline = baseline_kmeans_segmentation(colors, positions, k=k)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Originalna slika')
    axes[1].imshow(labels_baseline[segments], cmap='viridis')
    axes[1].set_title('Baseline (K-means)')
    axes[2].imshow(labels_spectral[segments], cmap='viridis')
    axes[2].set_title('Spektralno klasterovanje')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def analyze_image(
    image_path,
    n_segments=200,
    compactness=10,
    k_values=[2, 3, 4, 5],
    position_ratio=0.1,
):
    """
    Runs the image pipeline and displays a k-value comparison grid.
    """
    image = io.imread(image_path)
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=0)
    colors, positions = compute_superpixel_features(image, segments)

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Originalna slika')
    axes[0].axis('off')
    axes[1].imshow(mark_boundaries(image, segments))
    axes[1].set_title(f'SLIC (n_segments={n_segments})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    run_segmentation_grid(
        segments,
        colors,
        positions,
        param_name='k',
        param_values=k_values,
        fixed_position_ratio=position_ratio,
    )

    return image, segments, colors, positions


def run_segmentation_grid(
    segments,
    colors,
    positions,
    param_name,
    param_values,
    fixed_k=4,
    fixed_sigma_color=20,
    fixed_position_ratio=0.1,
):
    """
    Runs segmentation while varying one parameter.

    Parameters
    ----------
    param_name : {'k', 'sigma_color', 'position_ratio'}
        Parameter to vary. ``position_ratio`` is converted to sigma_position
        using the diagonal of the image represented by ``segments``.
    """
    _, axes = plt.subplots(1, len(param_values), figsize=(4 * len(param_values), 4))
    axes = np.atleast_1d(axes)

    for ax, value in zip(axes, param_values):
        k = value if param_name == 'k' else fixed_k
        sigma_color = value if param_name == 'sigma_color' else fixed_sigma_color
        position_ratio = value if param_name == 'position_ratio' else fixed_position_ratio
        sigma_position = scaled_sigma_position(segments.shape, position_ratio)

        W = build_image_similarity_matrix(
            colors,
            positions,
            sigma_color=sigma_color,
            sigma_position=sigma_position,
        )
        labels = spectral_clustering_from_similarity(W, k=k, normalized=True)
        segmented = labels[segments]

        ax.imshow(segmented, cmap='viridis')
        if param_name == 'position_ratio':
            ax.set_title(
                f'position_ratio = {value:g}\n'
                f'(σ_position ≈ {sigma_position:.1f}px)'
            )
        else:
            ax.set_title(f'{param_name} = {value}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
