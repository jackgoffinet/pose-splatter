"""
Calculate and plot a visual embedding of the animal pose.

"""
__date__ = "January 2025"

from apca.models import AAPCA
import argparse
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score

from src.config_utils import Config

PRE_PCA_COMPONENTS = 2000
PCA_COMPONENTS = 50


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script for the model")
    parser.add_argument("config", type=str, help="Path to the config JSON file")
    args = parser.parse_args()
    
    config = Config(args.config)

    embed = np.load(config.feature_fn)

    embed = embed.reshape(len(embed), -1)
    print("features:", embed.shape)

    d = np.load(config.center_rotation_fn)
    angles = d["angles"]
    centers = d["centers"]
    angles2 = np.stack(
        [
            np.cos(angles),
            np.sin(angles),
        ],
        axis=1,
    )

    embed -= np.mean(embed, axis=0, keepdims=True)

    print("Doing Pre-PCA...")
    pca = PCA(PRE_PCA_COMPONENTS, random_state=42)
    embed = pca.fit_transform(embed)

    joblib.dump(pca, os.path.join(config.project_directory, 'pca_model.joblib'))

    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.savefig("variance.pdf")
    # plt.close("all")

    print("Doing adversarial PCA...")
    aapca = AAPCA(PCA_COMPONENTS, mu=1e2, pow_iter=20, random_state=42) # NOTE: adjust mu for proper regularization
    temp_embed = aapca.fit_transform(embed, angles2)
    _, rec_angles = aapca.reconstruct(embed, angles2)
    print("r2", r2_score(angles2, rec_angles))
    # quit()
    embed = temp_embed

    joblib.dump(aapca, os.path.join(config.project_directory, 'aapca_model.joblib'))

    # Save.
    np.save(os.path.join(config.project_directory, "embedding.npy"), embed)

    print("Doing t-SNE...")
    embed = TSNE(random_state=42).fit_transform(embed)

    print("Plotting...")
    np.random.seed(42)
    perm = np.random.permutation(len(embed))
    time = np.linspace(0,1,len(embed))
    angles, centers, embed = angles[perm], centers[perm], embed[perm]
    time = time[perm]
    
    # Plot by angle.
    angles = (angles % (2 * np.pi) / (2 * np.pi))
    centers = centers[:,-1]
    centers = centers - np.min(centers)
    centers = (centers / np.quantile(centers, 0.98)).clip(0,1)
    cmap = matplotlib.colormaps['viridis']
    
    _, axarr = plt.subplots(ncols=3, figsize=(10,5))
    color_bys = [angles, centers, time]
    cmaps = [matplotlib.colormaps['hsv'], matplotlib.colormaps['viridis'], matplotlib.colormaps['viridis']]
    for ax, colors, cmap in zip(axarr, color_bys, cmaps):
        plt.sca(ax)
        plt.scatter(embed[:,0], embed[:,1], c=cmap(colors), s=2.0, alpha=0.4, edgecolors=None)
        ax.set_aspect("equal")
        plt.axis("off")
    axarr[0].set_title("Angle")
    axarr[1].set_title("Height")
    axarr[2].set_title("Time")
    plt.savefig(os.path.join(config.project_directory, "tsne.pdf"))


###