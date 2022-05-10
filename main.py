import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import processing
import os
import argparse


def OmicIntegration(df1, df2):
    """

    :param df1: (averaged) protein values dataframe
    :param df2: (averaged) RNA values dataframe
    :return: UMAP for df1, df2 and their joint embedding
    """

    umapModel = umap.UMAP(n_components=2, min_dist=0.01, n_neighbors=30, n_epochs=500, verbose=2)

    # Fit the averaged values of protein
    umapProt = umapModel.fit_transform(df1.to_numpy())

    # Fit averaged values of RNA
    umapRNA = umapModel.fit_transform(df2.to_numpy())
    fitP = umapModel.fit(df1.to_numpy())
    fitR = umapModel.fit(df2.to_numpy())
    intersection = umap.general_simplicial_set_intersection(fitP.graph_, fitR.graph_, weight=0.5)
    intersection = umap.reset_local_connectivity(intersection)
    embedding, aux_data = umap.simplicial_set_embedding(
        data=fitP._raw_data,
        graph=intersection,
        n_components=fitP.n_components,
        initial_alpha=fitP.learning_rate,
        a=fitP._a,
        b=fitP._b,
        gamma=fitP.repulsion_strength,
        negative_sample_rate=fitP.negative_sample_rate,
        n_epochs=100,
        init='random',
        random_state=np.random,
        metric=fitP.metric,
        metric_kwds=fitP._metric_kwds,
        densmap=False,
        densmap_kwds=False,
        output_dens=False
    )
    umapJoint = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=df1.index)

    return umapProt, umapRNA, umapJoint


def PlotJointEmbedding(umapProt, umapRNA, umapJoint):
    """

    :param umapRNA: umap of RNA dataset
    :param umapProt: umap of Protein dataset
    :param umapJoint: umap of the joint embeddging
    :return: None; save figure
    """

    def add_repeated(ax):
        ax.set_xlabel('UMAP1', fontsize=12)
        ax.set_ylabel('UMAP2', fontsize=12)
        ax.grid(b=None)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 16))
    ax1.scatter(umapProt[:, 0], umapProt[:, 1], c='orange', s=15)
    ax1.set_title('UMAP Proteomics Only', fontsize=16)

    add_repeated(ax1)

    ax2.scatter(umapRNA[:, 0], umapRNA[:, 1], c='darkred', s=15)
    ax2.set_title('UMAP RNA-Seq Only', fontsize=16)
    add_repeated(ax2)

    ax3.scatter(umapJoint['UMAP1'], umapJoint['UMAP2'], c='blue', s=15)
    ax3.set_title('UMAP Joint Proteomics and RNA-Seq', fontsize=16)
    add_repeated(ax3)
    plt.savefig("jointUmap.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct joint UMAP embedding')
    parser.add_argument('--data_dir', default='data', help='directory with data files')
    parser.add_argument(
        '--file1',
        default='Ecoli_MS_NormTotal_Fusion_160519_70_77_Proteins.txt.gz?raw=true',
        help='protein levels file'
    )
    parser.add_argument('--file2', default='GSE92601_Escherichia_coli.Gene.rpkm.txt.gz', help='RNA levels file')
    args = parser.parse_args()
    file1 = os.path.join(args.data_dir, args.file1)
    file2 = os.path.join(args.data_dir, args.file2)
    df1, df2 = processing.process(file1, file2)
    umapProt, umapRNA, umapJoint = OmicIntegration(df1, df2)
    PlotJointEmbedding(umapProt, umapRNA, umapJoint)
