import pandas as pd
import numpy as np

protein_file = 'Ecoli_MS_NormTotal_Fusion_160519_70_77_Proteins.txt.gz?raw=true'
RNA_file = 'GSE92601_Escherichia_coli.Gene.rpkm.txt.gz'


def get_gene_name(descr):
    try:
        gn = descr.split('GN=')[1]
        gn = gn.split(' ')[0]
    except:
        gn = ''
    return gn


def max_LogFC_avg_groups(inDF):
    # difference between average values for each of the conditions
    replArrays = [inDF.iloc[:, :2], inDF.iloc[:, 2:4], inDF.iloc[:, 4:6], inDF.iloc[:, 6:8]]
    logMean = np.array([np.mean(a, 1) for a in replArrays]).T
    outMean = pd.DataFrame(logMean, index=inDF.index, columns=['ORF_avg', 'ORFIPTG_avg', 'SVI_avg', 'SVIPTG_avg'])
    maxDiff = lambda x: max(x) - min(x)
    outMean['MaxLog2AVGFC'] = [maxDiff(row) for _, row in outMean.iterrows()]
    return outMean


def replicate_LogFC(inDF):
    # calculate absolute difference between replicates for each gene
    replArrays = [inDF.iloc[:, :2], inDF.iloc[:, 2:4], inDF.iloc[:, 4:6], inDF.iloc[:, 6:8]]
    concatArr = np.concatenate(replArrays, axis=0)
    # Absolute difference between the Log-values here gives the LogFC between replicates
    diffArray = np.abs(concatArr[:, :1] - concatArr[:, 1:])
    return diffArray.reshape(1, -1)[0]


def process(protein_file, RNA_file):
    dfProt = pd.read_csv(protein_file, sep='\t', compression='gzip')

    dfRNA = pd.read_csv(RNA_file, index_col='Symbol', na_values='-', sep='\t', compression='gzip')

    dfProt['Gene'] = [get_gene_name(d) for d in dfProt['Description']]

    # Protein data correction
    dfP = np.log2(dfProt.set_index('Gene').iloc[:, 17:26].dropna(axis='rows').copy())
    dfP = dfP - dfP.median()  # correct relative ratio for imperfections

    # RNA data correction -- distribution of RPKM values
    dfRNA.iloc[:, 22:32].replace(0, np.nan).dropna(axis='rows').boxplot(figsize=(10, 5), vert=False, fontsize=14)

    dfR = dfRNA.iloc[:, 22:32].replace(0, np.nan).dropna(axis='rows').copy()
    rnaWT1 = dfR['WT1_IPTG_rpkm'].copy()
    dfR = dfR.drop(columns=[
        'WT1_IPTG_rpkm',
    ])

    #Create a dataframe from the repeated WT1 values
    rnaWT1frame = pd.concat([
        rnaWT1.to_frame(),
    ] * 9, axis='columns')
    rnaWT1frame.columns = dfR.columns

    #Divide the RPKMs with the corresponding WT1 values
    dfR = dfR.div(rnaWT1frame, axis='columns')

    dfR = np.log2(dfR)

    # re-normalize RNA-seq to equal medians
    dfR = dfR - dfR.median()

    #rename for convenience

    dfP_rename = {
        'Abundance Ratio S02_WT  S01_WT': 'WT',
        'Abundance Ratio S03_ORF  S01_WT': 'ORF_1',
        'Abundance Ratio S04_ORF  S01_WT': 'ORF_2',
        'Abundance Ratio S05_ORF_IPTG  S01_WT': 'ORF_IPTG_1',
        'Abundance Ratio S06_ORF_IPTG  S01_WT': 'ORF_IPTG_2',
        'Abundance Ratio S07_SVI  S01_WT': 'SVI_1',
        'Abundance Ratio S08_SVI  S01_WT': 'SVI_2',
        'Abundance Ratio S09_SVI_IPTG  S01_WT': 'SVI_IPTG_1',
        'Abundance Ratio S10_SVI_IPTG  S01_WT': 'SVI_IPTG_2'
    }

    dfR_rename = {
        'ORF1_1_rpkm': 'ORF_1',
        'ORF1_1_IPTG_rpkm': 'ORF_IPTG_1',
        'ORF1_2_rpkm': 'ORF_2',
        'ORF1_2_IPTG_rpkm': 'ORF_IPTG_2',
        'Svi3_3_1_rpkm': 'SVI_1',
        'Svi3_3_1_IPTG_rpkm': 'SVI_IPTG_1',
        'Svi3_3_2_rpkm': 'SVI_2',
        'Svi3_3_2_IPTG_rpkm': 'SVI_IPTG_2',
        'WT2_IPTG_rpkm': 'WT'
    }

    dfP.rename(columns=dfP_rename, inplace=True)

    dfR.rename(columns=dfR_rename, inplace=True)

    # sort by column name
    dfR.sort_index(axis=1, inplace=True)
    dfP.sort_index(axis=1, inplace=True)

    protReplLogFC = replicate_LogFC(dfP)
    rnaReplLogFC = replicate_LogFC(dfR)

    dfP_AVG = max_LogFC_avg_groups(dfP)
    dfR_AVG = max_LogFC_avg_groups(dfR)

    # take 90th percentile of maximum average change between replicates
    dfP_AVG = dfP.join(dfP_AVG, how='left')
    dfR_AVG = dfR.join(dfR_AVG, how='left')

    dfPandR = dfP_AVG.join(dfR_AVG, lsuffix='_Pr', rsuffix='_RNA', how='inner')
    dfPandR = dfPandR.loc[(dfPandR['MaxLog2AVGFC_Pr'] >= np.percentile(protReplLogFC, 90)) |
                          (dfPandR['MaxLog2AVGFC_RNA'] >= np.percentile(rnaReplLogFC, 90))].copy()
    dfProteins = dfPandR.iloc[:, 9:13]
    dfRNAs = dfPandR.iloc[:, 23:27]
    return dfProteins, dfRNAs
