#!/usr/bin/env python
# coding: utf-8
# Date  : 2018-08-06 11:38:56
# Author: b4zinga
# Email : b4zinga@outlook.com
# Func  :

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from pca import PCA

from settings import AMINO


class FrequencyCluster:
    """Calculate the frequency of amino acid sequences of
    virus in the FA file, and Cluster it with PCA
    """
    def __init__(self, fa):
        self.fa = fa

    def loadDate(self):
        """Take out all amino acid sequences of each virus in the FA file.
        return a dict like that: {('ARQ87434', '2017/04/06'):'MKAILVVLLYTFTTANADTLCIGYHANNSTDTVDTVL...',}
        """
        amino_acid_sequences = {}
        regex = '>(.*?)\s\w/.*?/\d+\s+(\d+/\d*/\d*)\s.*?$'
        with open(self.fa) as file:
            for f in file:
                if f.startswith('>'):
                    items = re.findall(regex, f)
                    if items:
                        virus_name = items[0]
                        amino_acid_sequences[virus_name] = ''
                    else:
                        virus_name = None
                else:
                    try:
                        amino_acid_sequences[virus_name] += f.strip('\n')
                    except KeyError:
                        pass

        return amino_acid_sequences

    def calcWeight(self):
        """Calc of the frequency of amino in acid sequence.
        return a list like that: [[34, 16, 23, 33, 21, 44, 14, 35, 37, 48, 8, 45, 21, 16, 20, 43, 38, 34, 10, 26],...]
        """
        virus = self.loadDate()  # {, 'ARQ87217': 'MKAILVVLLYTFTTANADTLC...',...}
        amino_frequency = []
        for vir in virus:
            amino = AMINO
            temp = []
            for am in amino:
                temp.append(virus[vir].count(am))
            amino_frequency.append(temp)

        return amino_frequency

    def cluster(self):
        """cluster the frequency data with PCA.
        return the <class 'numpy.matrixlib.defmatrix.matrix'> after PCA:

        [[ 2.73957257  4.37035856]
         [ 4.89129411  3.42768334]
         [ 3.98168808  2.79871617]
         ...
         [-2.05245831  4.4499471 ]
         [-2.63118146  4.34273584]
         [-2.40254118  5.18975161]]
        """
        list_data = self.calcWeight()
        weight = np.array(list_data, dtype=int)
        pca = PCA(n_components=2)
        new_data = pca.fit_transform(weight)
        data_mat = np.mat(new_data)

        return data_mat

    def show(self, marker='.', s=1, color='RdYlBu_r'):
        """cluster the frequency data with PCA and show the result.

        list_data: the frequency list of amino acid sequence.
        marker   : scatter markers , the marker style.
        s        : scatter s , the size of markers.
        color    : the color show in PCA.

        Possible values are: Accent, Accent_r, Blues, Blues_r,
        BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r,
        Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys,
        Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r,
        Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r,
        PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r,
        PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy,
        RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r,
        Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r,
        Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu,
        YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r ...
        其中末尾加r是颜色取反
        """
        list_data = self.calcWeight()
        weight = np.array(list_data, dtype=int)
        pca = PCA(n_components=2)
        new_data = pca.fit_transform(weight)
        data_mat = np.mat(new_data)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cm = plt.cm.get_cmap(color)
        sc = ax.scatter(x=data_mat[:, 0].flatten().A[0],
                        y=data_mat[:, 1].flatten().A[0],
                        c=data_mat[:, 1].flatten().A[0]*0.2 + data_mat[:, 0].flatten().A[0]*0.8,
                        # c=data_mat[:, 1].flatten().A[0],
                        # c=data_mat[:, 0].flatten().A[0],
                        marker=marker,
                        s=s,
                        vmin=0, vmax=20,
                        cmap=cm)
        plt.grid(ls=':')
        plt.colorbar(sc)
        plt.show()        



if __name__ == '__main__':
    fc = FrequencyCluster('data/H1N1_1976-2017_without_XXX.fa')
    fc.show()