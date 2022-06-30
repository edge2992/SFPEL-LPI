import scipy.io as spio


class InputData:
    def __init__(self, filename="NP2.0.mat"):
        self.loader(filename)

    def loader(self, filename):
        mat = spio.loadmat(filename, squeeze_me=True)
        self._InteractionMatrix = mat["InteractionMatrix"]
        self._PCPseAACFeature_Protein = mat["PCPseAACFeature_Protein"]
        self._PCPseDNCFeature_LNCRNA = mat["PCPseDNCFeature_LNCRNA"]
        self._SCPseAACFeature_Protein = mat["SCPseAACFeature_Protein"]
        self._SCPseDNCFeature_LNCRNA = mat["SCPseDNCFeature_LNCRNA"]
        self._lncRNA_subgraph_similarity_normalize = mat[
            "lncRNA_subgraph_similarity_normalize"
        ]
        self._protein_subgraph_similarity_normalize = mat[
            "protein_subgraph_similarity_normalize"
        ]

    @property
    def InteractionMatrix(self):
        return self._InteractionMatrix

    @property
    def PCPseAACFeature_Protein(self):
        return self._PCPseAACFeature_Protein

    @property
    def PCPseDNCFeature_LNCRNA(self):
        return self._PCPseDNCFeature_LNCRNA

    @property
    def SCPseAACFeature_Protein(self):
        return self._SCPseAACFeature_Protein

    @property
    def SCPseDNCFeature_LNCRNA(self):
        return self._SCPseDNCFeature_LNCRNA

    @property
    def lncRNA_subgraph_similarity_normalize(self):
        return self._lncRNA_subgraph_similarity_normalize

    @property
    def protein_subgraph_similarity_normalize(self):
        return self._protein_subgraph_similarity_normalize
