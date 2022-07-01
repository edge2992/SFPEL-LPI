from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np

from src.featurization import fast_LNC_calculate
from src.novel_mol_interaction_prediction import fit_binary


class LPI(BaseEstimator, ClassifierMixin):
    def __init__(self, mu=0.001, lam=0.0001, gam=4, max_iter=10, eps=1e-6):
        self.mu = mu
        self.lam = lam
        self.gam = gam
        self.max_iter = max_iter
        self.eps = eps

    def _fit_binary(self, X, Ls, y):
        """Fit a binary classifier on X and y"""
        Gs, alpha, F_mat, n_iter_ = fit_binary(
            X, Ls, y, self.mu, self.lam, self.gam, self.max_iter, self.eps
        )
        self.Gs_ = Gs
        self.alpha_ = alpha
        self.Fmat_ = F_mat
        self.n_iter_ = n_iter_
        return self

    def fit(self, X, y):
        # fitの後に値が確定する変数はコンストラクタに書かず、
        # fitの中でsaffixに_を付けて宣言する
        Ls = self.prepare_Ls(X)  # similalityの計算
        return self._fit_binary(X, Ls, y)

    def predict(self, X, check=True):
        if check:
            check_is_fitted(self, "Gs_")
            check_is_fitted(self, "self.alpha__")
            check_is_fitted(self, "Fmat_")
        Xs = X

        y_score = np.zeros((Xs[0].shape[0], self.Gs_[0].shape[1]))
        for i in range(len(Xs)):
            y_score += self.alpha_[i, 0] * Xs[i] @ self.Gs_[i]
        # TODO: thresholdを調べておく
        y_pred = y_score
        return y_pred

    def prepare_Ls(self, Xs):
        train_lncRNA_simi = []
        for x in Xs:
            train_lncRNA_simi.append(fast_LNC_calculate(x, x.shape[0]))
        Ls = []
        for simi in train_lncRNA_simi:
            Ls.append(np.eye(simi.shape[0]) - simi)
        return Ls
