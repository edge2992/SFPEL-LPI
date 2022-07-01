from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve
from sklearn.utils.validation import check_is_fitted
import numpy as np

from src.featurization import fast_LNC_calculate
from src.novel_mol_interaction_prediction import fit_binary


class LPI(BaseEstimator, ClassifierMixin):
    """lncRNA Protein Interaction predictor for novel molculers"""

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

    def _fit_threshold(self, X, y):
        """f1 scoreが一番高いthresholdを選ぶ"""
        y_score = self.predict_proba(X)
        precisions, recalls, thresholds = precision_recall_curve(
            y.ravel(), y_score.ravel()
        )
        fscore = (2 * precisions * recalls) / (precisions + recalls + 1e-20)
        ix = np.argmax(fscore)
        return thresholds[ix]

    def fit(self, X, y):
        Ls = self.prepare_Ls(X)  # similalityの計算
        self._fit_binary(X, Ls, y)
        self.threshold_ = self._fit_threshold(X, y)
        return self

    def predict(self, X, check=True):
        "return 01 labaled value (y_pred)."
        if check:
            check_is_fitted(self, "Gs_")
            check_is_fitted(self, "alpha_")
            check_is_fitted(self, "Fmat_")
        y_score = self.predict_proba(X)
        # matlabコードではテストデータの答えを見て、
        # 一番良いthresholdを選んでいた。
        # 訓練データからthresholdを学習すると精度が10%落ちる。
        y_pred = y_score >= self.threshold_
        return y_pred

    def predict_proba(self, X):
        "return raw y_score (not probability)."
        y_score = np.zeros((X[0].shape[0], self.Gs_[0].shape[1]))
        for i in range(len(X)):
            y_score += self.alpha_[i, 0] * X[i] @ self.Gs_[i]
        return y_score

    def prepare_Ls(self, Xs):
        """calculate linear neighborrhood similarity (LNS)."""
        train_lncRNA_simi = []
        for x in Xs:
            train_lncRNA_simi.append(fast_LNC_calculate(x, x.shape[0]))
        Ls = []
        for simi in train_lncRNA_simi:
            Ls.append(np.eye(simi.shape[0]) - simi)
        return Ls
