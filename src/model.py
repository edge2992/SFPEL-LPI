from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import auc, roc_curve
from sklearn.utils.validation import check_is_fitted
import numpy as np

from src.featurization import fast_LNC_calculate


class LPI(BaseEstimator, ClassifierMixin):
    def __init__(self, mu=0.001, lam=0.0001, gam=4, max_iter=10, eps=1e-6):
        self.mu = mu
        self.lam = lam
        self.gam = gam
        self.max_iter = max_iter
        self.eps = eps

    def _fit_binary(self, X, y, sample_weight, max_iter):
        """Fit a binary classifier on X and y"""
        Gs, alpha, F_mat, n_iter_ = fit_binary()

        self.n_iter_ = n_iter_
        # update parameter
        self.Gs_ = Gs
        self.alpha_ = alpha
        self.F_mat = F_mat

    def fit(self, X, y):
        # fitの後に値が確定する変数はコンストラクタに書かず、
        # fitの中でsaffixに_を付けて宣言する
        Xs = X.copy()
        # similalityの計算
        train_lncRNA_simi = []
        for x in Xs:
            train_lncRNA_simi.append(fast_LNC_calculate(x, x.shape[0]))
        Ls = []
        for simi in train_lncRNA_simi:
            Ls.append(np.eye(simi.shape[0]) - simi)

        n, c = y.shape
        mx = len(Xs)
        ml = len(Ls)
        Xs = [x.T for x in Xs]
        self.Gs_, self.alpha_, e1ds = prepare_fit_binary(Xs, ml, c)

        Gs_old = zeros_like_numpy_list(self.Gs_)
        As = zeros_like_numpy_list(self.Gs_)
        As_pos = zeros_like_numpy_list(self.Gs_)
        As_neg = zeros_like_numpy_list(self.Gs_)
        Bs = zeros_like_numpy_list(self.Gs_)
        Bs_pos = zeros_like_numpy_list(self.Gs_)
        Bs_neg = zeros_like_numpy_list(self.Gs_)

        L = np.zeros((n, n))
        for i in range(ml):
            L += self.alpha_[i, 0] ** self.gam * Ls[i]
        for t in range(self.max_iter):
            Q = y.astype("float64")
            for i in range(mx):
                Gs_old[i] = self.Gs_[i].copy()
                Q += self.mu * Xs[i].T @ self.Gs_[i]
            P = np.linalg.inv(L + (1 + mx * self.mu) * np.diag(np.ones(n)))
            self.Fmat_ = P @ Q
            for i in range(mx):
                # fmt: off
                # TODO: XとX.Tに挟まれた部分が不明
                As[i] = Xs[i] @ (self.mu * np.diag(np.ones(n)) - self.mu**2 * P.T) @ Xs[i].T + \
                    self.lam * (e1ds[i] @ e1ds[i].T)
                # fmt: on

                As_pos[i] = (np.abs(As[i]) + As[i]) / 2
                As_neg[i] = (np.abs(As[i]) - As[i]) / 2
                Bs[i] = self.mu * Xs[i] @ P @ y
                for j in range(mx):
                    if i == j:
                        continue
                    else:
                        Bs[i] += self.mu**2 * Xs[i] @ P.T @ Xs[j].T @ self.Gs_[j]
                Bs_pos[i] = (np.abs(Bs[i]) + Bs[i]) / 2
                Bs_neg[i] = (np.abs(Bs[i]) - Bs[i]) / 2

            # fmt: off
            for i in range(mx):
                self.Gs_[i] = self.Gs_[i] * np.sqrt((Bs_pos[i] + As_neg[i] @ self.Gs_[i]) / (Bs_neg[i] + As_pos[i] @ self.Gs_[i]))

            for i in range(ml):
                self.alpha_[i, 0] = (1 / np.sum(np.diag((self.Fmat_.T @ Ls[i] @ self.Fmat_)))) ** (1.0 / (self.gam - 1.0))
            # fmt: on
            self.alpha_ = self.alpha_ / np.sum(self.alpha_)

            # 次の準備
            L = np.zeros((n, n))
            for i in range(ml):
                L += self.alpha_[i, 0] ** self.gam * Ls[i]

            # diff_Gを計算する
            diff_G = np.zeros((mx, 1))
            for i in range(mx):
                diff_G[i, 0] = np.linalg.norm(
                    self.Gs_[i] - Gs_old[i], "fro"
                ) / np.linalg.norm(Gs_old[i], "fro")
            if np.mean(diff_G) < self.eps:
                # 学習を終了させる
                return self

            # ログを出力
            y_pred = self.predict(X, check=False)
            fpr, tpr, _ = roc_curve(np.ravel(y), np.ravel(y_pred))
            print(
                "epoch {}, diffG: {:.3f}, auc: {:.3f}".format(
                    t,
                    np.mean(diff_G),
                    auc(fpr, tpr),
                )
            )

        return self

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


def zeros_like_numpy_list(As):
    return [np.zeros_like(A) for A in As]


def prepare_fit_binary(Xs, ml, c):
    """initalization for fit_binary.
    returns Gs, alpha, e1ds
    """
    ds = np.array([X.shape[0] for X in Xs])  # 特徴量の個数

    Gs = [np.random.rand(fn, c) for fn in ds]  # self.Gs_にはG_の転置行列が入る
    alpha = np.ones((ml, 1)) / ml
    e1ds = [np.ones((fn, 1)) for fn in ds]
    return (Gs, alpha, e1ds)


def fit_binary():
    pass
