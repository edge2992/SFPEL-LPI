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

        def calc_L(Ls, alpha, gam):
            L = np.zeros_like(Ls[0])
            for alp, l in zip(alpha, Ls):
                L += alp**gam * l
            return L

        def calc_Q(y, mu, Xs, Gs):
            Q = y.astype("float64")
            for x, g in zip(Xs, Gs):
                Q += mu * x.T @ g
            return Q

        def separate_positive(A):
            return (np.abs(A) + A) / 2

        def separate_negative(A):
            return (np.abs(A) - A) / 2

        def separate_mat(A):
            return separate_positive(A), separate_negative(A)

        def calc_A(x, mu, n, P, lam, e1ds):
            # Mが不明
            M = mu * np.diag(np.ones(n)) - mu**2 * P.T
            A = x @ M @ x.T + lam * (e1ds @ e1ds.T)
            return A

        def calc_B(i, mu, mx, Xs, P, y, Gs):
            B = mu * Xs[i] @ P @ y
            for j in range(mx):
                if i == j:
                    # なぜ場合分けが必要なのか
                    continue
                else:
                    Rj = Xs[j].T @ Gs[j]
                    B += mu**2 * Xs[i] @ P.T @ Rj
            return B

        def update_G(Xs, mu, n, P, lam, mx, Gs, e1ds):
            for i in range(mx):
                A = calc_A(Xs[i], mu, n, P, lam, e1ds[i])
                B = calc_B(i, mu, mx, Xs, P, y, Gs)
                A_pos, A_neg = separate_mat(A)
                B_pos, B_neg = separate_mat(B)

                Gs[i] = Gs[i] * np.sqrt(
                    (B_pos + A_neg @ Gs[i]) / (B_neg + A_pos @ Gs[i])
                )
            return Gs

        for t in range(self.max_iter):
            Gs_old = [g.copy() for g in self.Gs_]
            L = calc_L(Ls, self.alpha_, self.gam)
            Q = calc_Q(y, self.mu, Xs, self.Gs_)
            P = np.linalg.inv(L + (1 + mx * self.mu) * np.diag(np.ones(n)))
            self.Fmat_ = P @ Q

            self.Gs_ = update_G(Xs, self.mu, n, P, self.lam, mx, self.Gs_, e1ds)

            for i in range(ml):
                self.alpha_[i, 0] = (
                    1 / np.sum(np.diag((self.Fmat_.T @ Ls[i] @ self.Fmat_)))
                ) ** (1.0 / (self.gam - 1.0))
            # fmt: on
            self.alpha_ = self.alpha_ / np.sum(self.alpha_)

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
