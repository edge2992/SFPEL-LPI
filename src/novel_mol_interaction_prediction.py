import numpy as np


def separate_mat(A):
    def separate_positive(A):
        return (np.abs(A) + A) / 2

    def separate_negative(A):
        return (np.abs(A) - A) / 2

    return separate_positive(A), separate_negative(A)


def prepare_fit_binary(Xs, ml, c):
    """initalization for fit_binary.
    returns Gs, alpha, e1ds
    """
    ds = np.array([X.shape[1] for X in Xs])  # 特徴量の個数

    Gs = [np.random.rand(fn, c) for fn in ds]  # self.Gs_にはG_の転置行列が入る
    alpha = np.ones((ml, 1)) / ml
    e1ds = [np.ones((fn, 1)) for fn in ds]
    return (Gs, alpha, e1ds)


def fit_binary(X, Ls, y, mu, lam, gam, max_iter, eps):
    def calc_L(Ls, alpha, gam):
        L = np.zeros_like(Ls[0])
        for alp, l in zip(alpha, Ls):
            L += alp**gam * l
        return L

    def calc_Q(y, mu, X, Gs):
        Q = y.astype("float64")
        for x, g in zip(X, Gs):
            Q += mu * x @ g
        return Q

    def calc_A(x, P, e1ds, mu, lam):
        # Mが不明
        n = P.shape[0]
        M = mu * np.diag(np.ones(n)) - mu**2 * P.T
        A = x.T @ M @ x + lam * (e1ds @ e1ds.T)
        return A

    def calc_B(X, i, y, P, Gs, mu):
        B = mu * X[i].T @ P @ y
        for j in range(len(X)):
            if i == j:
                # なぜ場合分けが必要なのか
                continue
            else:
                Rj = X[j] @ Gs[j]
                B += mu**2 * X[i].T @ P.T @ Rj
        return B

    def update_G(X, y, mu, P, lam, Gs, e1ds):
        for i in range(len(X)):
            A = calc_A(X[i], P, e1ds[i], mu, lam)
            B = calc_B(X, i, y, P, Gs, mu)
            A_pos, A_neg = separate_mat(A)
            B_pos, B_neg = separate_mat(B)

            Gs[i] = Gs[i] * np.sqrt((B_pos + A_neg @ Gs[i]) / (B_neg + A_pos @ Gs[i]))
        return Gs

    def update_alpha(Fmat, Ls, gam):
        ml = len(Ls)
        alpha = np.zeros((ml, 1))
        for i in range(ml):
            alpha[i, 0] = (1 / np.sum(np.diag((Fmat.T @ Ls[i] @ Fmat)))) ** (
                1.0 / (gam - 1.0)
            )
        return alpha / np.sum(alpha)

    def calc_diffG(Gs_new, Gs_old):
        diff_G = np.zeros((len(Gs_new), 1))
        for i in range(len(Gs_new)):
            diff_G[i, 0] = np.linalg.norm(
                Gs_new[i] - Gs_old[i], "fro"
            ) / np.linalg.norm(Gs_old[i], "fro")
        return np.mean(diff_G)

    Gs_, alpha_, e1ds = prepare_fit_binary(X, len(Ls), y.shape[1])

    for n_iter_ in range(max_iter):
        Gs_old = [g.copy() for g in Gs_]
        L = calc_L(Ls, alpha_, gam)
        Q = calc_Q(y, mu, X, Gs_)
        P = np.linalg.inv(L + (1 + len(X) * mu) * np.diag(np.ones(X[0].shape[0])))
        Fmat_ = P @ Q
        Gs_ = update_G(X, y, mu, P, lam, Gs_, e1ds)
        alpha_ = update_alpha(Fmat_, Ls, gam)
        diff_G = calc_diffG(Gs_, Gs_old)
        if diff_G < eps:
            # 学習を終了させる
            break

    return Gs_, alpha_, Fmat_, n_iter_
