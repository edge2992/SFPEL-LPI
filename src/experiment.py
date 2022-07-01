from sklearn.metrics import (
    auc,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from src.eval import evaluation_metric
from src.featurization import fast_LNC_calculate
from src.loader import InputData
import numpy as np


# experiment for new lncRNA
def cross_validation(seed=1, CV=5):
    data = InputData()
    interaction_matrix = data.InteractionMatrix
    row = interaction_matrix.shape[0]
    CV_matrix = np.ceil(np.random.rand(row) * CV)
    result = np.zeros(7)

    for cv in range(1, CV + 1):
        train_index = np.where(CV_matrix != cv)[0]
        test_index = np.where(CV_matrix == cv)[0]
        train_interaction_matrix = interaction_matrix[train_index, :]
        test_interaction_matrix = interaction_matrix[test_index, :]

        X_feature_num = 2
        train_X = []
        test_X = []
        train_X.append(data.PCPseDNCFeature_LNCRNA[train_index, :])
        train_X.append(data.SCPseDNCFeature_LNCRNA[train_index, :])

        test_X.append(data.PCPseDNCFeature_LNCRNA[test_index, :])
        test_X.append(data.SCPseDNCFeature_LNCRNA[test_index, :])
        train_lncRNA_simi = []
        test_lncRNA_simi = []
        for i in range(X_feature_num):
            train_lncRNA_simi.append(
                fast_LNC_calculate(train_X[i], train_X[i].shape[0])
            )
            test_lncRNA_simi.append(fast_LNC_calculate(test_X[i], test_X[i].shape[0]))

        score_matrix = LPI_pred(
            train_X, train_interaction_matrix, train_lncRNA_simi, test_X
        )
        # 評価
        real_score = np.ravel(test_interaction_matrix)
        predict_score = np.ravel(score_matrix)
        aupr = average_precision_score(real_score, predict_score)
        auc_score = roc_auc_score(real_score, predict_score)
        print("aupr: {}, auc: {}".format(aupr, auc_score))
        [sen, spec, precision, accuracy, f1] = evaluation_metric(
            real_score, predict_score
        )
        result += np.array([aupr, auc_score, sen, spec, precision, accuracy, f1])
        # plot_ROC(real_score, predict_score)
    return result / CV


def LPI_pred(Xs, train_interaction_matrix, train_lncRNA_simi, test_X):
    mu = 0.001
    lam = 0.0001
    gam = 4
    max_iter = 10
    eps = 1e-6
    Y = train_interaction_matrix.copy()
    simi_num = len(train_lncRNA_simi)
    Ls = []
    for i in range(simi_num):
        # TODO: Dの値がWのrow sumではなく1となっている
        Ls.append(np.eye(train_lncRNA_simi[i].shape[0]) - train_lncRNA_simi[i])
    mx = len(Xs)
    ml = len(Ls)
    for i in range(mx):
        Xs[i] = Xs[i].T
    train_res = lpi(Xs, Ls, Y, mx, ml, mu, lam, gam, max_iter, eps)
    # 予測
    Gs = train_res["Gs"]
    alpha = train_res["alpha"]
    y_pred = np.zeros((test_X[0].shape[0], train_interaction_matrix.shape[1]))
    for i in range(mx):
        y_pred += alpha[i, 0] * test_X[i] @ Gs[i]
    return y_pred


def predict(Xs, pn, alpha, Gs):
    y_pred = np.zeros((Xs[0].shape[0], pn))
    for i in range(len(Xs)):
        y_pred += alpha[i, 0] * Xs[i] @ Gs[i]
    return y_pred


def lpi(Xs, Ls, Y, mx, ml, mu, lam, gam, max_iter, eps):
    # XsにはXの転置行列が入る
    n, c = Y.shape
    ds = np.array([Xs[i].shape[0] for i in range(mx)])
    # Gs_iにはG_iの転置行列が入る
    Gs = [np.random.rand(ds[i], c) for i in range(mx)]
    e1ds = [np.ones((ds[i], 1)) for i in range(mx)]

    alpha = np.ones((ml, 1)) / ml
    Gs_old = [np.zeros_like(Gs[i]) for i in range(mx)]
    As = [np.zeros_like(Gs[i]) for i in range(mx)]
    As_pos = [np.zeros_like(Gs[i]) for i in range(mx)]
    As_neg = [np.zeros_like(Gs[i]) for i in range(mx)]
    Bs = [np.zeros_like(Gs[i]) for i in range(mx)]
    Bs_pos = [np.zeros_like(Gs[i]) for i in range(mx)]
    Bs_neg = [np.zeros_like(Gs[i]) for i in range(mx)]
    L = np.zeros((n, n))
    for i in range(ml):
        L += alpha[i, 0] ** gam * Ls[i]
    for t in range(max_iter):
        Q = Y.astype("float64")
        for i in range(mx):
            Gs_old[i] = Gs[i].copy()
            Q += mu * Xs[i].T @ Gs[i]
        P = np.linalg.inv(L + (1 + mx * mu) * np.diag(np.ones(n)))
        F_mat = P @ Q
        for i in range(mx):
            # fmt: off
            # TODO: XとX.Tに挟まれた部分が不明
            As[i] = Xs[i] @ (mu * np.diag(np.ones(n)) - mu**2 * P.T) @ Xs[i].T + \
                lam * (e1ds[i] @ e1ds[i].T)
            # fmt: on

            As_pos[i] = (np.abs(As[i]) + As[i]) / 2
            As_neg[i] = (np.abs(As[i]) - As[i]) / 2
            Bs[i] = mu * Xs[i] @ P @ Y
            for j in range(mx):
                if i == j:
                    continue
                else:
                    Bs[i] += mu**2 * Xs[i] @ P.T @ Xs[j].T @ Gs[j]
            Bs_pos[i] = (np.abs(Bs[i]) + Bs[i]) / 2
            Bs_neg[i] = (np.abs(Bs[i]) - Bs[i]) / 2

        # fmt: off
        for i in range(mx):
            Gs[i] = Gs[i] * np.sqrt((Bs_pos[i] + As_neg[i] @ Gs[i]) / (Bs_neg[i] + As_pos[i] @ Gs[i]))

        for i in range(ml):
            alpha[i, 0] = (1 / np.sum(np.diag((F_mat.T @ Ls[i] @ F_mat)))) ** (1.0 / (gam - 1.0))
        # fmt: on
        alpha = alpha / np.sum(alpha)
        L = np.zeros((n, n))
        for i in range(ml):
            L += alpha[i, 0] ** gam * Ls[i]
        diff_G = np.zeros((mx, 1))
        for i in range(mx):
            diff_G[i, 0] = np.linalg.norm(Gs[i] - Gs_old[i], "fro") / np.linalg.norm(
                Gs_old[i], "fro"
            )
        if np.mean(diff_G) < eps:
            # 学習を終了させる
            break

        y_pred = predict([x.T for x in Xs], Y.shape[1], alpha, Gs)
        fpr2, tpr2, _ = roc_curve(np.ravel(Y), np.ravel(y_pred))
        fpr, tpr, _ = roc_curve(np.ravel(Y), np.ravel(F_mat))
        print(
            "epoch {}, train Loss: {:.3f}, diffG: {:.3f}, auc: {:.3f} {:.3f}".format(
                t,
                calc_loss(Gs, F_mat, Y, Xs, Ls, mx, ml, mu, lam, gam, alpha),
                np.mean(diff_G),
                auc(fpr, tpr),
                auc(fpr2, tpr2),
            )
        )
    return {"Gs": Gs, "alpha": alpha, "predict": F_mat}


def calc_loss(Gs, F_mat, Y, Xs, Ls, mx, ml, mu, lam, gam, alpha):
    loss = np.linalg.norm(F_mat - Y)
    for i in range(mx):
        loss += mu * np.linalg.norm(Xs[i].T @ Gs[i] - F_mat)
        loss += lam * np.linalg.norm(np.linalg.norm(Gs[i].T, ord=1, axis=1))
    for i in range(ml):
        loss += alpha[i, 0] ** gam * np.trace(F_mat.T @ Ls[i] @ F_mat)

    return loss


if __name__ == "__main__":
    result = cross_validation()
    print(
        """average_precision_score: {:.3f}, AUC_score: {:.3f}, sensitivity: {:.3f},
        specificity: {:.3f}, precision: {:.3f}, accuracy: {:.3f}, f1: {:.3f}""".format(
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
        )
    )
