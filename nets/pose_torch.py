# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> pose_torch
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   17/06/2022 11:49
=================================================='''
import torch
import numpy as np


def _homo(x):
    # input: x [N, 2] or [batch_size, N, 2]
    # output: x_homo [N, 3]  or [batch_size, N, 3]
    assert len(x.size()) in [2, 3]
    print(f"x: {x.size()[0]}, {x.size()[1]}, {x.dtype}, {x.device}")
    if len(x.size()) == 2:
        ones = torch.ones(x.size()[0], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 1)
    elif len(x.size()) == 3:
        ones = torch.ones(x.size()[0], x.size()[1], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 2)
    return x_homo


def _de_homo(x_homo):
    # input: x_homo [N, 3] or [batch_size, N, 3]
    # output: x [N, 2] or [batch_size, N, 2]
    assert len(x_homo.size()) in [2, 3]
    epi = 1e-10
    if len(x_homo.size()) == 2:
        x = x_homo[:, :-1] / ((x_homo[:, -1] + epi).unsqueeze(-1))
    else:
        x = x_homo[:, :, :-1] / ((x_homo[:, :, -1] + epi).unsqueeze(-1))
    return x


def _normalize_XY(X, Y):
    """ The Hartley normalization. Following https://github.com/marktao99/python/blob/da2682f8832483650b85b0be295ae7eaf179fcc5/CVP/samples/sfm.py#L157
    corrected with https://www.mathworks.com/matlabcentral/fileexchange/27541-fundamental-matrix-computation
    and https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm """
    if X.size()[0] != Y.size()[0]:
        raise ValueError("Number of points don't match.")
    X = _homo(X)
    mean_1 = torch.mean(X[:, :2], dim=0, keepdim=True)
    S1 = np.sqrt(2) / torch.mean(torch.norm(X[:, :2] - mean_1, 2, dim=1))
    # print(mean_1.size(), S1.size())
    T1 = torch.tensor([[S1, 0, -S1 * mean_1[0, 0]], [0, S1, -S1 * mean_1[0, 1]], [0, 0, 1]], device=X.device)
    X_normalized = _de_homo(torch.mm(T1, X.t()).t())  # ideally zero mean (x, y), and sqrt(2) average norm

    # xxx = X_normalized.numpy()
    # print(np.mean(xxx, axis=0))
    # print(np.mean(np.linalg.norm(xxx, 2, axis=1)))

    Y = _homo(Y)
    mean_2 = torch.mean(Y[:, :2], dim=0, keepdim=True)
    S2 = np.sqrt(2) / torch.mean(torch.norm(Y[:, :2] - mean_2, 2, dim=1))
    T2 = torch.tensor([[S2, 0, -S2 * mean_2[0, 0]], [0, S2, -S2 * mean_2[0, 1]], [0, 0, 1]], device=X.device)
    Y_normalized = _de_homo(torch.mm(T2, Y.t()).t())

    return X_normalized, Y_normalized, T1, T2


def _normalize_XY_batch(X, Y):
    """ The Hartley normalization. Following https://github.com/marktao99/python/blob/da2682f8832483650b85b0be295ae7eaf179fcc5/CVP/samples/sfm.py#L157
    corrected with https://www.mathworks.com/matlabcentral/fileexchange/27541-fundamental-matrix-computation
    and https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm """
    # X: [batch_size, N, 2]
    if X.size()[1] != Y.size()[1]:
        raise ValueError("Number of points don't match.")
    X = _homo(X)
    mean_1s = torch.mean(X[:, :, :2], dim=1, keepdim=True)
    S1s = np.sqrt(2) / torch.mean(torch.norm(X[:, :, :2] - mean_1s, 2, dim=2), dim=1)
    T1_list = []
    for S1, mean_1 in zip(S1s, mean_1s):
        T1_list.append(
            torch.tensor([[S1, 0, -S1 * mean_1[0, 0]], [0, S1, -S1 * mean_1[0, 1]], [0, 0, 1]], device=X.device))
    T1s = torch.stack(T1_list)
    X_normalized = _de_homo(
        torch.bmm(T1s, X.transpose(1, 2)).transpose(1, 2))  # ideally zero mean (x, y), and sqrt(2) average norm

    # xxx = X_normalized.numpy()
    # print(np.mean(xxx, axis=0))
    # print(np.mean(np.linalg.norm(xxx, 2, axis=1)))

    Y = _homo(Y)
    mean_2s = torch.mean(Y[:, :, :2], dim=1, keepdim=True)
    S2s = np.sqrt(2) / torch.mean(torch.norm(Y[:, :, :2] - mean_2s, 2, dim=2), dim=1)
    T2_list = []
    for S2, mean_2 in zip(S2s, mean_2s):
        T2_list.append(
            torch.tensor([[S2, 0, -S2 * mean_2[0, 0]], [0, S2, -S2 * mean_2[0, 1]], [0, 0, 1]], device=X.device))
    T2s = torch.stack(T2_list)
    Y_normalized = _de_homo(torch.bmm(T2s, Y.transpose(1, 2)).transpose(1, 2))

    return X_normalized, Y_normalized, T1s, T2s


def _E_from_XY(X, Y, K, W=None, if_normzliedK=False, normalize=True,
               show_debug=False):  # Ref: https://github.com/marktao99/python/blob/master/CVP/samples/sfm.py#L55
    """ Normalized Eight Point Algorithom for E: [Manmohan] In practice, one would transform the data points by K^{-1}, then do a Hartley normalization, then estimate the F matrix (which is now E matrix), then set the singular value conditions, then denormalize. Note that it's better to set singular values first, then denormalize.
        X, Y: [N, 2] """
    if if_normzliedK:
        X_normalizedK = X
        Y_normalizedK = Y
    else:
        X_normalizedK = _de_homo(torch.mm(torch.inverse(K), _homo(X).t()).t())
        Y_normalizedK = _de_homo(torch.mm(torch.inverse(K), _homo(Y).t()).t())

    if normalize:
        X, Y, T1, T2 = _normalize_XY(X_normalizedK, Y_normalizedK)
    else:
        X, Y = X_normalizedK, Y_normalizedK
    xx = torch.cat([X.t(), Y.t()], dim=0)
    XX = torch.stack([
        xx[2, :] * xx[0, :], xx[2, :] * xx[1, :], xx[2, :],
        xx[3, :] * xx[0, :], xx[3, :] * xx[1, :], xx[3, :],
        xx[0, :], xx[1, :], torch.ones_like(xx[0, :])
    ], dim=0).t()  # [N, 9]
    # print(XX.size())
    if W is not None:
        XX = torch.mm(W, XX)  # [N, 9]
    # print(XX[:2])
    U, D, V = torch.svd(XX, some=True)
    if show_debug:
        print('[info.Debug @_E_from_XY] Singualr values of XX:\n', D.numpy())

    # U_np, D_np, V_np = np.linalg.svd(XX.numpy())

    F_recover = torch.reshape(V[:, -1], (3, 3))
    # print('-', F_recover)

    FU, FD, FV = torch.svd(F_recover, some=True)
    if show_debug:
        print('[info.Debug @_E_from_XY] Singular values for recovered E(F):\n', FD.numpy())

    # FDnew = torch.diag(FD);
    # FDnew[2, 2] = 0;
    # F_recover_sing = torch.mm(FU, torch.mm(FDnew, FV.t()))
    S_110 = torch.diag(torch.tensor([1., 1., 0.], dtype=FU.dtype, device=FU.device))
    E_recover_110 = torch.mm(FU, torch.mm(S_110, FV.t()))
    # F_recover_sing_rescale = F_recover_sing / torch.norm(F_recover_sing) * torch.norm(F)

    # print(E_recover_110)
    if normalize:
        E_recover_110 = torch.mm(T2.t(), torch.mm(E_recover_110, T1))
    return E_recover_110


def _E_from_XY_batch(X, Y, K, W=None, if_normzliedK=False, normalize=True,
                     show_debug=False):  # Ref: https://github.com/marktao99/python/blob/master/CVP/samples/sfm.py#L55
    # from batch_svd import batch_svd  # https://github.com/KinglittleQ/torch-batch-svd.git
    """ Normalized Eight Point Algorithom for E: [Manmohan] In practice, one would transform the data points by K^{-1}, then do a Hartley normalization, then estimate the F matrix (which is now E matrix), then set the singular value conditions, then denormalize. Note that it's better to set singular values first, then denormalize.
        X, Y: [N, 2] """
    assert X.dtype == torch.float32, 'batch_svd currently only supports torch.float32!'
    if if_normzliedK:
        X_normalizedK = X.float()
        Y_normalizedK = Y.float()
    else:
        X_normalizedK = _de_homo(
            torch.bmm(torch.inverse(K), _homo(X).transpose(1, 2)).transpose(1, 2)).float()
        Y_normalizedK = _de_homo(
            torch.bmm(torch.inverse(K), _homo(Y).transpose(1, 2)).transpose(1, 2)).float()

    # assert normalize==False, 'Not supported in batch mode yet!'
    if normalize:
        X, Y, T1, T2 = _normalize_XY_batch(X_normalizedK, Y_normalizedK)
    else:
        X, Y = X_normalizedK, Y_normalizedK

    xx = torch.cat([X, Y], dim=2)
    XX = torch.stack([
        xx[:, :, 2] * xx[:, :, 0], xx[:, :, 2] * xx[:, :, 1], xx[:, :, 2],
        xx[:, :, 3] * xx[:, :, 0], xx[:, :, 3] * xx[:, :, 1], xx[:, :, 3],
        xx[:, :, 0], xx[:, :, 1], torch.ones_like(xx[:, :, 0])
    ], dim=2)

    if W is not None:
        XX = torch.bmm(W, XX)
    # U, D, V = torch.svd(XX, some=False)
    # print(XX[0, :2])
    # print(XX.size())
    # U, D, V = batch_svd(XX)
    V_list = []
    for XX_single in XX:
        _, _, V_single = torch.svd(XX_single, some=True)
        V_list.append(V_single[:, -1])
    V_last_col = torch.stack(V_list)
    # print(V_last_col.size(), '----')

    # if show_debug:
    #     print('[info.Debug @_E_from_XY] Singualr values of XX:\n', D[0].numpy())

    # F_recover = torch.reshape(V[:, :, -1], (-1, 3, 3))
    F_recover = V_last_col.view(-1, 3, 3)

    # FU, FD, FV= torch.svd(F_recover, some=False)
    # FU, FD, FV = batch_svd(F_recover)
    FU, FD, FV = torch.linalg.svd(F_recover)

    if show_debug:
        print('[info.Debug @_E_from_XY] Singular values for recovered E(F):\n', FD[0].numpy())

    # FDnew = torch.diag(FD);
    # FDnew[2, 2] = 0;
    # F_recover_sing = torch.mm(FU, torch.mm(FDnew, FV.t()))
    S_110 = torch.diag(torch.tensor([1., 1., 0.], dtype=FU.dtype, device=FU.device)).unsqueeze(0).expand(FV.size()[0],
                                                                                                         -1, -1)

    E_recover_110 = torch.bmm(FU, torch.bmm(S_110, FV.transpose(1, 2)))
    # F_recover_sing_rescale = F_recover_sing / torch.norm(F_recover_sing) * torch.norm(F)
    # print(E_recover_110)
    if normalize:
        E_recover_110 = torch.bmm(T2.transpose(1, 2), torch.bmm(E_recover_110, T1))
    return -E_recover_110
