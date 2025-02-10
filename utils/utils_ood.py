import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn import metrics as metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit


def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh


def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate(
        (np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def save_class_token_dataframe(class_tockens, labels, cls_size=768):
    # x = np.asarray(x)
    x = class_tockens
    x = pd.DataFrame(x, columns=[f'ct{i}' for i in range(cls_size)])
    range(x.shape[1]-1)[-1]
    print(f"class_tockens shaep {class_tockens.shape}")
    for i in range(class_tockens.shape[1]):
        x[f'ct{i}'] = class_tockens[:, i]
    x['label'] = labels
    return x


def plot_histogram(ID_scores, OOD_scores, path):

    # Normalize
    kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)

    # Plot
    plt.style.use('classic')
    plt.rcParams.update({'font.size': 20})
    plt.hist(ID_scores, **kwargs, color='c', label='ID')
    plt.hist(OOD_scores, **kwargs, color='k', label='OOD')
    plt.tight_layout()
    plt.gca().set(ylabel='Density', xlabel='NECO value')
    #
    plt.legend()

    plt.savefig(path+".png")
    return


def get_weight_bias(model, args_experiment):
    model_type = args_experiment.model_architecture_type
    if 'resnet' in model_type:
        model_layers = nested_children(model)
        print(model_layers)
        last_layer = model_layers['linear']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        bias = last_layer.bias
        bias.requires_grad = False
        bias = bias.detach().cpu().numpy()
        weight = last_layer.weight
        weight.requires_grad = False
        weight = weight.detach().cpu().numpy()
        w, b = weight, bias
        print(f'{w.shape=}, {b.shape=}')
    elif 'vit' in model_type:
        model_layers = nested_children(model)
        print(model_layers)
        last_layer = model_layers['head']
        bias = last_layer.bias
        bias.requires_grad = False
        bias = bias.detach().cpu().numpy()
        weight = last_layer.weight
        weight.requires_grad = False
        weight = weight.detach().cpu().numpy()
        w, b = weight, bias
        print(f'{w.shape=}, {b.shape=}')
    return w, b


def remove_classes(pd_dataframe, classes_list=[]):
    for i in classes_list:
        pd_dataframe.drop(
            pd_dataframe[pd_dataframe['labels'] == i].index, inplace=True)
    return pd_dataframe


def PCA_plot(csvFile_train, csvFile_test, csvFile_ood, title, path, num_classes, n_componnents=2, fontsize=25, cls_size=768, in_dataset='cifar10', out_dataset="svhn", model_type='vit'):
    features = [f'ct{i}' for i in range(cls_size)]
    plt.style.use('classic')
    print(f"csvFile_train shape  {csvFile_train.shape}")
    print(f"csvFile_test shape  {csvFile_test.shape}")
    cls_train_original = csvFile_train.loc[:, features].values
    cls_test_original = csvFile_test.loc[:, features].values
    cls_ood_original = csvFile_ood.loc[:, features].values
    csvFile_ood['label'] = -1
    print(csvFile_test['label'][:20])
    test_target = csvFile_test.loc[:, ['label']].values.ravel()
    ood_target = csvFile_ood.loc[:, ['label']].values.ravel()
    ood_target = np.full(ood_target.shape, 'OOD')
    pca = PCA(n_components=n_componnents)
    _ = pca.fit_transform(cls_train_original)
    principalComponents_test = pca.transform(cls_test_original)
    principalComponents_ood = pca.transform(cls_ood_original)
    fig = plt.figure()
    ax = fig.gca()

    tindex = [t for t, _ in StratifiedShuffleSplit(
        1, train_size=0.1, random_state=17).split(principalComponents_test, test_target)][0]
    tmpX = principalComponents_test[tindex, :2].T
    tmpY = test_target[tindex].T
    print(tmpY[:20])
    print(tmpY.shape)
    print(tmpX.shape)
    plt.style.use('classic')
    plt.rcParams.update({'font.size': 50})
    ax.scatter(x=tmpX[0, :], y=tmpX[1, :], c=tmpY,
               cmap='Spectral', marker='o', s=40)  # Spectral
    plt.xlabel(f'PC 1')
    plt.ylabel(f'PC 2')
    plt.style.use('classic')
    tmpX = principalComponents_ood[:, :2].T
    ax.scatter(*tmpX, c='black', marker='.', s=0.5, label='OOD', alpha=0.1)
    plt.style.use('classic')
    xmin, ymin = np.min(principalComponents_test[tindex, 0].ravel()), np.min(
        principalComponents_test[tindex, 1].ravel())
    xmax, ymax = np.max(principalComponents_test[tindex, 0].ravel()), np.max(
        principalComponents_test[tindex, 1].ravel())
    ax.set_xlim(xmin, xmax)
    ax.set_xlim(ymin, ymax)
    plt.legend()
    matplotlib.rc('xtick', labelsize=40)
    matplotlib.rc('ytick', labelsize=40)
    lgnd = plt.legend(['ID', 'OOD'], numpoints=3, fontsize=30)
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    lgnd.legendHandles[1]._alpha = 1
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    plt.tight_layout()
    plt.savefig(
        path+f"ID_{in_dataset}_OOD_{out_dataset}_model_{model_type}.png", bbox_inches=None)
    return


def compute_NC5(id_data, ood_data, id_targets, num_classes=1000, device='cuda'):
    id_data = torch.from_numpy(id_data)
    ood_data = torch.from_numpy(ood_data)
    id_data = id_data.to(device)
    ood_data = ood_data.to(device)
    C = num_classes
    N = [0 for _ in range(num_classes+1)]
    mean = [0 for _ in range(num_classes+1)]

    for c in range(num_classes):
        idxs = (id_targets == c).nonzero()[0]
        if len(idxs) == 0:  # If no class-c in this batch
            continue
        h_c = id_data[idxs, :]
        mean[c] += torch.sum(h_c, dim=0)  #  CHW
        N[c] += h_c.shape[0]
    c = C
    h_c = ood_data
    mean[c] += torch.sum(h_c, dim=0)  #  CHW
    N[c] += h_c.shape[0]
    for c in range(C):
        mean[c] /= N[c]
    M = torch.stack(mean).T
    M = M.to(device)
    M_OOD = M[:, -1]
    M_OOD = M_OOD.reshape(-1, 1)
    M_OOD = M_OOD
    mu_OOD_norms = torch.norm(M_OOD,  dim=0)
    M_OOD = torch.squeeze(M_OOD)
    tmp_list = []
    for _ in range(num_classes):
        M_tmp = M[:, _]
        M_tmp_norms = torch.norm(M_tmp)
        M_tmp = torch.squeeze(M_tmp)
        prod_scalaire = torch.abs(
            torch.dot(M_tmp, M_OOD.T)/(mu_OOD_norms*M_tmp_norms))
        tmp_list.append(prod_scalaire.item())
    print(f" NC5 Value : {prod_scalaire}")
    return prod_scalaire


def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output
