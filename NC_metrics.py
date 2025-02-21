import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.models as models
from PIL import Image
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from tqdm import tqdm

import extract_utils


def get_default_params():
    parser_experiment = argparse.ArgumentParser()
    parser_experiment.add_argument("--model_architecture_type", choices = ["vit", "resnet"], default = "vit", help = "what type of model to use")
    parser_experiment.add_argument("--base_path", default = "./", help = "directory where the model is saved.")
    parser_experiment.add_argument("--save_path", default = "./", help = "directory where the features will be saved.")
    parser_experiment.add_argument("--data_path", default = "./", help = "directory where the datasets are saved.")
    parser_experiment.add_argument("--img_size", default = 224, type = int, help = "Resolution size")
    parser_experiment.add_argument("--local_rank", type = int, default = -1, help = "local_rank for distributed training on gpus")
    parser_experiment.add_argument("--model_type", choices = ["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_16-224", "ViT-L_32", "ViT-H_14",
                                                              "R50-ViT-B_16"], default = "ViT-B_16", help = "Which variant to use.")
    parser_experiment.add_argument('--seed', type = int, default = 42, help = "random seed for initialization")
    parser_experiment.add_argument("--dataset", choices = ["cifar10", "cifar100", "SVHN", "imagenet"], default = "cifar10", help = "Which downstream task.")
    parser_experiment.add_argument("--train_batch_size", default = 128, type = int, help = "Total batch size for training.")
    parser_experiment.add_argument("--eval_batch_size", default = 64, type = int, help = "Total batch size for eval.")
    parser_experiment.add_argument("--model_name", default = "-100_500_checkpoint.bin", help = "Which model to use.")
    parser_experiment.add_argument("--in_dataset", choices = ["cifar10", "cifar100", 'imagenet'], default = "cifar10", help = "Which downstream task is ID.")
    parser_experiment.add_argument("--out_dataset", choices = ["cifar10", "SUN", "places", "cifar100", "SVHN", 'tiny_imagenet', 'imagenet-o', 'imagenet-a', 'imagenet', 'texture', 'inaturalist',
                                                               'open-images'], default = "cifar100", help = "Which downstream task is OOD.")
    parser_experiment.add_argument("--cls_size", type = int, default = 768, help = "size of the class token to be used ")
    parser_experiment.add_argument("--class_index", type = int, default = 0, help = "s ")
    parser_experiment.add_argument("--save_preds", type = bool, default = False, help = "if set to True, recompute the models prediction and save them, else use saved predictions")
    parser_experiment.add_argument("--use_ood", type = bool, default = False, help = "use ood along ID data ")
    parser_experiment.add_argument("--add_noise", type = bool, default = False, help = "add_noise_to ID data  ")
    parser_experiment.add_argument("--add_noise_ID", type = bool, default = False, help = "add_noise_to ID data and treat it as ood  ")
    parser_experiment.add_argument("--per_class_noise", type = bool, default = False, help = "add_noise_to ID data and treat it as ood, one class at a time")
    parser_experiment.add_argument("--noise_value", type = float, default = 1.0, help = "add_noise_to ID data  ")
    parser_experiment.add_argument("--use_vit", type = bool, default = False, help = "Use vit for NC")
    parser_experiment.add_argument("--loss_name", choices = ["CrossEntropyLoss", "MSELoss"], default = "CrossEntropyLoss")
    return parser_experiment


def check_params(args_experiment):
    # judgement
    if not args_experiment.use_vit:
        if args_experiment.model_name not in ["resnet34", "resnet18"]:
            raise ValueError("when use_vit is False, model_name must  be 'resnet34' or 'resnet18'.")
    return


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            data = line.strip().rsplit(maxsplit = 1)
            if len(data) == 2:
                impath, imlabel = data
            else:
                impath, imlabel = data[0], 0
            imlist.append((impath, int(imlabel)))
    return imlist

# JSON file saving
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform = None, target_transform = None,
                 flist_reader = default_flist_reader, loader = default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def define_dataset_params(args_experiment):
    # dataset parameters
    # ofr mnist
    im_size = 28
    padded_im_size = 32

    input_ch = 3  # number of channels
    ood_n_classes = 0
    if args_experiment.in_dataset == 'mnist':
        input_ch = 1
        C = 10  # number of classes
    if args_experiment.in_dataset == "cifar10":
        C = 10
    elif args_experiment.in_dataset == "cifar100":
        C = 100
    elif args_experiment.in_dataset == "imagenet":
        C = 1000
    if args_experiment.out_dataset == 'mnist':
        input_ch = 1
        ood_n_classes = 10
    if args_experiment.out_dataset == "cifar10":
        ood_n_classes = 10
    elif args_experiment.out_dataset == "cifar100":
        ood_n_classes = 100
    elif args_experiment.out_dataset == "imagenet":
        ood_n_classes = 1000
    elif args_experiment.out_dataset == "inaturalist":
        ood_n_classes = 1
    elif args_experiment.out_dataset == "texture":
        ood_n_classes = 1
    elif args_experiment.out_dataset == "SUN":
        ood_n_classes = 1
    elif args_experiment.out_dataset == "places":
        ood_n_classes = 1
    elif args_experiment.out_dataset == "SVHN":
        ood_n_classes = 10
    if args_experiment.add_noise_ID:
        ood_n_classes = C
    return C, ood_n_classes, input_ch, im_size, padded_im_size


def train(model, criterion, device, num_classes, train_loader, optimizer, epoch):
    model.train()

    pbar = tqdm(total = len(train_loader), position = 0, leave = True)
    for batch_idx, (data, target) in enumerate(train_loader, start = 1):
        if data.shape[0] != batch_size:
            continue

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        if str(criterion) == 'CrossEntropyLoss()':
            loss = criterion(out, target)
        elif str(criterion) == 'MSELoss()':
            loss = criterion(out, F.one_hot(
                target, num_classes = num_classes).float())

        loss.backward()
        optimizer.step()

        accuracy = torch.mean(
            (torch.argmax(out, dim = 1) == target).float()).item()

        pbar.update(1)
        pbar.set_description(
            'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                accuracy))

        if debug and batch_idx > 20:
            break
    pbar.close()


def get_weight_bias(model):
    # elif 'vit' in model_type:
    model.eval()
    model_layers = extract_utils.nested_children(model)

    last_layer = model_layers['head']
    bias = last_layer.bias
    bias.requires_grad = False
    bias = bias
    weight = last_layer.weight
    weight.requires_grad = False
    weight = weight
    w, b = weight, bias
    print(f'{w.shape=}, {b.shape=}')

    return w, b


def ViT_NC_analysis(graphs, model, device, num_classes, csv_file_train, cls_size = 768):
    ########################   NC evaulation for ViT, using ID data only ########################
    model.eval()

    N = [0 for _ in range(C)]
    mean = [0 for _ in range(C)]
    Sw = 0

    loss = 0
    net_correct = 0
    NCC_match_net = 0

    # B CHW

    features_names = [f'ct{i}' for i in range(cls_size)]

    extracted_features = csv_file_train.loc[:, features_names].values
    labels = csv_file_train.loc[:, 'label'].values
    extracted_features = torch.from_numpy(extracted_features)
    labels = torch.from_numpy(labels)

    h = extracted_features
    target = labels

    for computation in ['Mean', 'Cov']:

        for c in range(C):
            # features belonging to class c
            idxs = (target == c).nonzero(as_tuple = True)[0]

            if len(idxs) == 0:  # If no class-c in this batch
                continue

            h_c = h[idxs, :]  # B CHW

            if computation == 'Mean':
                # update class means
                mean[c] += torch.sum(h_c, dim = 0)  #  CHW
                N[c] += h_c.shape[0]

            elif computation == 'Cov':
                # update within-class cov

                z = h_c - mean[c].unsqueeze(0)  # B CHW
                cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                    z.unsqueeze(1))  # B 1 CHW
                Sw += torch.sum(cov, dim = 0)
        if computation == 'Mean':

            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)
    # graphs.accuracy.append(net_correct/sum(N))
    # graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))
    # global mean
    M = M.to(device)
    muG = torch.mean(M, dim = 1, keepdim = True)  # CHW 1
    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C
    # avg norm
    # W  = classifier.weight
    W, bias = get_weight_bias(model)
    W = W.to(device = device)
    bias = bias.to(device = device)
    M_norms = torch.norm(M_, dim = 0)
    W_norms = torch.norm(W.T, dim = 0)
    graphs.norm_M_CoV.append((torch.std(M_norms) / torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms) / torch.mean(W_norms)).item())
    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    if num_classes > 100:  # If we are using ImageNet, the class token size is smaller than N classes, hence SVD wont work with N classes, and it  should be set to the vector size
        eigvec, eigval, _ = svds(Sb, k = cls_size - 1)
        inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    else:
        eigvec, eigval, _ = svds(Sb, k = C - 1)
        inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T

    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb) / C)

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    normalized_W = W.T / torch.norm(W.T, 'fro')
    # normalized_W=normalized_W.to(device)
    # # normalized_M=normalized_M.to(device)

    graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M) ** 2).item())

    # mutual coherence
    def coherence(V):

        G = V.T @ V
        G += torch.ones((C, C), device = device) / (C - 1)
        G -= torch.diag(torch.diag(G))
        res = torch.norm(G, 1).item() / (C * (C - 1))

        return res

    graphs.cos_M.append(coherence(M_ / M_norms))
    graphs.cos_W.append(coherence(W.T / W_norms))


def ViT_NC_analysis_plus_ood(graphs, model, device, num_classes, csv_file_train, start_epoch, csv_file_ood = None, epoch = -1, cls_size = 768):
    ########################   NC evaulation for ViT, using ID and OOD data ########################
    model.eval()
    # print(f" epoch { epoch}")
    if ood_loader is not None and epoch == start_epoch:
        global C
        C = C + 1
    else:
        num_classes -= 1
    N = [0 for _ in range(C)]
    mean = [0 for _ in range(C)]
    Sw = 0
    loss = 0
    net_correct = 0
    NCC_match_net = 0
    features_names = [f'ct{i}' for i in range(cls_size)]
    extracted_features = csv_file_train.loc[:, features_names].values
    labels = csv_file_train.loc[:, 'label'].values
    extracted_features = torch.from_numpy(extracted_features)
    labels = torch.from_numpy(labels)
    h = extracted_features
    target = labels
    extracted_features_ood = csv_file_ood.loc[:, features_names].values
    labels_ood = csv_file_ood.loc[:, 'label'].values
    labels_ood = np.full(labels_ood.shape, num_classes)
    extracted_features_ood = torch.from_numpy(extracted_features_ood)
    labels_ood = torch.from_numpy(labels_ood)
    h_ood = extracted_features_ood
    target_ood = labels_ood
    for computation in ['Mean', 'Cov']:
        for c in range(C - 1):
            # features belonging to class c
            idxs = (target == c).nonzero(as_tuple = True)[0]
            if len(idxs) == 0:  # If no class-c in this batch
                continue
            h_c = h[idxs, :]  # B CHW
            if computation == 'Mean':
                # update class means
                mean[c] += torch.sum(h_c, dim = 0)  #  CHW
                N[c] += h_c.shape[0]
            elif computation == 'Cov':
                # update within-class cov
                z = h_c - mean[c].unsqueeze(0)  # B CHW
                cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                    z.unsqueeze(1))  # B 1 CHW
                Sw += torch.sum(cov, dim = 0)
        for c in range(C):
            idxs = (target_ood == c).nonzero(as_tuple = True)[0]
            if len(idxs) == 0:  # If no class-c in this batch
                continue
            h_c = h_ood[idxs, :]  # B CHW
            if computation == 'Mean':
                # update class means
                mean[c] += torch.sum(h_c, dim = 0)  #  CHW
                N[c] += h_c.shape[0]
            elif computation == 'Cov':
                # update within-class cov
                z = h_c - mean[c].unsqueeze(0)  # B CHW
                cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                    z.unsqueeze(1))  # B 1 CHW
                Sw += torch.sum(cov, dim = 0)
                # during calculation of within-class covariance, calculate:
                # 1) network's accuracy
                # net_pred = torch.argmax(output[idxs,:], dim=1)
                # net_correct += sum(net_pred==target[idxs]).item()

                # # 2) agreement between prediction and nearest class center
                # print(f" h_c[i,:] shape {h_c[i,:].shape}    M shape { M.shape}     ")
                # NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                #                         for i in range(h_c.shape[0])])
                # NCC_pred = torch.argmin(NCC_scores, dim=1)
                # NCC_match_net += sum(NCC_pred==net_pred).item()

        if computation == 'Mean':
            print(N)
            print(C)
            for c in range(C):
                mean[c] /= N[c]
            M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)
    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct / sum(N))
    graphs.NCC_mismatch.append(1 - NCC_match_net / sum(N))
    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param ** 2).item()
    graphs.reg_loss.append(reg_loss)

    # global mean

    M = M.to(device)
    M_OOD = M[:, -1]
    M_OOD = M_OOD.reshape(-1, 1)
    M_ID_only = M[:, :-1]
    muG = torch.mean(M_ID_only, dim = 1, keepdim = True)
    M_ = M - muG
    M_OOD = M_OOD
    prod_scalaire_list = []
    mu_OOD_norms = torch.norm(M_OOD, dim = 0)
    M_OOD = torch.squeeze(M_OOD)
    tmp_list = []
    #  NC5 computation
    for _ in range(num_classes):
        M_tmp = M[:, _]
        M_tmp_norms = torch.norm(M_tmp)
        M_tmp = torch.squeeze(M_tmp)
        prod_scalaire = torch.abs(
            torch.dot(M_tmp, M_OOD.T) / (mu_OOD_norms * M_tmp_norms))
        tmp_list.append(prod_scalaire.item())
    prod_scalaire_list.append(np.mean(np.array(tmp_list)))
    graphs.NC_ortho.append(prod_scalaire_list)
    Sb = torch.matmul(M_, M_.T) / C
    W, bias = get_weight_bias(model)
    W = W.to(device = device)
    bias = bias.to(device = device)
    M_norms = torch.norm(M_, dim = 0)
    W_norms = torch.norm(W.T, dim = 0)
    # M_norms = torch.norm(M_,  dim=0)
    # W_norms = torch.norm(W.T, dim=0)
    graphs.norm_M_CoV.append((torch.std(M_norms) / torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms) / torch.mean(W_norms)).item())
    # De
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    print(f" my num classes {C}, my N {N}")
    if num_classes > 100:
        print(f"we are here   {cls_size}")
        eigvec, eigval, _ = svds(Sb, k = cls_size - 1)
        inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    else:
        eigvec, eigval, _ = svds(Sb, k = C - 1)
        inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T

    # print(f"my sw shape {Sw.shape} inv_sb shape { inv_Sb.shape}")
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb) / C)

    # graphs.sigma_W.append(np.trace(Sw )/C)
    # graphs.sigma_B.append(np.trace(inv_Sb )/C)
    # ||W^T - M_||
    # normalized_M = M_ / torch.norm(M_,'fro')
    # normalized_W = W.T / torch.norm(W.T,'fro')
    # mutual coherence

    def coherence(V, C = C):

        G = V.T @ V
        G += torch.ones((C, C), device = device) / (C - 1)
        G -= torch.diag(torch.diag(G))
        res = torch.norm(G, 1).item() / (C * (C - 1))

        return res

    graphs.cos_M.append(coherence(M_ / M_norms, C = C))


def analysis(graphs, model, criterion_summed, device, num_classes, loader, add_noise = False, noise_value = 1.0):
    ########################   NC evaulation for ResNet, using ID data only ########################

    model.eval()

    N = [0 for _ in range(C)]
    mean = [0 for _ in range(C)]
    Sw = 0

    loss = 0
    net_correct = 0
    NCC_match_net = 0

    for computation in ['Mean', 'Cov']:
        pbar = tqdm(total = len(loader), position = 0, leave = True)
        for batch_idx, (data, target) in enumerate(loader, start = 1):

            data, target = data.to(device), target.to(device)
            if add_noise:
                noise = torch.randn(data.shape).to(device)
                data = data + noise / noise_value

            output = model(data)
            h = features.value.data.view(data.shape[0], -1)  # B CHW

            # during calculation of class means, calculate loss
            if computation == 'Mean':
                if str(criterion_summed) == 'CrossEntropyLoss()':
                    loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == 'MSELoss()':
                    loss += criterion_summed(output, F.one_hot(target,
                        num_classes = num_classes).float()).item()

            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple = True)[0]

                if len(idxs) == 0:  # If no class-c in this batch
                    continue

                h_c = h[idxs, :]  # B CHW

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim = 0)  #  CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0)  # B CHW
                    cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                        z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim = 0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs, :], dim = 1)
                    net_correct += sum(net_pred == target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i, :] - M.T, dim = 1)
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim = 1)
                    NCC_match_net += sum(NCC_pred == net_pred).item()

            pbar.update(1)
            pbar.set_description(
                'Analysis {}\t'
                'Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    computation,
                    epoch,
                    batch_idx,
                    len(loader),
                    100. * batch_idx / len(loader)))

            if debug and batch_idx > 20:
                break
        pbar.close()

        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)

    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct / sum(N))
    graphs.NCC_mismatch.append(1 - NCC_match_net / sum(N))

    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param ** 2).item()
    graphs.reg_loss.append(reg_loss)

    # global mean
    muG = torch.mean(M, dim = 1, keepdim = True)  # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # avg norm
    W = classifier.weight
    M_norms = torch.norm(M_, dim = 0)
    W_norms = torch.norm(W.T, dim = 0)

    graphs.norm_M_CoV.append((torch.std(M_norms) / torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms) / torch.mean(W_norms)).item())

    # Decomposition of MSE #
    if loss_name == 'MSELoss':
        wd = 0.5 * weight_decay  # "\lambda" in manuscript, so this is halved
        St = Sw + Sb
        size_last_layer = Sb.shape[0]
        eye_P = torch.eye(size_last_layer).to(device)
        eye_C = torch.eye(C).to(device)

        St_inv = torch.inverse(St + (wd / (wd + 1)) * (muG @ muG.T) + wd * eye_P)

        w_LS = 1 / C * (M.T - 1 / (1 + wd) * muG.T) @ St_inv
        b_LS = (1 / C * torch.ones(C).to(device) -
                w_LS @ muG.T.squeeze(0)) / (1 + wd)
        w_LS_ = torch.cat([w_LS, b_LS.unsqueeze(-1)], dim = 1)  # c x n
        b = classifier.bias
        w_ = torch.cat([W, b.unsqueeze(-1)], dim = 1)  # c x n

        LNC1 = 0.5 * (torch.trace(w_LS @ (Sw + wd * eye_P)
                                  @ w_LS.T) + wd * torch.norm(b_LS) ** 2)
        LNC23 = 0.5 / C * torch.norm(w_LS @ M + b_LS.unsqueeze(1) - eye_C) ** 2

        A1 = torch.cat([St + muG @ muG.T + wd * eye_P, muG], dim = 1)
        A2 = torch.cat([muG.T, torch.ones([1, 1]).to(device) + wd], dim = 1)
        A = torch.cat([A1, A2], dim = 0)
        Lperp = 0.5 * torch.trace((w_ - w_LS_) @ A @ (w_ - w_LS_).T)

        MSE_wd_features = loss + 0.5 * weight_decay * \
                          (torch.norm(W) ** 2 + torch.norm(b) ** 2).item()
        MSE_wd_features *= 0.5

        graphs.MSE_wd_features.append(MSE_wd_features)
        graphs.LNC1.append(LNC1.item())
        graphs.LNC23.append(LNC23.item())
        graphs.Lperp.append(Lperp.item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k = C - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb) / C)

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    normalized_W = W.T / torch.norm(W.T, 'fro')

    # mutual coherence
    def coherence(V):

        G = V.T @ V
        G += torch.ones((C, C), device = device) / (C - 1)
        G -= torch.diag(torch.diag(G))
        res = torch.norm(G, 1).item() / (C * (C - 1))

        return res

    graphs.cos_M.append(coherence(M_ / M_norms))
    graphs.cos_W.append(coherence(W.T / W_norms))


def analysis_plus_ood(graphs, model, criterion_summed, device, num_classes, loader, ood_loader = None, epoch = -1, add_noise_ID = False, noise_value = 1.0):
    ########################   NC evaulation for ResNet, using ID and OOD data ########################

    model.eval()
    if ood_loader is not None and epoch == 1:
        global C
        C = C + 1
    N = [0 for _ in range(C)]
    mean = [0 for _ in range(C)]
    Sw = 0
    loss = 0
    net_correct = 0
    NCC_match_net = 0

    for computation in ['Mean', 'Cov']:
        pbar = tqdm(total = len(loader), position = 0, leave = True)
        for batch_idx, (data, target) in enumerate(loader, start = 1):

            data, target = data.to(device), target.to(device)
            # print(f"id target{target} ")

            output = model(data)
            h = features.value.data.view(data.shape[0], -1)  # B CHW

            # during calculation of class means, calculate loss
            if computation == 'Mean':
                if str(criterion_summed) == 'CrossEntropyLoss()':
                    loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == 'MSELoss()':
                    loss += criterion_summed(output, F.one_hot(target,
                        num_classes = num_classes).float()).item()
            # print(f" C value right now {C} at epoch ")

            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple = True)[0]

                if len(idxs) == 0:  # If no class-c in this batch
                    continue

                h_c = h[idxs, :]  # B CHW

                if computation == 'Mean':
                    # update class means

                    mean[c] += torch.sum(h_c, dim = 0)  #  CHW

                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0)  # B CHW
                    cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                        z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim = 0)
                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs, :], dim = 1)
                    net_correct += sum(net_pred == target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i, :] - M.T, dim = 1)
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim = 1)
                    NCC_match_net += sum(NCC_pred == net_pred).item()

            pbar.update(1)
            pbar.set_description(
                'Analysis {}\t'
                'Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    computation,
                    epoch,
                    batch_idx,
                    len(loader),
                    100. * batch_idx / len(loader)))

            if debug and batch_idx > 20:
                break
        pbar.close()
        pbar = tqdm(total = len(ood_loader), position = 0, leave = True)
        if ood_loader:
            for batch_idx, (data, target) in enumerate(ood_loader, start = 1):
                # TODO: here we set all targets to to K+1 ,
                target = torch.full(target.shape, C - 1)
                data, target = data.to(device), target.to(device)
                if add_noise_ID:
                    noise = torch.randn(data.shape).to(device)
                    data = data + noise / noise_value

                output = model(data)
                h = features.value.data.view(data.shape[0], -1)  # B CHW
                c = C - 1
                idxs = (target == c).nonzero(as_tuple = True)[0]
                if len(idxs) == 0:  # If no class-c in this batch
                    continue
                h_c = h[idxs, :]  # B CHW
                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim = 0)  #  CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0)  # B CHW
                    cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                        z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim = 0)
                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs, :], dim = 1)
                    net_correct += sum(net_pred == target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i, :] - M.T, dim = 1)
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim = 1)
                    NCC_match_net += sum(NCC_pred == net_pred).item()

                pbar.update(1)
                pbar.set_description(
                    'Analysis {}\t'
                    'Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        computation,
                        epoch,
                        batch_idx,
                        len(ood_loader),
                        100. * batch_idx / len(ood_loader)))

                if debug and batch_idx > 20:
                    break
            pbar.close()

        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
            M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)
    print(f" my N == {N}")
    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct / sum(N))
    graphs.NCC_mismatch.append(1 - NCC_match_net / sum(N))
    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param ** 2).item()
    graphs.reg_loss.append(reg_loss)

    M = M.to(device)
    M_OOD = M[:, -1]
    M_OOD = M_OOD.reshape(-1, 1)
    M_ID_only = M[:, :-1]
    muG = torch.mean(M_ID_only, dim = 1, keepdim = True)
    M_ = M - muG
    M_OOD = M_OOD
    prod_scalaire_list = []
    mu_OOD_norms = torch.norm(M_OOD, dim = 0)
    M_OOD = torch.squeeze(M_OOD)
    tmp_list = []
    #  NC5 computation
    for _ in range(num_classes):
        M_tmp = M[:, _]
        M_tmp_norms = torch.norm(M_tmp)
        M_tmp = torch.squeeze(M_tmp)
        prod_scalaire = torch.abs(
            torch.dot(M_tmp, M_OOD.T) / (mu_OOD_norms * M_tmp_norms))
        # print(f" prod scaleir {prod_scalaire.item()}")
        tmp_list.append(prod_scalaire.item())
    prod_scalaire_list.append(np.mean(np.array(tmp_list)))
    graphs.NC_ortho.append(prod_scalaire_list)
    Sb = torch.matmul(M_, M_.T) / C
    W = classifier.weight
    M_norms = torch.norm(M_, dim = 0)
    W_norms = torch.norm(W.T, dim = 0)
    graphs.norm_M_CoV.append((torch.std(M_norms) / torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms) / torch.mean(W_norms)).item())
    if loss_name == 'MSELoss':
        wd = 0.5 * weight_decay  # "\lambda" in manuscript, so this is halved
        St = Sw + Sb
        size_last_layer = Sb.shape[0]
        eye_P = torch.eye(size_last_layer).to(device)
        eye_C = torch.eye(C).to(device)

        St_inv = torch.inverse(St + (wd / (wd + 1)) * (muG @ muG.T) + wd * eye_P)

        w_LS = 1 / C * (M.T - 1 / (1 + wd) * muG.T) @ St_inv
        b_LS = (1 / C * torch.ones(C).to(device) -
                w_LS @ muG.T.squeeze(0)) / (1 + wd)
        w_LS_ = torch.cat([w_LS, b_LS.unsqueeze(-1)], dim = 1)  # c x n
        b = classifier.bias
        w_ = torch.cat([W, b.unsqueeze(-1)], dim = 1)  # c x n

        LNC1 = 0.5 * (torch.trace(w_LS @ (Sw + wd * eye_P)
                                  @ w_LS.T) + wd * torch.norm(b_LS) ** 2)
        LNC23 = 0.5 / C * torch.norm(w_LS @ M + b_LS.unsqueeze(1) - eye_C) ** 2

        A1 = torch.cat([St + muG @ muG.T + wd * eye_P, muG], dim = 1)
        A2 = torch.cat([muG.T, torch.ones([1, 1]).to(device) + wd], dim = 1)
        A = torch.cat([A1, A2], dim = 0)
        Lperp = 0.5 * torch.trace((w_ - w_LS_) @ A @ (w_ - w_LS_).T)

        MSE_wd_features = loss + 0.5 * weight_decay * \
                          (torch.norm(W) ** 2 + torch.norm(b) ** 2).item()
        MSE_wd_features *= 0.5

        graphs.MSE_wd_features.append(MSE_wd_features)
        graphs.LNC1.append(LNC1.item())
        graphs.LNC23.append(LNC23.item())
        graphs.Lperp.append(Lperp.item())
        # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k = C - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb) / C)

    # ||W^T - M_||
    # normalized_M = M_ / torch.norm(M_,'fro')
    # normalized_W = W.T / torch.norm(W.T,'fro')

    # mutual coherenceood_n_classes
    def coherence(V, C = C):

        G = V.T @ V
        G += torch.ones((C, C), device = device) / (C - 1)
        G -= torch.diag(torch.diag(G))
        res = torch.norm(G, 1).item() / (C * (C - 1))
        return res

    graphs.cos_M.append(coherence(M_ / M_norms, C = C))
    graphs.cos_W.append(coherence(W.T / W_norms, C = C - 1))


class features:
    pass


def hook(self, input, output):
    features.value = input[0].clone()


def load_data(args, dataset_name):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    if dataset_name == "mnist":
        transform = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.1307, 0.3081)])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(f'{args.data_path}/mnist', train = True,
                download = True, transform = transform),
            batch_size = batch_size, shuffle = True)

        analysis_loader = torch.utils.data.DataLoader(
            datasets.MNIST(f'{args.data_path}/mnist', train = True,
                download = True, transform = transform),
            batch_size = batch_size, shuffle = True)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root = f"{args.data_path}/cifar10", train = True, download = False, transform = transform)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)
        analysis_loader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)
    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root = f"{args.data_path}/cifar100", train = True, download = False, transform = transform)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)
        analysis_loader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)
    elif dataset_name == "SVHN":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN(
            root = f"{args.data_path}/SVHN", split = 'test', download = False, transform = transform)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)
        analysis_loader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)
    else:
        if dataset_name == 'imagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop((384, 384), scale = (0.05, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
            ])
            transform_test = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
            ])
            transform = transform_test
            trainset = torchvision.datasets.ImageNet(
                f"{args.data_path}/train", 'train', transform = transform_train)

        if dataset_name == 'inaturalist':
            data_path = f'{args.data_path}/inaturalist/iNaturalist'
            trainset = None

            full_dataset = torchvision.datasets.ImageFolder(
                root = data_path,
                transform = transform
            )
            train_size = int(1.0 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            generator = torch.Generator()
            generator.manual_seed(0)

            trainset, _ = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator = generator)
        elif dataset_name == 'texture':
            data_path = f'{args.data_path}/texture/dtd/images'
            img_list = "dataset_samples/texture.txt"

            full_dataset = ImageFilelist(
                data_path, img_list, transform = transform)
            train_size = int(1.0 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            print(f" my dataset length {len(full_dataset)}")
            generator = torch.Generator()
            generator.manual_seed(0)
            trainset, _ = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator = generator)
        elif dataset_name == 'imagenet-o':
            imagenet_o_folder = f'{args.data_path}/imagenet-o'
            val_examples_imagenet_o = datasets.ImageFolder(
                root = imagenet_o_folder, transform = transform)
            trainset, _ = val_examples_imagenet_o, None
        elif dataset_name == 'places':
            imagenet_o_folder = f'{args.data_path}/places/images'
            val_examples_imagenet_o = datasets.ImageFolder(
                root = imagenet_o_folder, transform = transform)
            trainset, _ = val_examples_imagenet_o, None
        elif dataset_name == 'SUN':
            imagenet_o_folder = f'{args.data_path}/SUN'
            val_examples_imagenet_o = datasets.ImageFolder(
                root = imagenet_o_folder, transform = transform)
            trainset, _ = val_examples_imagenet_o, None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)
        analysis_loader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)
    return train_loader, analysis_loader


class graphs:
    def __init__(self):
        self.accuracy = []
        self.loss = []
        self.reg_loss = []

        # NC1
        self.Sw_invSb = []

        # NC2
        self.norm_M_CoV = []
        self.norm_W_CoV = []
        self.cos_M = []
        self.cos_W = []

        self.St_tr = []
        self.Sb = []
        self.Sw = []

        # NC3
        self.W_M_dist = []

        # NC4
        self.NCC_mismatch = []

        # NC5
        self.NC_ortho = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []


if __name__ == '__main__':
    # parameter settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser_experiment = get_default_params()
    args_experiment, unknown = parser_experiment.parse_known_args()

    '''modifications'''
    # changes the parameters for different experiments
    debug = False  # Only runs 20 batches per epoch for debugging
    epochs = 350
    args_experiment.model_architecture_type = "resnet"  # model architecture, defaultly use vit, choose from: ["vit", "resnet"]
    args_experiment.base_path = "./"  # base path to save the model
    args_experiment.save_path = "./"  # paht to save the characteristics
    args_experiment.data_path = "./data"  # path to save the dataset
    args_experiment.img_size = 224  # image size(dpi), defaultly 224
    args_experiment.local_rank = -1  # local GPU index for distributed training
    args_experiment.model_type = "ViT-B_16"  # Specific ViTs, defaultly use ViT-B_16, choose from: ["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_16-224", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"]
    args_experiment.dataset = "cifar10"  # type of data set used by the downstream task, defaultly use cifar10, choose from: ["cifar10", "cifar100", "SVHN", "imagenet"]
    args_experiment.train_batch_size = 128  # total batch size for training, defaultly 128
    args_experiment.eval_batch_size = 64  # total batch size for evaluation, defaultly 64
    args_experiment.model_name = "resnet18"  # model name
    args_experiment.in_dataset = "cifar10"  # ID dataset, defaultly cifar10, choose from: ["cifar10", "cifar100", "imagenet"]
    args_experiment.out_dataset = "cifar100"  # OOD dataset, defaultly cifar100, choose from: ["cifar10", "SUN", "places", "cifar100", "SVHN", "tiny_imagenet", "imagenet-o", "imagenet-a", "imagenet", "texture", "inaturalist", "open-images"]
    args_experiment.cls_size = 768  # class size 768
    args_experiment.class_index = 0  # class index 0
    args_experiment.save_preds = False  # recompute model prediction and save? defaultly False
    args_experiment.use_ood = True  # use OOD datas in ID? defaultly False
    args_experiment.use_vit = False  # use ViT in NC? defaultly False, then model_name must be "resnet34" or "resnet18"
    print(f" my args : {args_experiment}")

    # return number of classes, number of OOD classes, channels, original image size and padded image size
    C, ood_n_classes, input_ch, im_size, padded_im_size = define_dataset_params(args_experiment)
    print(f" ood n classes {ood_n_classes}")
    loss_name = args_experiment.loss_name
    '''end'''

    # train parameters
    # Optimization hyperparameters
    lr_decay = 0.1
    lr = 0.0679

    epochs_lr_decay = [epochs // 3, epochs * 2 // 3]
    batch_size = 128
    momentum = 0.9
    weight_decay = 5e-4

    # analysis parameters (figure generation list)
    epoch_list = list(range(11)) + list(range(10, epochs + 1, 5)) + epochs_lr_decay
    epoch_list_plot = [1, 5, 50, 250, epochs - 1, epochs, epochs + 1] + epochs_lr_decay

    # training and evaluation data, same one
    train_loader, analysis_loader = load_data(args_experiment, args_experiment.in_dataset)
    print(f" analysis_loader len {len(analysis_loader)}     in data {args_experiment.in_dataset}")

    '''check'''
    check_params(args_experiment)

    if not args_experiment.use_vit:
        print("don't use VIT")
        if args_experiment.model_name == "resnet34":
            model = models.resnet34(pretrained = False, num_classes = C)
        elif args_experiment.model_name == "resnet18":
            model = models.resnet18(pretrained = False, num_classes = C)
            # Small dataset filter size used by He et al. (2015)
            model.conv1 = nn.Conv2d(
                input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
            model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
            model = model.to(device)
            # register hook that saves last-layer input into features
            classifier = model.fc
            classifier.register_forward_hook(hook)

    # odd parameters
    if args_experiment.use_ood:
        if args_experiment.add_noise_ID:
            ood_loader, ood_analysis_loader = load_data(args_experiment, args_experiment.in_dataset)
        else:
            ood_loader, ood_analysis_loader = load_data(args_experiment, args_experiment.out_dataset)

    if not args_experiment.use_vit:
        if loss_name == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
            criterion_summed = nn.CrossEntropyLoss(reduction = 'sum')
        elif loss_name == 'MSELoss':
            criterion = nn.MSELoss()
            criterion_summed = nn.MSELoss(reduction = 'sum')

        optimizer = optim.SGD(model.parameters(),
            momentum = momentum,
            lr = lr,
            weight_decay = weight_decay)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
            milestones = epochs_lr_decay,
            gamma = lr_decay)

    graphs = graphs()
    result_dict = dict()

    cur_epochs = []

    '''vit'''
    if args_experiment.use_vit:
        step_size = 390
        step_max = 6000
        dataset_steps = list(range(step_size, step_max, step_size))
        for curr_step in dataset_steps:
            if args_experiment.out_dataset == 'cifar10':
                ood_n_classes = 10
            if args_experiment.out_dataset == 'cifar100':
                ood_n_classes = 100
            if args_experiment.out_dataset == 'SVHN':
                ood_n_classes = 10
            cur_epochs.append(curr_step)
            save_path_vit = f'figs/{args_experiment.in_dataset}'
            if args_experiment.use_ood:
                save_path_vit = f'figs/{args_experiment.in_dataset}/{args_experiment.out_dataset}'
            args_experiment.model_path = f"{args_experiment.base_path}/{args_experiment.model_name}_step{curr_step}.bin"
            csvFile_train, csvFile_ood, model = extract_utils.load_model_and_data_train(
                args_experiment, args_experiment, model_train_evaluating = True, eval_step = curr_step, extract_ood = True, per_class_noise = args_experiment.per_class_noise)
            if args_experiment.use_ood:
                ViT_NC_analysis_plus_ood(graphs, model, device, C, csvFile_train,
                    csv_file_ood = csvFile_ood, start_epoch = step_size, epoch = curr_step, cls_size = 768)
            else:
                print(" \n \n \n using ID _ only \n \n \n ")
                ViT_NC_analysis(graphs, model, device, C,
                    csvFile_train, args_experiment.cls_size)
            plt.figure(1)

            plt.plot(cur_epochs, graphs.norm_M_CoV, color = 'magenta')
            plt.plot(cur_epochs, graphs.norm_W_CoV, color = 'g')
            plt.legend(['Class Means', 'Classifiers'])
            plt.xlabel('Epoch')
            plt.ylabel('Std/Avg of Norms')
            plt.title(f'NC2: Equinorm {args_experiment.in_dataset}')
            plt.savefig(f'{save_path_vit}/NC2 Equinorm VIT  {curr_step} ')

            result_dict["norm_M_CoV"] = list(graphs.norm_M_CoV)
            result_dict["norm_W_CoV"] = list(graphs.norm_W_CoV)
            plt.figure(2)

            plt.plot(cur_epochs, graphs.sigma_B,color='magenta')
            plt.plot(cur_epochs, graphs.sigma_W,color='g')
            plt.legend(['sigma_B-1','sigma_W'])
            plt.xlabel('Epoch')
            plt.ylabel('intra/inter covariance')
            plt.title(f' sigma_W/sigma_B-1 {args_experiment.in_dataset}')
            plt.savefig(f'{save_path_vit}/sigma_W {curr_step} ')

            result_dict["sigma_B"]=list(graphs.sigma_B)
            result_dict["sigma_W"]=list(graphs.sigma_W)

            result_dict["M"]=list(graphs.M)
            result_dict["M_OOD"]=list(graphs.M_OOD)

            plt.figure(3)
            if (args_experiment.use_ood) or args_experiment.per_class_noise:
                # for _ in range(C-1):
                #     class_c_metrics=[ element[_] for element in graphs.NC_ortho ]
                plt.plot(cur_epochs, graphs.NC_ortho, color = 'magenta')
                plt.xlabel('steps')
                plt.ylabel('NC5: ID/OOD orthogonality')
                plt.title(f' NC5 {args_experiment.in_dataset}')
                plt.savefig(f'{save_path_vit}/NC5  {curr_step} ')

                result_dict["NC5"] = list(graphs.NC_ortho)
            plt.figure(4)

            plt.semilogy(cur_epochs, graphs.Sw_invSb, color = 'b')
            plt.xlabel('Epoch')
            plt.ylabel('Tr{Sw Sb^-1}')
            plt.title(f'NC1: Activation Collapse {args_experiment.in_dataset}')

            plt.savefig(f'{save_path_vit}/NC1 Activation Collapse  {curr_step}')
            result_dict["Tr{Sw Sb^-1}"] = list(graphs.Sw_invSb)
            plt.figure(5)
            plt.plot(cur_epochs, graphs.cos_M, color = 'g')
            plt.legend(['Class Means'])
            plt.xlabel('Epoch')
            plt.ylabel('Avg|Cos + 1/(C-1)|')
            plt.title(f'NC2: Maximal Equiangularity {args_experiment.in_dataset} ')
            result_dict["cos_M"] = list(graphs.cos_M)
            plt.savefig(f'{save_path_vit}/NC2 Maximal Equiangularity  {curr_step}')
            if args_experiment.use_ood == False:  # if we use OOD, W and M have diffferent sizes
                if curr_step > 5000:
                    plt.figure(6)
                    plt.plot(cur_epochs, graphs.W_M_dist, color = 'c')
                    plt.xlabel('Epoch')
                    plt.ylabel('||W^T - H||^2')
                    plt.title(f'NC3: Self Duality {args_experiment.in_dataset}')
                    plt.savefig(f'{save_path_vit}/NC3 Self Duality {curr_step}')
                result_dict["W_M_dist"] = list(graphs.W_M_dist)

            plt.figure(7)
            plt.plot(cur_epochs,graphs.NCC_mismatch)
            plt.xlabel('Epoch')
            plt.ylabel('Proportion Mismatch from NCC')
            plt.title(f'NC4: Convergence to NCC {args_experiment.in_dataset}')
            result_dict["NCC_mismatch"]=list( graphs.NCC_mismatch)

            plt.savefig(f'{save_path_vit}/NC4 Convergence to NCC')

            path = f'{save_path_vit}/vit_{args_experiment.in_dataset}_{args_experiment.out_dataset}.txt'
            if args_experiment.use_ood:
                save_path_root = f'figs/{args_experiment.in_dataset}'
                ood_case = "K_plus_one"
                path = f'{save_path_root}/output_ID_{args_experiment.in_dataset}_OOD_{args_experiment.out_dataset}_case_{ood_case}.txt'

    else:
        class_index = args_experiment.class_index
        save_path_root = f'figs/{args_experiment.in_dataset}'
        if not os.path.isdir(save_path_root):
            os.makedirs(save_path_root, exist_ok = True)
        result_dict["lr_reduction_steps"] = epochs_lr_decay
        result_dict["N_epochs"] = epochs
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            train(model, criterion, device, C, train_loader, optimizer, epoch)
            lr_scheduler.step()

            if epoch in epoch_list:

                cur_epochs.append(epoch)
                if args_experiment.use_ood:
                    print(" analysing K+1 case")
                    print(" analysing ID and OOD")
                    '''analysis codes'''
                    analysis_plus_ood(graphs, model, criterion_summed, device, C, analysis_loader, ood_analysis_loader,
                        epoch = epoch, add_noise_ID = args_experiment.add_noise_ID, noise_value = args_experiment.noise_value)
                else:
                    print(" analysing ID only")
                    analysis(graphs, model, criterion_summed, device, C, analysis_loader,
                        add_noise = args_experiment.add_noise, noise_value = args_experiment.noise_value)
                plt.figure(1)
                plt.semilogy(cur_epochs, graphs.reg_loss, color = 'r')
                plt.legend(['Loss + Weight Decay'])
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.title(f'Training Loss {args_experiment.in_dataset}')
                if epoch in epoch_list_plot:
                    if args_experiment.use_ood:
                        plt.savefig(
                            f'{save_path_root}/Training Loss  {epoch} , {args_experiment.in_dataset}_{args_experiment.loss_name}_OOD_dateset_{args_experiment.out_dataset}_{args_experiment.model_name}')
                    else:
                        plt.savefig(
                            f'{save_path_root}/Training Loss  {epoch} , {args_experiment.in_dataset}_{args_experiment.loss_name}_{args_experiment.model_name}')
                result_dict["train_loss"] = list(graphs.reg_loss)
                plt.figure(2)
                plt.plot(cur_epochs, 100 * (1 - np.array(graphs.accuracy)), color = 'g')

                plt.xlabel('Epoch')
                plt.ylabel('Training Error (%)')
                plt.title(f'Training Error {args_experiment.in_dataset}')
                if epoch in epoch_list_plot:

                    if args_experiment.use_ood:
                        plt.savefig(
                            f'{save_path_root}/Training Error  {epoch}, {args_experiment.in_dataset}_{args_experiment.loss_name}_OOD_dateset_{args_experiment.out_dataset}_{args_experiment.model_name}')
                    else:
                        plt.savefig(
                            f'{save_path_root}/Training Error  {epoch}, {args_experiment.in_dataset}_{args_experiment.loss_name}_{args_experiment.model_name}')
                plt.figure(3)
                result_dict["train_err"] = list(100 * (1 - np.array(graphs.accuracy)))
                print(f" graphs.norm_M_CoV {graphs.norm_M_CoV}")
                plt.plot(cur_epochs, graphs.norm_M_CoV, color = 'magenta')
                plt.legend(['Class Means', 'Classifiers'])
                plt.xlabel('Epoch')
                plt.ylabel('Std/Avg of Norms')
                plt.title(f'NC2: Equinorm {args_experiment.in_dataset}')

                if epoch in epoch_list_plot:
                    if args_experiment.use_ood:
                        plt.savefig(
                            f'{save_path_root}/NC2 Equinorm {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_OOD_dateset_{args_experiment.out_dataset}_{args_experiment.model_name}')
                    else:
                        plt.savefig(
                            f'{save_path_root}/NC2 Equinorm {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_{args_experiment.model_name}')
                result_dict["norm_M_CoV"] = list(graphs.norm_M_CoV)

                plt.figure(10)
                if args_experiment.use_ood:
                    plt.plot(cur_epochs, graphs.NC_ortho, color = 'magenta')
                    plt.xlabel('Epoch')
                    plt.ylabel('NC5: ID/OOD orthogonality')
                    plt.title(f' NC5 {args_experiment.in_dataset}')
                    plt.savefig(f'{save_path_root}/NC5  {epoch} ')
                    result_dict["NC5"] = list(graphs.NC_ortho)
                    result_dict["St_tr"] = list(graphs.St_tr)
                plt.figure(4)
                plt.semilogy(cur_epochs, graphs.Sw_invSb, color = 'b')
                plt.xlabel('Epoch')
                plt.ylabel('Tr{Sw Sb^-1}')
                plt.title(f'NC1: Activation Collapse {args_experiment.in_dataset}')

                if epoch in epoch_list_plot:
                    if args_experiment.use_ood:
                        plt.savefig(
                            f'{save_path_root}/NC1 Activation Collapse {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_OOD_dateset_{args_experiment.out_dataset}_{args_experiment.model_name}')
                    else:
                        plt.savefig(
                            f'{save_path_root}/NC1 Activation Collapse {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_{args_experiment.model_name}')
                result_dict["Tr{Sw Sb^-1}"] = list(graphs.Sw_invSb)
                plt.figure(5)
                plt.plot(cur_epochs, graphs.cos_M, color = 'g')

                plt.legend(['Class Means', 'Classifiers'])
                plt.xlabel('Epoch')
                plt.ylabel('Avg|Cos + 1/(C-1)|')
                plt.title(f'NC2: Maximal Equiangularity {args_experiment.in_dataset}')
                result_dict["cos_M"] = list(graphs.cos_M)

                if epoch in epoch_list_plot:
                    if args_experiment.use_ood:
                        plt.savefig(
                            f'{save_path_root}/NC2 Maximal Equiangularity {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_OOD_dateset_{args_experiment.out_dataset}_{args_experiment.model_name}')
                    else:
                        plt.savefig(
                            f'{save_path_root}/NC2 Maximal Equiangularity {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_{args_experiment.model_name}')

                if not args_experiment.use_ood:  # if we use OOD, W and M have diffferent sizes
                    plt.figure(6)
                    plt.plot(cur_epochs, graphs.W_M_dist, color = 'c')
                    plt.xlabel('Epoch')
                    plt.ylabel('||W^T - H||^2')
                    plt.title(f'NC3: Self Duality {args_experiment.in_dataset}')
                    result_dict["W_M_dist"] = list(graphs.W_M_dist)

                    if epoch in epoch_list_plot:
                        if args_experiment.use_ood:
                            plt.savefig(
                                f'{save_path_root}/NC3 Self Duality {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_OOD_dateset_{args_experiment.out_dataset}_{args_experiment.model_name}')
                        else:
                            plt.savefig(
                                f'{save_path_root}/NC3 Self Duality {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_{args_experiment.model_name}')
                plt.figure(7)
                plt.plot(cur_epochs, graphs.NCC_mismatch)
                plt.xlabel('Epoch')
                plt.ylabel('Proportion Mismatch from NCC')
                plt.title(f'NC4: Convergence to NCC {args_experiment.in_dataset}')
                result_dict["NCC_mismatch"] = list(graphs.NCC_mismatch)
                if epoch in epoch_list_plot:
                    if args_experiment.use_ood:
                        plt.savefig(
                            f'{save_path_root}/NC4 Convergence to NCC {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_OOD_dateset_{args_experiment.out_dataset}_{args_experiment.model_name}')
                    else:
                        plt.savefig(
                            f'{save_path_root}/NC4 Convergence to NCC {epoch} ,{args_experiment.in_dataset}_{args_experiment.loss_name}_{args_experiment.model_name}')
        path = f'{save_path_root}/output_ID_{args_experiment.in_dataset}.txt'
        if args_experiment.use_ood:
            ood_case = "K_plus_one"
            path = f'{save_path_root}/output_ID_{args_experiment.in_dataset}_OOD_{args_experiment.out_dataset}_case_{ood_case}_{args_experiment.model_name}.txt'

        if args_experiment.add_noise_ID:
            path = f'{save_path_root}/output_ID_{args_experiment.in_dataset}_add_noise_ID_{args_experiment.add_noise_ID}_noise_value_{args_experiment.noise_value}.txt'
    with open(path, "w") as fp:
        json.dump(result_dict, fp, sort_keys = True, indent = 4)
    print("Done!")
