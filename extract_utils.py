import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from tqdm import tqdm

import resnet.resnets as resnet_models
from models.modeling import CONFIGS, VisionTransformer
from utils.data_utils import get_loader, get_loader_resnet
from utils.utils_ood import nested_children

warnings.simplefilter('ignore')
device = 'cuda'


def load_model_and_data(experiment_args, model_train_evaluating = False, eval_step = 0):
    os.makedirs(experiment_args.save_path, exist_ok = True)

    if experiment_args.in_dataset == "cifar10":
        num_classes = 10
    elif experiment_args.in_dataset == "cifar100":
        num_classes = 100
    elif experiment_args.in_dataset == "imagenet":
        num_classes = 1000
    else:
        raise ValueError(
            'we havent trained a resnet-18 model on this dataset yet')
    if experiment_args.model_architecture_type in ('vit', 'swin'):
        experiment_args.dataset = experiment_args.in_dataset
        if experiment_args.in_dataset == 'imagenet':
            model_path = f"{experiment_args.base_path}/{experiment_args.model_name}"
        else:
            model_path = f"{experiment_args.base_path}/{experiment_args.dataset}{experiment_args.model_name}"
        if '384' in experiment_args.model_name:
            experiment_args.img_size = 384
        swin = experiment_args.swin
        print(f"model path {model_path}")
        if 'swin' in experiment_args.model_name:
            swin = True
            experiment_args.swin = True
        print(f" model path {model_path}")
        train_loader, test_loader = load_dataset(
            experiment_args, experiment_args.dataset, swin)
        _, test_loader_OOD = load_dataset(
            experiment_args, experiment_args.out_dataset, swin)
        if 'swin' in experiment_args.model_name:
            if experiment_args.in_dataset == 'imagenet':
                model = tv.models.swin_v2_b()
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                token_extractor = nn.Sequential(*(list(model.children())[:-1]))
                token_extractor.to(device)
                head = model.head
                swin = True
        else:
            model = load_pretrained_model(
                experiment_args, checkpoint = model_path, num_classes = num_classes)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

            model.to(device)
            token_extractor = nn.Sequential(*(list(model.children())[:-1]))
            token_extractor.to(device)
            swin = False
            head = model.get_head()
        if experiment_args.in_dataset == 'imagenet':
            train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train_{experiment_args.model_name}.csv'
            test_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_test_{experiment_args.model_name}.csv'
            test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_{experiment_args.model_name}.csv'
            if model_train_evaluating:
                train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train_{experiment_args.model_name}_step{eval_step}.csv'
                test_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_test_{experiment_args.model_name}_step{eval_step}.csv'
                test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_{experiment_args.model_name}_step{eval_step}.csv'
        else:
            train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train.csv'
            test_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_test.csv'
            test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test.csv'
            if "swin" in experiment_args.model_name:
                train_cls_tocken_path_ID = f'{experiment_args.save_path}/{experiment_args.model_name}_ID_{experiment_args.in_dataset}_train.csv'
                test_cls_tocken_path_ID = f'{experiment_args.save_path}/{experiment_args.model_name}_ID_{experiment_args.in_dataset}_test.csv'
                test_cls_tocken_path_OOD = f'{experiment_args.save_path}/{experiment_args.model_name}_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test.csv'
            if model_train_evaluating:
                train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train_{experiment_args.model_name}_step{eval_step}.csv'
                test_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_test_{experiment_args.model_name}_step{eval_step}.csv'
                test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_{experiment_args.model_name}_step{eval_step}.csv'
        files_ID_already_extracted = os.path.exists(train_cls_tocken_path_ID)
        if files_ID_already_extracted == False:
            train_class_tokens_ID, train_labels_ID, _, _ = valid_head(
                head, token_extractor, train_loader, swin = swin)
            df_train_ID = save_class_token_dataframe(
                train_class_tokens_ID, train_cls_tocken_path_ID, experiment_args, train_labels_ID, cls_size = experiment_args.cls_size)
        files_ID_already_extracted = os.path.exists(test_cls_tocken_path_ID)
        if files_ID_already_extracted == False:
            test_class_tokens_ID, test_labels_ID, test_preds_ID, _ = valid_head(
                head, token_extractor, test_loader, swin = swin)
            df_test_ID = save_class_token_dataframe(
                test_class_tokens_ID, test_cls_tocken_path_ID, experiment_args, test_labels_ID, cls_size = experiment_args.cls_size)
            print(" accuracy current ", np.mean(
                np.array(test_labels_ID) == test_preds_ID) * 100)
        files_OOD_already_extracted = os.path.exists(test_cls_tocken_path_OOD)
        if files_OOD_already_extracted == False:
            test_class_tokens_OOD, test_labels_OOD, _, _ = valid_head(
                head, token_extractor, test_loader_OOD, swin = swin)
            df_test_OOD = save_class_token_dataframe(
                test_class_tokens_OOD, test_cls_tocken_path_OOD, experiment_args, test_labels_OOD, cls_size = experiment_args.cls_size)

        df_test_ID = pandas.read_csv(test_cls_tocken_path_ID, index_col = 0)
        df_train_ID = pandas.read_csv(train_cls_tocken_path_ID, index_col = 0)

        df_test_OOD = pandas.read_csv(test_cls_tocken_path_OOD, index_col = 0)

        csvFile_train, csvFile_test = df_train_ID, df_test_ID
        csvFile_ood = df_test_OOD
        return csvFile_train, csvFile_test, csvFile_ood, model
    if experiment_args.model_architecture_type == "deit":

        if experiment_args.in_dataset == 'imagenet':
            model_path = f"{experiment_args.base_path}/{experiment_args.model_name}"
        else:
            model_path = f"{experiment_args.base_path}/{experiment_args.dataset}{experiment_args.model_name}"
        train_loader, test_loader = load_dataset(
            experiment_args, experiment_args.dataset, swin)
        _, test_loader_OOD = load_dataset(
            experiment_args, experiment_args.out_dataset, swin)
        head = model.get_head()

        if experiment_args.in_dataset == 'imagenet':
            train_cls_tocken_path_ID = f'{experiment_args.save_path}/deit_cls_tocken_ID_{experiment_args.in_dataset}_train_{experiment_args.model_name}.csv'
            test_cls_tocken_path_ID = f'{experiment_args.save_path}/deit_cls_tocken_ID_{experiment_args.in_dataset}_test_{experiment_args.model_name}.csv'
            test_cls_tocken_path_OOD = f'{experiment_args.save_path}/deit_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_{experiment_args.model_name}.csv'

        files_ID_already_extracted = os.path.exists(train_cls_tocken_path_ID)
        if files_ID_already_extracted == False:
            train_class_tokens_ID, train_labels_ID, _, _ = valid_head(
                head, token_extractor, train_loader, swin = swin)
            df_train_ID = save_class_token_dataframe(
                train_class_tokens_ID, train_cls_tocken_path_ID, experiment_args, train_labels_ID, cls_size = experiment_args.cls_size)
        files_ID_already_extracted = os.path.exists(test_cls_tocken_path_ID)
        if files_ID_already_extracted == False:
            test_class_tokens_ID, test_labels_ID, test_preds_ID, _ = valid_head(
                head, token_extractor, test_loader, swin = swin)
            df_test_ID = save_class_token_dataframe(
                test_class_tokens_ID, test_cls_tocken_path_ID, experiment_args, test_labels_ID, cls_size = experiment_args.cls_size)
            print(df_test_ID.shape)
        files_OOD_already_extracted = os.path.exists(test_cls_tocken_path_OOD)
        if files_OOD_already_extracted == False:
            test_class_tokens_OOD, test_labels_OOD, _, _ = valid_head(
                head, token_extractor, test_loader_OOD, swin = swin)
            df_test_OOD = save_class_token_dataframe(
                test_class_tokens_OOD, test_cls_tocken_path_OOD, experiment_args, test_labels_OOD, cls_size = experiment_args.cls_size)
            print(df_test_OOD.shape)

        df_test_ID = pandas.read_csv(test_cls_tocken_path_ID, index_col = 0)
        df_train_ID = pandas.read_csv(train_cls_tocken_path_ID, index_col = 0)
        df_test_OOD = pandas.read_csv(test_cls_tocken_path_OOD, index_col = 0)
        csvFile_train, csvFile_test = df_train_ID, df_test_ID
        csvFile_ood = df_test_OOD
        return csvFile_train, csvFile_test, csvFile_ood, model
    elif experiment_args.model_architecture_type == 'resnet':
        experiment_args.dataset = experiment_args.in_dataset
        resnet_18 = False
        mobile_net = False
        model_path = f"{experiment_args.base_path}/{experiment_args.model_name}_{experiment_args.in_dataset}.pth"
        if 'resnet34' in experiment_args.model_name:
            resnet_18 = True
            model = resnet_models.ResNet34(num_classes)
            experiment_args.cls_size = 512
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict['net'], strict = False)
            print(" acc ", state_dict['acc'])
        else:
            resnet_18 = True
            if num_classes == 1000:
                resnet_18 = False
                model = tv.models.resnet18()
                pth = f"{experiment_args.base_path}/resnet18_imagenet.pth"
                state_dict = torch.load(pth)
                model.load_state_dict(state_dict)
            else:
                model = resnet_models.ResNet18(num_classes)
                resnet_18_checkpoint = model_path
                state_dict = torch.load(resnet_18_checkpoint)
                model.load_state_dict(state_dict['net'], strict = False)
                print(" acc ", state_dict['acc'])
            experiment_args.cls_size = 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        train_loader, test_loader = get_loader_resnet(
            experiment_args, experiment_args.in_dataset, experiment_args.in_dataset)
        print(f" \n \n out dataset {experiment_args.out_dataset}")
        _, test_loader_OOD = get_loader_resnet(
            experiment_args, experiment_args.out_dataset, experiment_args.in_dataset)
        token_extractor = nn.Sequential(*(list(model.children())[:-1]))
        token_extractor.to(device)
        train_cls_tocken_path_ID = f'{experiment_args.save_path}/{experiment_args.model_name}_ID_{experiment_args.in_dataset}_train.csv'
        test_cls_tocken_path_ID = f'{experiment_args.save_path}/{experiment_args.model_name}_ID_{experiment_args.in_dataset}_test.csv'

        test_cls_tocken_path_OOD = f'{experiment_args.save_path}/{experiment_args.model_name}_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test.csv'
        if model_train_evaluating:
            train_cls_tocken_path_ID = f'{experiment_args.save_path}/{experiment_args.model_name}_ID_{experiment_args.in_dataset}_train_step{eval_step}.csv'
            test_cls_tocken_path_ID = f'{experiment_args.save_path}/{experiment_args.model_name}_ID_{experiment_args.in_dataset}_test_step{eval_step}.csv'
            test_cls_tocken_path_OOD = f'{experiment_args.save_path}/{experiment_args.model_name}_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_step{eval_step}.csv'
        if experiment_args.save_preds == True:
            files_ID_already_extracted = os.path.exists(
                train_cls_tocken_path_ID)
            if files_ID_already_extracted == False:
                test_class_tokens_ID, test_labels_ID, test_preds_ID, test_confidence_ID = valid_resnet(
                    model, token_extractor, test_loader, resnet_18 = resnet_18, mobile_net = mobile_net)
                df_test_ID = save_class_token_dataframe(
                    test_class_tokens_ID, test_cls_tocken_path_ID, experiment_args, test_labels_ID, cls_size = experiment_args.cls_size)
                print(" accuracy current ", np.mean(
                    np.array(test_labels_ID) == test_preds_ID) * 100)
                train_class_tokens_ID, train_labels_ID, _, _ = valid_resnet(
                    model, token_extractor, train_loader, resnet_18 = resnet_18, mobile_net = mobile_net)
                df_train_ID = save_class_token_dataframe(
                    train_class_tokens_ID, train_cls_tocken_path_ID, experiment_args, train_labels_ID, cls_size = experiment_args.cls_size)

            files_OOD_already_extracted = os.path.exists(
                test_cls_tocken_path_OOD)
            if files_OOD_already_extracted == False:
                test_class_tokens_OOD, test_labels_OOD, _, _ = valid_resnet(
                    model, token_extractor, test_loader_OOD, resnet_18 = resnet_18, mobile_net = mobile_net)
                df_test_OOD = save_class_token_dataframe(
                    test_class_tokens_OOD, test_cls_tocken_path_OOD, experiment_args, test_labels_OOD, cls_size = experiment_args.cls_size)

        df_test_ID = pandas.read_csv(test_cls_tocken_path_ID, index_col = 0)
        print(df_test_ID.shape)
        df_train_ID = pandas.read_csv(train_cls_tocken_path_ID, index_col = 0)
        print(df_train_ID.shape)

        df_test_OOD = pandas.read_csv(test_cls_tocken_path_OOD, index_col = 0)
        print(df_test_OOD.shape)

        csvFile_train, csvFile_test = df_train_ID, df_test_ID
        csvFile_ood = df_test_OOD
        return csvFile_train, csvFile_test, csvFile_ood, model


def load_model_and_data_train(args, experiment_args, model_train_evaluating = False, eval_step = 0, extract_ood = False, per_class_noise = False):
    print(f" args ewperiment {experiment_args}")
    if experiment_args.model_architecture_type == "vit":
        if experiment_args.in_dataset == 'imagenet':
            model_path = f"{experiment_args.base_path}/{experiment_args.model_name}"
        else:
            model_path = f"{experiment_args.base_path}/{experiment_args.model_name}"
        if model_train_evaluating:
            model_path += f"_step{eval_step}"
        if '384' in experiment_args.model_name:
            args.img_size = 384
        swin = False
        print(f"model path {model_path}")
        if 'swin' in experiment_args.model_name:
            args.img_size = 256
            swin = True
        if experiment_args.in_dataset == "cifar10":
            num_classes = 10
        elif experiment_args.in_dataset == "cifar100":
            num_classes = 100
        elif experiment_args.in_dataset == "imagenet":
            num_classes = 1000
        train_loader, _ = load_dataset(args, args.dataset, swin)
        model = load_pretrained_model(
            args, checkpoint = model_path, num_classes = num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        token_extractor = nn.Sequential(*(list(model.children())[:-1]))
        token_extractor.to(device)
        swin = False
        head = model.get_head()
        if experiment_args.in_dataset == 'imagenet':
            train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train_{experiment_args.model_name}.csv'
            test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_{experiment_args.model_name}.csv'
            if model_train_evaluating:
                train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train_{experiment_args.model_name}_step{eval_step}.csv'
                test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_{experiment_args.model_name}_step{eval_step}.csv'
        else:
            train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train.csv'
            test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test.csv'
            if model_train_evaluating:
                train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train_{experiment_args.model_name}_step{eval_step}.csv'
                test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_{experiment_args.model_name}_step{eval_step}.csv'
            if per_class_noise:
                train_cls_tocken_path_ID = f'{experiment_args.save_path}/ViT_cls_tocken_ID_{experiment_args.in_dataset}_train_{experiment_args.model_name}_step{eval_step}_per_class_noise.csv'
                test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test_{experiment_args.model_name}_step{eval_step}_per_class_noise.csv'
        files_ID_already_extracted = os.path.exists(train_cls_tocken_path_ID)
        if files_ID_already_extracted == False:
            train_class_tokens_ID, train_labels_ID, train_preds_ID, _ = valid_head(
                head, token_extractor, train_loader, swin = swin)
            print(" accuracy current ", np.mean(
                np.array(train_labels_ID) == train_preds_ID) * 100)
            df_train_ID = save_class_token_dataframe(
                train_class_tokens_ID, train_cls_tocken_path_ID, args, train_labels_ID, cls_size = experiment_args.cls_size)
        files_OOD_already_extracted = os.path.exists(test_cls_tocken_path_OOD)
        if files_OOD_already_extracted == False:
            test_cls_tocken_path_OOD = f'{experiment_args.save_path}/ViT_cls_tocken_trained_on_{experiment_args.in_dataset}_OOD_{experiment_args.out_dataset}_test{experiment_args.model_name}_step{eval_step}.csv'
        df_train_ID = pandas.read_csv(train_cls_tocken_path_ID, index_col = 0)
        print(df_train_ID.shape)
        if extract_ood:
            df_ood = pandas.read_csv(test_cls_tocken_path_OOD, index_col = 0)
            print(df_train_ID.shape)
        csvFile_train = df_train_ID
        if extract_ood:
            csvFile_ood = df_ood
            return csvFile_train, csvFile_ood, model
    return csvFile_train, model


def apply_head(head, token):
    logits = head(token)
    score = torch.softmax(logits, dim = 1)
    confidences, preds = torch.max(score, dim = 1)
    preds = torch.argmax(logits, dim = -1)
    return preds, confidences


def load_dataset(args, dataset_name = "cifar10", swin = False):
    temp = args.dataset
    if dataset_name:
        args.dataset = dataset_name
    train_loader, test_loader = get_loader(args)
    args.dataset = temp
    return train_loader, test_loader


def load_pretrained_model(args, checkpoint = "", num_classes = 10):
    config = CONFIGS[args.model_type]
    if checkpoint == "":
        checkpoint = f"{args.base_path}/cifar10-100_500_checkpoint.bin"
    print(f" \n \n {args} \n \n ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(config, args.img_size,
        zero_head = True, num_classes = num_classes)
    save_path_model = checkpoint
    folder_exist = os.path.exists(save_path_model)
    if folder_exist == False:
        checkpoint += '.bin'
        folder_exist = args.in_dataset in checkpoint
        if folder_exist == False:
            l = checkpoint.split('/')
            l[-1] = args.in_dataset + l[-1]
            checkpoint = "/".join(l)
        print(checkpoint)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def valid(model, token_extractor, test_loader):
    all_class_tockens = []

    model.eval()
    all_preds, all_label, all_confidences = [], [], []
    epoch_iterator = tqdm(test_loader,
        desc = "Validating... (loss=X.X)",
        bar_format = "{l_bar}{r_bar}",
        dynamic_ncols = True,
        disable = False)
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]
            batch_cl_tockens, _ = token_extractor(x)
            batch_cl_tocken = batch_cl_tockens[:, 0]
            score = torch.softmax(logits, dim = 1)
            confidences, preds = torch.max(score, dim = 1)
            preds = torch.argmax(logits, dim = -1)
        if len(all_label) == 0:
            all_label.append(y.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_confidences.append(confidences.detach().cpu().numpy())
            all_class_tockens.append(batch_cl_tocken.detach().cpu().numpy())
        else:
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis = 0)
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis = 0)
            all_confidences[0] = np.append(
                all_confidences[0], confidences.detach().cpu().numpy(), axis = 0)
            all_class_tockens[0] = np.append(
                all_class_tockens[0], batch_cl_tocken.detach().cpu().numpy(), axis = 0)
    all_label, all_preds[0], all_class_tockens = all_label[0], all_preds[0], all_class_tockens[0]
    print(f" Accyracy {np.mean(all_label == all_preds)}")
    return all_class_tockens, all_label, all_preds, all_confidences


def valid_head(head, token_extractor, test_loader, swin = False):
    all_class_tockens = []
    all_preds, all_label, all_confidences = [], [], []
    epoch_iterator = tqdm(test_loader,
        desc = "Validating... (loss=X.X)",
        bar_format = "{l_bar}{r_bar}",
        dynamic_ncols = True,
        disable = False)
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            if swin:
                batch_cl_tocken = token_extractor(x)
            else:
                batch_cl_tockens, _ = token_extractor(x)
                batch_cl_tocken = batch_cl_tockens[:, 0]

            logits = head(batch_cl_tocken)
            score = torch.softmax(logits, dim = 1)
            confidences, preds = torch.max(score, dim = 1)
            preds = torch.argmax(logits, dim = -1)
        if len(all_label) == 0:
            all_label.append(y.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_confidences.append(confidences.detach().cpu().numpy())
            all_class_tockens.append(batch_cl_tocken.detach().cpu().numpy())
        else:
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis = 0)
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis = 0)
            all_confidences[0] = np.append(
                all_confidences[0], confidences.detach().cpu().numpy(), axis = 0)
            all_class_tockens[0] = np.append(
                all_class_tockens[0], batch_cl_tocken.detach().cpu().numpy(), axis = 0)
    all_label, all_preds[0], all_class_tockens = all_label[0], all_preds[0], all_class_tockens[0]
    print(f" my accyracy {np.mean(all_label == all_preds)}")
    return all_class_tockens, all_label, all_preds, all_confidences


def valid_resnet(model, token_extractor, test_loader, resnet_18 = True, mobile_net = False):
    all_class_tockens = []

    model.eval()
    all_preds, all_label, all_confidences = [], [], []
    epoch_iterator = tqdm(test_loader,
        desc = "Validating... (loss=X.X)",
        bar_format = "{l_bar}{r_bar}",
        dynamic_ncols = True,
        disable = False)
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            feats = token_extractor(x)
            if mobile_net:
                feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
                feats = torch.flatten(feats, 1)
            if resnet_18:
                feats = F.avg_pool2d(feats, 4)
                feats = feats.view(feats.size(0), -1)
            score = torch.softmax(logits, dim = 1)
            confidences, preds = torch.max(score, dim = 1)
            preds = torch.argmax(logits, dim = -1)

        if len(all_label) == 0:
            all_label.append(y.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_confidences.append(confidences.detach().cpu().numpy())
            all_class_tockens.append(feats.detach().cpu().numpy())
        else:
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis = 0)
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis = 0)
            all_confidences[0] = np.append(
                all_confidences[0], confidences.detach().cpu().numpy(), axis = 0)
            all_class_tockens[0] = np.append(
                all_class_tockens[0], feats.detach().cpu().numpy(), axis = 0)
    all_label, all_preds[0], all_class_tockens = all_label[0], all_preds[0], all_class_tockens[0]
    print(f" my accyracy {np.mean(all_label == all_preds)}")
    return all_class_tockens, all_label, all_preds, all_confidences


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


def remove_classes(pd_dataframe, classes_list = []):
    for i in classes_list:
        pd_dataframe.drop(
            pd_dataframe[pd_dataframe['labels'] == i].index, inplace = True)
    return pd_dataframe


def save_class_token_dataframe(class_tockens, path, args, labels, cls_size = 768):
    x = []
    class_tockens = np.squeeze(class_tockens)
    b_size = args.eval_batch_size
    for i in range(len(class_tockens) // b_size):
        x.extend(np.zeros((b_size, cls_size + 1)))
    if len(class_tockens) % b_size != 0:
        x.extend(np.zeros((len(class_tockens) % b_size, cls_size + 1)))
    x = np.asarray(x)
    class_tockens = np.asarray(class_tockens)
    x = pd.DataFrame(x, columns = [f'ct{i}' for i in range(cls_size)] + ['label'])
    range(x.shape[1] - 1)[-1]
    for i in range(class_tockens.shape[1]):
        x[f'ct{i}'] = class_tockens[:, i]
    x['label'] = labels
    print(f" my path : {path}")
    x.to_csv(path)
    df = pd.read_csv(path, index_col = 0)
    return df
