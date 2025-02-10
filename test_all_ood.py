import argparse

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from numpy.linalg import pinv
from scipy.special import softmax

import extract_utils
import ood_methods
import resnet.resnets as resnet_models


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')

    parser.add_argument('--clip_quantile', default=0.99,
                        help='Clip quantile to react')

    parser.add_argument('--img_list', default=None, help='Path to image list')
    parser.add_argument("--in_dataset", choices=["cifar10", "cifar100", 'imagenet'], default="imagenet",
                        help="Which downstream task is ID.")
    parser.add_argument("--out_dataset", choices=["cifar10", "cifar100", "SVHN", "SUN", "places", 'tiny_imagenet', 'imagenet-o', 'imagenet-a', 'imagenet', 'texture', 'inaturalist', 'open-images'], default="imagenet-o",
                        help="Which downstream task is OOD.")
    parser.add_argument("--cls_size", type=int, default=768,
                        help="size of the class token to be used ")
    parser.add_argument("--model_name",
                        default="-100_500_checkpoint.bin",
                        help="Which model to use.")
    parser.add_argument("--model_architecture_type", choices=["vit", "deit", "resnet", 'swin'],
                        default="vit",
                        help="what type of model to use")
    parser.add_argument("--base_path", default="./",
                        help="directory where the model is saved.")
    parser.add_argument("--save_path", default="./",
                        help="directory where the features will be saved.")
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--neco_dim', default=100,
                        help='ETF approximative dimention')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--n_components_null_space", type=int, default=2,
                        help="Number of PCA components to be used for the null space norm")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.in_dataset == "cifar10":
        num_classes = 10
    elif args.in_dataset == "cifar100":
        num_classes = 100
    elif args.in_dataset == "imagenet":
        num_classes = 1000
    if args.model_architecture_type in ("vit", 'swin', 'deit'):
        if args.in_dataset == 'imagenet':
            train_cls_tocken_path_ID = f'{args.save_path}/ViT_cls_tocken_ID_{args.in_dataset}_train_{args.model_name}.csv'
            test_cls_tocken_path_ID = f'{args.save_path}/ViT_cls_tocken_ID_{args.in_dataset}_test_{args.model_name}.csv'
            test_cls_tocken_path_OOD = f'{args.save_path}/ViT_cls_tocken_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test_{args.model_name}.csv'
        else:
            train_cls_tocken_path_ID = f'{args.save_path}/ViT_cls_tocken_ID_{args.in_dataset}_train.csv'
            test_cls_tocken_path_ID = f'{args.save_path}/ViT_cls_tocken_ID_{args.in_dataset}_test.csv'
            test_cls_tocken_path_OOD = f'{args.save_path}/ViT_cls_tocken_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'
        if args.in_dataset == 'imagenet':

            model_path = f"{args.base_path}/{args.model_name}"
        else:
            model_path = f"{args.base_path}/{args.in_dataset}{args.model_name}"
        print(f" args : {args}")
        args.ood_features = test_cls_tocken_path_OOD  # TODO
        ood_name = args.out_dataset
        args_vit = args

        if '384' in model_path:
            args_vit.img_size = 384
        print(f" args_vit{ args_vit}")
        if "swin" in args.model_name:
            model = tv.models.swin_v2_b()
        else:
            model = extract_utils.load_pretrained_model(
                args_vit, checkpoint=model_path, num_classes=num_classes)

        model_layers = extract_utils.nested_children(model)
        last_layer = model_layers['head']
        bias = last_layer.bias
        bias.requires_grad = False
        bias = bias.detach().cpu().numpy()
        weight = last_layer.weight
        weight.requires_grad = False
        weight = weight.detach().cpu().numpy()
        w, b = weight, bias
        print(f'{w.shape=}, {b.shape=}')
    elif args.model_architecture_type == "resnet":
        train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_train.csv'
        test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_test.csv'
        test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'

        print(f" my args : {args}")
        args.ood_features = test_cls_tocken_path_OOD
        ood_name = args.out_dataset
        print(f"ood datasets: {ood_name}")
        model_path = f"{args.base_path}/{args.model_name}_{args.in_dataset}.pth"

        if args.model_name == 'resnet50':
            model_path = f"{args.base_path}/resnet50_{args.in_dataset}.pth"
            args.cls_size = 2048

            model = tv.models.resnet50()
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            model_layers = extract_utils.nested_children(model)
            last_layer = model_layers['fc']
            train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_train.csv'
            test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_test.csv'
            test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'

        elif args.model_name == "resnet34":
            model = resnet_models.ResNet34(num_classes)
            resnet_18_checkpoint = model_path
            state_dict = torch.load(resnet_18_checkpoint)
            model.load_state_dict(state_dict['net'], strict=False)
            print(" acc ", state_dict['acc'])
            model_layers = extract_utils.nested_children(model)
            last_layer = model_layers['linear']
            args.cls_size = 512

            train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_train.csv'
            test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_test.csv'
            test_preds_path_ID = f'{args.save_path}/{args.model_name}_preds_ID_{args.in_dataset}_test.csv'
            test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'
            test_preds_path_OOD = f'{args.save_path}/{args.model_name}_preds_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'
        elif args.model_name == "resnet18":
            model = resnet_models.ResNet18(num_classes)
            resnet_18_checkpoint = model_path
            print(f" model path {model_path}")
            state_dict = torch.load(resnet_18_checkpoint)
            model.load_state_dict(state_dict['net'], strict=False)
            print(" acc ", state_dict['acc'])
            args.cls_size = 512
            model_layers = extract_utils.nested_children(model)
            last_layer = model_layers['linear']
            train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_train.csv'
            test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_test.csv'
            test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'
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

 ################################################################################################################################################################################################################################################################
    print('load features')
    df_train_ID = pd.read_csv(train_cls_tocken_path_ID, index_col=0)
    train_labels = df_train_ID['label'].values.ravel()
    df_test_ID = pd.read_csv(test_cls_tocken_path_ID, index_col=0)
    df_test_OOD = pd.read_csv(test_cls_tocken_path_OOD, index_col=0)
    features = [f'ct{i}' for i in range(args.cls_size)]
    feature_id_train = df_train_ID.loc[:, features].values
    feature_id_val = df_test_ID.loc[:, features].values
    feature_ood = df_test_OOD.loc[:, features].values

    print(f'{feature_id_train.shape=}, {feature_id_val.shape=}, {feature_ood.shape=}')
    print(f" my args {args}")
    print('computing logits')
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_ood = feature_ood @ w.T + b
    print('computing softmax')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_ood = softmax(logit_ood, axis=-1)
    u = -np.matmul(pinv(w), b)
    # ---------------------------------------
    method = 'MSP'
    print(f'\n{method}')
    ood_methods.msp(softmax_id_val, softmax_ood, ood_name)
    # ---------------------------------------
    method = 'MaxLogit'
    print(f'\n{method}')
    ood_methods.maxLogit(logit_id_val, logit_ood, ood_name)
    # ---------------------------------------
    method = 'Energy'
    print(f'\n{method}')
    ood_methods.energy(logit_id_val, logit_ood, ood_name)
    # ---------------------------------------
    method = 'Energy+React'
    print(f'\n{method}')
    result = []
    thresh = 0.99
    if args.model_architecture_type == "vit":
        thresh = 0.99
    if args.model_architecture_type == "resnet":
        thresh = 0.9
    if args.model_architecture_type == "swin":
        thresh = 0.95
    clip = np.quantile(feature_id_train, thresh)
    ood_methods.react(feature_id_val, feature_ood, clip, w, b, ood_name)
    # ---------------------------------------
    method = 'ViM'
    print(f'\n{method}')
    ood_methods.vim(feature_id_train, feature_id_val, feature_ood, logit_id_train,
                    logit_id_val, logit_ood, ood_name, args.model_architecture_type, args.model_name, u)

    # ---------------------------------------
    method = 'NECO'
    print(f'\n{method}')
    ood_methods.neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood,
                     model_architecture_type=args.model_architecture_type, neco_dim=args.neco_dim)

    # ---------------------------------------
    method = 'Residual'
    print(f'\n{method}')
    ood_methods.residual(feature_id_train, feature_id_val,
                         feature_ood, args.model_architecture_type, u, ood_name)
    # ---------------------------------------
    method = 'GradNorm'
    print(f'\n{method}')
    result = []
    ood_methods.gradNorm(feature_id_val, feature_ood,
                         ood_name, num_classes, w, b)
    # ---------------------------------------
    method = 'Mahalanobis'
    print(f'\n{method}')
    ood_methods.mahalanobis(feature_id_train, train_labels,
                            feature_id_val, feature_ood, ood_name, num_classes)
    # ---------------------------------------
    method = 'KL-Matching'
    print(f'\n{method}')
    ood_methods.kl_matching(softmax_id_train, softmax_id_val,
                            softmax_ood, ood_name, num_classes)


if __name__ == '__main__':
    main()
