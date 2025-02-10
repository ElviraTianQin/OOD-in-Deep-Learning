import argparse
import os

import extract_utils

if __name__ == '__main__':
    parser_experiment = argparse.ArgumentParser()
    parser_experiment.add_argument("--model_architecture_type", choices = ["vit", "resnet", 'deit', 'swin'],
        default = "vit",
        help = "what type of model to use")

    parser_experiment.add_argument("--model_name", default = "-100_500_checkpoint.bin",
        help = "Which model to use.")
    parser_experiment.add_argument("--base_path", default = "./",
        help = "directory where the model is saved.")
    parser_experiment.add_argument("--save_path", default = "./",
        help = "directory where the features will be saved.")
    parser_experiment.add_argument("--data_path", default = "./",
        help = "directory where the datatsets are saved.")
    parser_experiment.add_argument("--in_dataset", choices = ["cifar10", "cifar100", 'imagenet'], default = "cifar10",
        help = "Which downstream task is ID.")
    parser_experiment.add_argument("--out_dataset", choices = ["cifar10", "SUN", "places", "cifar100", "SVHN", 'tiny_imagenet', 'imagenet_v2', 'imagenet-o', 'imagenet-a', 'imagenet-r', 'imagenet-c',
                                                               'imagenet', 'ninco', 'ssb_hard', 'texture', 'inaturalist', 'open-images'], default = "cifar100",
        help = "Which downstream task is OOD.")
    parser_experiment.add_argument("--cls_size", type = int, default = 768,
        help = "size of the class token to be used ")
    parser_experiment.add_argument("--save_preds", type = bool, default = False,
        help = "if set to True, recompute the models prediction and save them, else use saved predictions")
    parser_experiment.add_argument("--model_type", choices = ["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                              "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"], default = "ViT-B_16",
        help = "Which variant to use.")

    parser_experiment.add_argument("--swin", type = bool, default = False,
        help = "")
    parser_experiment.add_argument("--img_size", default = 224, type = int,
        help = "Resolution size")
    parser_experiment.add_argument("--local_rank", type = int, default = -1,
        help = "local_rank for distributed training on gpus")
    parser_experiment.add_argument('--seed', type = int, default = 42,
        help = "random seed for initialization")
    parser_experiment.add_argument("--dataset", choices = ["cifar10", "cifar100", "SVHN", "imagenet"], default = "cifar10",
        help = "Which downstream task.")
    parser_experiment.add_argument("--train_batch_size", default = 128, type = int,
        help = "Total batch size for training.")
    parser_experiment.add_argument("--eval_batch_size", default = 64, type = int,
        help = "Total batch size for eval.")
    args_experiment, unknown = parser_experiment.parse_known_args()
    csvFile_train, csvFile_test, csvFile_ood, model = extract_utils.load_model_and_data(args_experiment)

    print('Done!')
