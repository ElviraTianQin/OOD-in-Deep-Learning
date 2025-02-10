from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
from datetime import timedelta

import numpy as np
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import extract_utils
from models.modeling import CONFIGS, VisionTransformer
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule

logger = logging.getLogger(__name__)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100
    if args.dataset == 'imagenet':
        num_classes = 1000
    print(f" num classes {num_classes}")
    model = VisionTransformer(config, args.img_size,
        zero_head = True, num_classes = num_classes)
    if args.from_scratch == False:
        print(" \n \n \n Training model from pretrained weights \n \n \n ")
        model.load_from(np.load(args.pretrained_dir))
    if args.start_step > 0:
        print(" \n \n \n resuming old training \n \n \n ")
        model = extract_utils.load_pretrained_model(args, checkpoint = os.path.join(
            args.resume_dir, f"{args.in_dataset}{args.name}_step{args.start_step}.bin"), num_classes = num_classes)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, step = 100):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    if args.test_NC:
        model_checkpoint = f"{args.output_dir}/{args.in_dataset}{args.name}_step{step}.bin"
        torch.save(model_to_save.state_dict(), model_checkpoint)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader, global_step):
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
        desc = "Validating... (loss=X.X)",
        bar_format = "{l_bar}{r_bar}",
        dynamic_ncols = True,
        disable = args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim = -1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis = 0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis = 0
            )
        epoch_iterator.set_description(
            "Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    return accuracy


def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok = True)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args, shuffle_train = args.SHUFFLE)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
        lr = args.learning_rate,
        momentum = 0.9,
        weight_decay = args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps = args.warmup_steps, t_total = t_total)
    else:
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps = args.warmup_steps, t_total = t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models = model,
            optimizers = optimizer,
            opt_level = args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size = 250000000,
            gradient_predivide_factor = get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d",
        args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps * (
            torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
        args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    os.makedirs('test_results', exist_ok = True)

    while True:

        model.train()
        epoch_iterator = tqdm(train_loader,
            desc = "Training (X / X Steps) (loss=X.X)",
            bar_format = "{l_bar}{r_bar}",
            dynamic_ncols = True,
            disable = args.local_rank not in [-1, 0])
        # print( f"\n\n  global_sterp {global_step}  \n \n  total step { t_total}")
        # print( f" spoch sim { len(train_loader)}")
        save_path_figs = f"./test_results/{args.name}"
        folder_exist = os.path.exists(save_path_figs)
        if folder_exist is False:
            os.mkdir(save_path_figs)
        for step, batch in enumerate(epoch_iterator):
            # print(" \n \n \n did we enter ? \n \n \n ")
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step,
                                                               t_total, losses.val)
                )
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, test_loader, global_step)
                    if args.test_NC:
                        save_model(args, model, step = global_step +
                                                       args.start_step)
                    if best_acc < accuracy:
                        save_model(args, model, step = global_step +
                                                       args.start_step)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break
        logger.info("Best Accuracy: \t%f" % best_acc)
        logger.info("End Training!")


def load_experiment_args():
    parser_experiment = argparse.ArgumentParser()

    args_experiment, _ = parser_experiment.parse_known_args()
    return args_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required = True,
        help = "Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices = ["cifar10", "cifar100", "tiny_imagenet", "imagenet", 'texture', 'inaturalist'], default = "imagenet",
        help = "Which downstream task.")
    parser.add_argument("--model_type", choices = ["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_16-224",
                                                   "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
        default = "ViT-B_16",
        help = "Which variant to use.")
    parser.add_argument("--pretrained_dir", type = str, default = "checkpoint/ViT-B_16.npz",
        help = "Where to search for pretrained ViT models.")
    parser.add_argument("--checkpoint_dir", type = str, default = "output",
        help = "Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default = "output", type = str,
        help = "The output directory where checkpoint was saved.")
    parser.add_argument("--data_path", default = "./data",
        help = "directory where the datatsets are saved.")

    parser.add_argument("--img_size", default = 224, type = int,
        help = "Resolution size")
    parser.add_argument("--cls_size", default = 768, type = int,
        help = "class token  size")
    parser.add_argument("--resume", default = 0, type = int,
        help = "Reume training from chevkpoint")
    parser.add_argument("--train_batch_size", default = 512, type = int,
        help = "Total batch size for training.")
    parser.add_argument("--eval_batch_size", default = 64, type = int,
        help = "Total batch size for eval.")
    parser.add_argument("--eval_every", default = 100, type = int,
        help = "Run prediction on validation set every so many steps."
               "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default = 3e-2, type = float,
        help = "The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default = 0, type = float,
        help = "Weight deay if we apply some.")
    parser.add_argument("--num_steps", default = 1000, type = int,
        help = "Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices = ["cosine", "linear"], default = "cosine",
        help = "How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default = 500, type = int,
        help = "Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default = 1.0, type = float,
        help = "Max gradient norm.")

    parser.add_argument("--local_rank", type = int, default = -1,
        help = "local_rank for distributed training on gpus")
    parser.add_argument('--seed', type = int, default = 42,
        help = "random seed for initialization")

    parser.add_argument('--start_step', type = int, default = 0,
        help = "random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps', type = int, default = 4,
        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action = 'store_true',
        help = "Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type = str, default = 'O2',
        help = "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
               "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--resume_dir', type = str,
        default = '/data/users/mben-ammar/Workspace/ViT-pytorch-main/output', )

    parser.add_argument('--loss_scale', type = float, default = 0,
        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
               "0 (default value): dynamic loss scaling.\n"
               "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--SHUFFLE", type = bool, default = False,
        help = "shuffle train data while training  ")
    parser.add_argument("--test_NC", type = bool, default = False,
        help = "test neural collapse metrics every eval step  ")

    parser.add_argument("--from_scratch", type = bool,
        default = False, help = "start training from scratch   ")

    ################################################ NC args ###################################
    parser.add_argument("--model_architecture_type", choices = ["vit", "resnet"],
        default = "vit",
        help = "what type of model to use")

    parser.add_argument("--model_name", default = "_PNAC_vit_B",
        help = "Which model to use.")
    parser.add_argument("--swin", type = bool, default = False,
        help = "")

    parser.add_argument("--in_dataset", choices = ["cifar10", "cifar100", 'imagenet'], default = "imagenet",
        help = "Which downstream task is ID.")
    parser.add_argument("--out_dataset", choices = ["cifar10", "SUN", "places", "cifar100", "SVHN", 'tiny_imagenet', 'imagenet-o', 'imagenet-a', 'imagenet', 'texture', 'inaturalist',
                                                    'open-images'], default = "imagenet-o",
        help = "Which downstream task is OOD.")

    parser.add_argument("--save_preds", type = bool, default = False,
        help = "if set to True, recompute the models prediction and save them, else use saved predictions")

    args = parser.parse_args()
    args.model_name = args.name
    args.in_dataset = args.dataset

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend = 'nccl',
            timeout = timedelta(minutes = 60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)
    print(f" current seed {args.seed}")

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    print(model)
    print(" \n \n ")
    print(args)
    train(args, model)


if __name__ == "__main__":
    main()
