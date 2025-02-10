import logging
import os
import os.path

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def get_loader_resnet(args, dataset_name = 'cifar10', id_data = 'cifar100'):
    num_workers = 0
    transform_train_50 = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [
            0.229, 0.224, 0.225]),
    ])
    transform_test_50 = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [
            0.229, 0.224, 0.225]),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ])
    print('==> Preparing data..')
    if dataset_name == 'cifar10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root = f"{args.data_path}/cifar10", train = True, download = True, transform = transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)

        testset = torchvision.datasets.CIFAR10(
            root = f"{args.data_path}/cifar10", train = False, download = True, transform = transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size = 100, shuffle = False, num_workers = 2)
        return trainloader, testloader
    elif dataset_name == 'SVHN':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN(
            root = f"{args.data_path}/SVHN", split = 'train', download = True, transform = transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = num_workers)
        testset = torchvision.datasets.SVHN(
            root = f"{args.data_path}/SVHN", split = 'test', download = True, transform = transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size = 100, shuffle = False, num_workers = num_workers)
        return trainloader, testloader
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        trainset = torchvision.datasets.CIFAR100(
            root = f"{args.data_path}/cifar100", train = True, download = True, transform = transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size = 128, shuffle = True, num_workers = 2)

        testset = torchvision.datasets.CIFAR100(
            root = f"{args.data_path}/cifar100", train = False, download = True, transform = transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size = 100, shuffle = False, num_workers = 2)
        return trainloader, testloader
    elif dataset_name == 'inaturalist':
        data_path = f'{args.data_path}/inaturalist/iNaturalist'
        trainset = None

        transform = transform_test_50
        if id_data != 'imagenet':
            transform = transform_test

        full_dataset = torchvision.datasets.ImageFolder(
            root = data_path,
            transform = transform
        )
        train_size = int(0.0 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        generator = torch.Generator()
        generator.manual_seed(0)

        trainset, testset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator = generator)
    elif dataset_name == 'texture':
        data_path = f'{args.data_path}/texture/dtd/images'
        img_list = "dataset_samples/texture.txt"
        transform = transform_test_50
        if id_data != 'imagenet':
            transform = transform_test
        full_dataset = ImageFilelist(data_path, img_list, transform = transform)
        train_size = int(0.0 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        print(f" my dataset length {len(full_dataset)}")

        # use torch.utils.data.random_split for training/test split
        generator = torch.Generator()
        generator.manual_seed(0)

        trainset, testset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator = generator)

    elif dataset_name == 'imagenet':
        testset = torchvision.datasets.ImageNet(
            f'{args.data_path}/IMAGENET/val', 'val', transform = transform_test_50)
        trainset = torchvision.datasets.ImageNet(
            f'{args.data_path}/IMAGENET/train', 'train', transform = transform_train_50)

    elif dataset_name == 'imagenet-o':
        imagenet_o_folder = f'{args.data_path}/imagenet-o'
        transform = transform_test_50
        if id_data != 'imagenet':
            transform = transform_test

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_o_folder, transform = transform)

        trainset, testset = None, val_examples_imagenet_o
    elif dataset_name == 'places':
        imagenet_o_folder = f'{args.data_path}/places/images'
        transform = transform_test_50
        if id_data != 'imagenet':
            transform = transform_test

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_o_folder, transform = transform)

        trainset, testset = None, val_examples_imagenet_o
    elif dataset_name == 'SUN':
        imagenet_o_folder = f'{args.data_path}/SUN'
        transform = transform_test_50
        if id_data != 'imagenet':
            transform = transform_test
        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_o_folder, transform = transform)
        trainset, testset = None, val_examples_imagenet_o

    elif dataset_name == 'imagenet-a':
        imagenet_a_folder = f'{args.data_path}/imagenet-a'

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_a_folder, transform = transform_test_50)
        trainset, testset = None, val_examples_imagenet_o
    elif dataset_name == 'open-images':
        data_path = f'{args.data_path}/open-images/human_annotated/2017_11/train'
        img_list = "dataset_samples/open_images.txt"
        full_dataset = ImageFilelist(
            data_path, img_list, transform = transform_test_50)
        train_size = int(0.0 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        print(f" my dataset length {len(full_dataset)}")
        generator = torch.Generator()
        generator.manual_seed(0)

        trainset, testset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator = generator)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size = 512, shuffle = False, num_workers = num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size = 512, shuffle = False, num_workers = num_workers)
    return trainloader, testloader


def get_loader(args, shuffle_train = False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            (args.img_size, args.img_size), scale = (0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        print(args.data_path)
        trainset = datasets.CIFAR10(root = f"{args.data_path}/cifar10",
            train = True,
            download = True,
            transform = transform_train)
        testset = datasets.CIFAR10(root = f"{args.data_path}/cifar10",
            train = False,
            download = True,
            transform = transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "SVHN":
        trainset = datasets.SVHN(root = f"{args.data_path}/SVHN",
            split = 'train',
            download = True,
            transform = transform_train)
        testset = datasets.SVHN(root = f"{args.data_path}/SVHN",
            split = 'test',
            download = True,
            transform = transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root = f"{args.data_path}/cifar100",
            train = True,
            download = True,
            transform = transform_train)
        testset = datasets.CIFAR100(root = f"{args.data_path}/cifar100",
            train = False,
            download = True,
            transform = transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == 'inaturalist':
        data_path = f'{args.data_path}/inaturalist/iNaturalist'
        trainset = None
        full_dataset = torchvision.datasets.ImageFolder(
            root = data_path,
            transform = transform_test)
        train_size = int(0.0 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        generator = torch.Generator()
        generator.manual_seed(0)

        trainset, testset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator = generator)
    elif args.dataset == 'SUN':
        data_path = f'{args.data_path}/SUN/images'
        trainset = None
        full_dataset = torchvision.datasets.ImageFolder(
            root = data_path,
            transform = transform_test
        )
        train_size = int(0.0 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # use torch.utils.data.random_split for training/test split
        generator = torch.Generator()
        generator.manual_seed(0)

        trainset, testset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator = generator)
    elif args.dataset == 'places':
        data_path = f'{args.data_path}/places/images'
        trainset = None
        full_dataset = torchvision.datasets.ImageFolder(
            root = data_path,
            transform = transform_test
        )
        train_size = int(0.0 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        generator = torch.Generator()
        generator.manual_seed(0)

        trainset, testset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator = generator)

    elif args.dataset == 'texture':
        data_path = f'{args.data_path}/texture/dtd/images'
        img_list = "/data/users/mben-ammar/Workspace/ViT-pytorch-main/texture.txt"
        full_dataset = ImageFilelist(
            data_path, img_list, transform = transform_test)
        train_size = int(0.0 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        print(f" my dataset length {len(full_dataset)}")
        generator = torch.Generator()
        generator.manual_seed(0)

        trainset, testset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator = generator)
    elif args.dataset == 'tiny_imagenet':
        data_path = f'{args.data_path}/tiny-imagenet-200'

        full_dataset = torchvision.datasets.ImageFolder(
            root = data_path,
            transform = transform_test
        )
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        generator = torch.Generator()
        generator.manual_seed(0)

        trainset, testset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator = generator)
    elif args.dataset == 'imagenet':
        testset = torchvision.datasets.ImageNet(
            f'{args.data_path}/IMAGENET/val', 'val', transform = transform_test)
        trainset = torchvision.datasets.ImageNet(
            f'{args.data_path}/IMAGENET/train', 'train', transform = transform_train)

    elif args.dataset == 'imagenet-o':
        imagenet_o_folder = f'{args.data_path}/imagenet-o'
        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_o_folder, transform = transform_test)
        trainset, testset = None, val_examples_imagenet_o

    elif args.dataset == 'imagenet-a':
        imagenet_a_folder = f'{args.data_path}/imagenet-a'

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_a_folder, transform = transform_test)
        trainset, testset = None, val_examples_imagenet_o
    elif args.dataset == 'open-images':
        imagenet_a_folder = f'{args.data_path}/images_largescale/openimage_o'

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_a_folder, transform = transform_test)
        trainset, testset = None, val_examples_imagenet_o
    elif args.dataset == 'imagenet-r':
        imagenet_a_folder = f'{args.data_path}/images_largescale/imagenet_r'
        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_a_folder, transform = transform_test)
        trainset, testset = None, val_examples_imagenet_o
    elif args.dataset == 'imagenet-c':
        imagenet_a_folder = f'{args.data_path}/images_largescale/imagenet_c'

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_a_folder, transform = transform_test)

        trainset, testset = None, val_examples_imagenet_o
    elif args.dataset == 'imagenet_v2':
        imagenet_a_folder = f'{args.data_path}/images_largescale/imagenet_v2'

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_a_folder, transform = transform_test)

        trainset, testset = None, val_examples_imagenet_o

    elif args.dataset == 'ninco':
        imagenet_a_folder = f'{args.data_path}/images_largescale/ninco'

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_a_folder, transform = transform_test)

        trainset, testset = None, val_examples_imagenet_o
    elif args.dataset == 'ssb_hard':
        imagenet_a_folder = f'{args.data_path}/images_largescale/ssb_hard'

        val_examples_imagenet_o = datasets.ImageFolder(
            root = imagenet_a_folder, transform = transform_test)
        trainset, testset = None, val_examples_imagenet_o
    if args.local_rank == 0:
        torch.distributed.barrier()

    if shuffle_train == True:
        train_sampler = RandomSampler(trainset) if (args.local_rank == -1) \
            else DistributedSampler(trainset)
        train_loader = DataLoader(trainset, sampler = train_sampler,
            batch_size = args.train_batch_size,
            num_workers = 4, pin_memory = True)
    else:
        shuffle = False
        train_loader = DataLoader(trainset, shuffle = shuffle,
            batch_size = args.train_batch_size,
            num_workers = 4, pin_memory = True)
    test_loader = DataLoader(testset, shuffle = False,
        batch_size = args.eval_batch_size,
        num_workers = 4,
        pin_memory = True) if testset is not None else None

    return train_loader, test_loader


def create_symlinks_to_imagenet(imagenet_folder, folder_to_scan, imagenet_o_folder):
    if not os.path.exists(imagenet_folder):
        os.makedirs(imagenet_folder)
        folders_of_interest = os.listdir(folder_to_scan)
        path_prefix = imagenet_o_folder
        for folder in folders_of_interest:
            os.symlink(path_prefix + folder, imagenet_folder +
                       folder, target_is_directory = True)


#################################


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n
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
