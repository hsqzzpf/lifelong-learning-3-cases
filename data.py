import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch
from Datasets.increm_animals import ClassSplit, FlexAnimalSet
from os.path import join
import random

def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image


def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name=='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


#----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


#----------------------------------------------------------------------------------------------------------#


# specify available data-sets.
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'CIFAR10': datasets.CIFAR10,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'CIFAR10': [
        transforms.ToTensor(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
    'CIFAR100-animal': {'size': 32, 'channels': 3, 'classes': 45},
    'ImageNet': {'size': 224, 'channels': 3, 'classes': 40}
}


#----------------------------------------------------------------------------------------------------------#


def get_multitask_experiment(name, scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                             exception=False, random_seed=0):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # prepare datasets
            train_datasets = []
            test_datasets = []
            for task_id, p in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class') else None
                train_datasets.append(get_dataset('mnist', type="train", permutation=p, dir=data_dir,
                                                  target_transform=target_transform, verbose=verbose))
                test_datasets.append(get_dataset('mnist', type="test", permutation=p, dir=data_dir,
                                                 target_transform=target_transform, verbose=verbose))
    elif name == 'splitMNIST':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'CIFAR10':

        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'CIFAR10' cannot have more than 10 tasks!")

        # configurations
        config = DATASET_CONFIGS['CIFAR10']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:

            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))

            # prepare train and test datasets with all classes
            mnist_train = get_dataset('CIFAR10', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            mnist_test = get_dataset('CIFAR10', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)

            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario == 'domain' else None
                train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'CIFAR100-animal':
        config = DATASET_CONFIGS['CIFAR100-animal']
        classes_per_task = int(np.floor(45 / tasks))
        if not only_config:

            if 50 <= random_seed < 100:
                a0 = [0, 2, 3, 1, 6]
                a1 = [12, 11, 17, 18, 16]
                a2 = [14, 4, 20, 9, 13]
                a3 = [23, 26, 28, 30, 25]
                a4 = [5, 21, 8, 22, 35]
                a5 = [27, 24, 32, 42, 7]
                a6 = [39, 15, 36, 44, 29]
                a7 = [38, 40, 31, 10, 43]
                a8 = [19, 41, 37, 34, 33]

                combine = list()
                combine.append(a0)
                combine.append(a1)
                combine.append(a2)
                combine.append(a3)
                combine.append(a4)
                combine.append(a5)
                combine.append(a6)
                combine.append(a7)
                combine.append(a8)

                random.seed(random_seed-50)
                sequence = [x for x in range(9)]
                random.shuffle(sequence)

                sort = list()
                for idx in sequence:
                    sort.extend(combine[idx])
                class_split = ClassSplit(45, [], random_sort=sort)

            elif 100 <= random_seed < 200:
                a0 = [2,  14,  25, 42, 31]
                a1 = [0,  16,  30, 32, 40]
                a2 = [3,   4,   5,  7, 10]
                a3 = [1,  20,  21, 39, 43]
                a4 = [6,   9,   8, 15, 19]
                a5 = [12, 13,  22, 36, 41]
                a6 = [11, 23,  35, 44, 37]
                a7 = [18, 28,  24, 38, 33]
                a8 = [17, 26,  27, 29, 34]

                combine = list()
                combine.append(a0)
                combine.append(a1)
                combine.append(a2)
                combine.append(a3)
                combine.append(a4)
                combine.append(a5)
                combine.append(a6)
                combine.append(a7)
                combine.append(a8)

                random.seed(random_seed - 100)
                sequence = [x for x in range(9)]
                random.shuffle(sequence)

                sort = list()
                for idx in sequence:
                    sort.extend(combine[idx])
                class_split = ClassSplit(45, [], random_sort=sort)
                # class_split = ClassSplit(45, [], random_sort=[2, 14, 25, 42, 31,
                #                                             0,  16,  30, 32, 40,
                #                                             3,   4,   5,  7, 10,
                #                                             1,  20,  21, 39, 43,
                #                                             6,   9,   8, 15, 19,
                #                                             12, 13,  22, 36, 41,
                #                                             11, 23,  35, 44, 37,
                #                                             18, 28,  24, 38, 33,
                #                                             17, 26,  27, 29, 34])
            elif 200 <= random_seed < 300:
                a0 = [0, 16, 21, 26, 30, 31, 42]
                a1 = [1, 8, 11, 15]
                a2 = [2, 6, 9, 10, 14, 18, 19, 23, 25, 29, 33, 34, 37, 38, 40, 41, 43]
                a3 = [3, 4, 5, 12, 17, 20, 22, 24, 32, 35, 39]
                a4 = [7, 13, 27, 28, 36, 44]

                combine = list()
                combine.append(a0)
                combine.append(a1)
                combine.append(a2)
                combine.append(a3)
                combine.append(a4)

                random.seed(random_seed - 200)
                sequence = [x for x in range(5)]
                random.shuffle(sequence)

                if random_seed < 250:
                    sort = list()
                    for idx in sequence:
                        sort.extend(combine[idx])
                    class_split = ClassSplit(45, [], random_sort=sort)
                else:
                    sort = list()
                    count = 0
                    horizon = 0
                    while count < 45:
                        for idx in sequence:
                            if len(combine[idx]) <= horizon:
                                continue
                            else:
                                sort.append(combine[idx][horizon])
                                count += 1
                        horizon += 1
                    class_split = ClassSplit(45, [], random_sort=sort)
            elif 300 <= random_seed < 400:
                a0 = [0, 12, 14, 16, 30, 32, 40, 41, 42]
                a1 = [1, 2, 9, 18, 24, 25, 26, 27, 28, 29, 31, 33, 38]
                a2 = [3, 4, 5, 7, 10, 11, 22, 23, 34, 35, 36, 37, 44]
                a3 = [6, 8, 13, 15, 19]
                a4 = [17, 20, 21, 39, 43]

                combine = list()
                combine.append(a0)
                combine.append(a1)
                combine.append(a2)
                combine.append(a3)
                combine.append(a4)

                random.seed(random_seed - 300)
                sequence = [x for x in range(5)]
                random.shuffle(sequence)

                if random_seed < 350:
                    sort = list()
                    for idx in sequence:
                        sort.extend(combine[idx])
                    class_split = ClassSplit(45, [], random_sort=sort)
                else:
                    sort = list()
                    count = 0
                    horizon = 0
                    while count < 45:
                        for idx in sequence:
                            if len(combine[idx]) <= horizon:
                                continue
                            else:
                                sort.append(combine[idx][horizon])
                                count += 1
                        horizon += 1
                    print(sort)
                    class_split = ClassSplit(45, [], random_sort=sort)

            else:
                class_split = ClassSplit(45, [], random_seed=random_seed)

            trainset = FlexAnimalSet(join('datasets', 'CIFAR100-animal'), True, class_split,
                                     [x for x in range(45)], None)
            testset = FlexAnimalSet(join('datasets', 'CIFAR100-animal'), False, class_split,
                                    [x for x in range(45)], None)

            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # labels_per_task = [[x for x in range(12)], [x for x in range(12, 30)], [x for x in range(30, 45)]]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario == 'domain' else None
                train_datasets.append(SubDataset(trainset, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(testset, labels, target_transform=target_transform))

    elif name == 'ImageNet':
        config = DATASET_CONFIGS['ImageNet']
        classes_per_task = int(np.floor(40 / tasks))
        if not only_config:
            if random_seed == 100:
                class_split = ClassSplit(40, [], random_sort=[x for x in range(40)])
            elif random_seed == 101:
                a1 = [x for x in range(20)]
                a2 = [x for x in range(20, 40)]
                a3 = []
                for i in range(0, 20, 5):
                    a3.extend(a1[i: i+5])
                    a3.extend(a2[i: i+5])
                print('a3: {}'.format(a3))
                class_split = ClassSplit(40, [], random_sort=a3)
            elif random_seed == 50:
                a1 = [x for x in range(20)]
                a2 = [x for x in range(20, 40)]

                a3 = []
                for i in range(len(a1)):
                    a3.append(a1[i])
                    a3.append(a2[i])
                class_split = ClassSplit(40, [15, 5, 5, 5, 5, 5, 5], random_sort=a3)
            elif random_seed == 51:
                a1 = [x for x in range(20)]
                a2 = [x for x in range(20, 40)]

                a3 = []
                for i in range(len(a1)):
                    a3.append(a2[i])
                    a3.append(a1[i])
                class_split = ClassSplit(40, [15, 5, 5, 5, 5, 5, 5], random_sort=a3)
            else:
                class_split = ClassSplit(40, [], random_seed=random_seed)

            trainset = FlexAnimalSet(join('datasets', 'ImageNet'), True, class_split,
                                     [x for x in range(40)], None, crop=True)
            testset = FlexAnimalSet(join('datasets', 'ImageNet'), False, class_split,
                                    [x for x in range(40)], None, crop=True)

            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario == 'domain' else None
                train_datasets.append(SubDataset(trainset, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(testset, labels, target_transform=target_transform))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario=='domain' else classes_per_task*tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)