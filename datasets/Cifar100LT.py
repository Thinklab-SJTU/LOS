import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as T

def get_train_transform():
    transforms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    return T.Compose(transforms)

def get_val_transform():
    transforms = [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    return T.Compose(transforms)

def get_cifar100(root, args):
    transform_train = get_train_transform()
    transform_val = get_val_transform()

    train_dataset = CIFAR100_train(root, args, imb_ratio=args.imb_ratio, train=True, transform=transform_train)
    test_dataset = CIFAR100_val(root, transform=transform_val)
    print(f"#Train: {len(train_dataset)}, #Test: {len(test_dataset)}")
    return train_dataset, test_dataset

class CIFAR100_train(torchvision.datasets.CIFAR100):
    def __init__(self, root, args, imb_type='exp', imb_ratio=100, train=True, transform=None, target_transform=None, download=True):

        super(CIFAR100_train, self).__init__(root, train=train, transform=transform,
                                             target_transform=target_transform,
                                             download=download)
        self.args = args
        self.cls_num = 100
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, 1. / imb_ratio)
        self.transform_train = transform
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # print(selec_idx)
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_category(self, target):
        per_class_num = self.num_per_cls_dict[target]
        if per_class_num > 100: return 'Many'
        if 100 >= per_class_num >= 20: return 'Medium'
        if per_class_num < 20: return 'Few'

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img = self.transform_train(img)

        return img, target, index


class CIFAR100_val(torchvision.datasets.CIFAR100):
    def __init__(self, root, transform=None, indexs=None,
                 target_transform=None, download=True):
        super(CIFAR100_val, self).__init__(root, train=False, transform=transform, target_transform=target_transform,
                                           download=download)

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index