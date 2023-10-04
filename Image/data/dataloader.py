import torch, torchvision
from torchvision.datasets import CIFAR100, CIFAR10


class data():
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.train_data = None
        self.test_data = None
        self.BATCH_SIZE = 64
        self.model = model

    def loadData(self):
        tt = torchvision.transforms

        if self.dataset == 'mnist':
            if self.model == 'vgg':
                train_transform = tt.Compose(
                    [tt.Resize(size=(224, 224)), tt.ToTensor(), tt.Normalize((0.1307,), (0.3081,))])
                test_transform = tt.Compose(
                    [tt.Resize(size=(224, 224)), tt.ToTensor(), tt.Normalize((0.1307,), (0.3081))])
            else:
                train_transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1307,), (0.3081,))])
                test_transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1307,), (0.3081))])
            train_data = torchvision.datasets.MNIST(root='./mnist', train=True, download=True,
                                                    transform=train_transform)
            test_data = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=test_transform)

        elif self.dataset == 'cifar10':
            if self.model == 'vgg':
                train_transform = tt.Compose([tt.Resize(size=(224, 224)),
                                              tt.ToTensor(),
                                              tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])

                test_transform = tt.Compose([tt.Resize(size=(224, 224)),
                                             tt.ToTensor(),
                                             tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])
            else:
                train_transform = tt.Compose([
                    tt.ToTensor(),
                    tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                test_transform = tt.Compose([
                    tt.ToTensor(),
                    tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

            train_data = CIFAR10(download=True, root="./cifar10", transform=train_transform)
            test_data = CIFAR10(root="./cifar10", train=False, transform=test_transform)

        elif self.dataset == 'cifar100':
            # online calculated stats for cifar100
            stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))

            if self.model == 'vgg':
                train_transform = tt.Compose([tt.Resize(size=(224, 224)),
                                              tt.RandomHorizontalFlip(),
                                              tt.RandomCrop(32, padding=4, padding_mode="reflect"),
                                              tt.ToTensor(),
                                              tt.Normalize(*stats)
                                              ])

                test_transform = tt.Compose([tt.Resize(size=(224, 224)),
                                             tt.ToTensor(),
                                             tt.Normalize(*stats)
                                             ])
            else:
                # read documentation on each function below
                # it's to prevent overfitting by adding noise
                train_transform = tt.Compose([
                    tt.RandomHorizontalFlip(),
                    tt.RandomCrop(32, padding=4, padding_mode="reflect"),
                    tt.ToTensor(),
                    tt.Normalize(*stats)
                ])

                test_transform = tt.Compose([
                    tt.ToTensor(),
                    tt.Normalize(*stats)
                ])

            train_data = CIFAR100(download=True, root="./cifar100", transform=train_transform)
            test_data = CIFAR100(root="./cifar100", train=False, transform=test_transform)

        else:
            print(f"You have entered {self.dataset} which is an invalid dataset")

        print(
            f"For the {self.dataset} dataset, we are trying to classify {len(train_data.classes)} different classes of the following type:")
        print(train_data.classes)

        self.train_data = train_data
        self.test_data = test_data

        # pin_memory automatically puts fetched data tensors in pinned memory
        train_dl = torch.utils.data.DataLoader(self.train_data, batch_size=self.BATCH_SIZE, shuffle=True)

        test_dl = torch.utils.data.DataLoader(self.test_data, batch_size=self.BATCH_SIZE, shuffle=False)

        return train_dl, test_dl

