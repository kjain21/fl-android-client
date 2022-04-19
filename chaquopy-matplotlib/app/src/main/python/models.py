from cgi import test
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from torch.nn import functional as F
import torchvision
from torchvision.datasets import MNIST
# from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import os
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10
from com.chaquo.python import Python

BATCH_SIZE = 256 # needed for config optimizers; TODO: figure out if needed
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class CIFARLitResnet(pl.LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}




class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height) 
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        self.log('val_acc', acc)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('test_loss', loss)
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,), (0.3081,))])
      
    # prepare transforms standard to MNIST
    files_dir = str(Python.getPlatform().getApplication().getFilesDir())
    files_dir = files_dir + "/MNIST"

    train_files_dir = files_dir + "/processed/train.pt"
    test_files_dir = files_dir + "/processed/test.pt"

    print(files_dir)

    mnist_train = MNIST(files_dir, train=True, download=False, transform=transform)
    mnist_test = MNIST(files_dir, train=False, download=False, transform=transform)

    print("got here after mnist_trian and test")
    
    self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
    self.mnist_test = mnist_test

  def train_dataloader(self):
    print("in train dataloader")
    return DataLoader(self.mnist_train, batch_size=64, num_workers=0, pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.mnist_val, batch_size=64, num_workers=0, pin_memory=True)

  def test_dataloader(self):
    print("in test dataloaader")
    return DataLoader(self.mnist_test, batch_size=64, num_workers=0, pin_memory=True)


class CIFAR10DataModule(pl.LightningDataModule):

    name = 'cifar10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = PATH_DATASETS,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 256,
            seed: int = 42,
            client_num: int = 0,
            non_iid: bool = False,
            *args,
            **kwargs,
    ):
        """
        Copied from pl_bolts
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
        """
        super().__init__(*args, **kwargs)
        self.dims = (3, 32, 32)
        self.DATASET = CIFAR10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split
        self.client_num = client_num
        self.non_iid = non_iid
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = test_transforms

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        self.DATASET(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor(), **self.extra_args)

    def train_dataloader(self):
        """
        CIFAR train set removes a subset to use for validation
        """
        transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        if self.non_iid:
          labels = []
          for _, label in dataset:
              labels.append(label)
          ind = {}

          for i in range(10):
              ind[i] = []
          for i in range(len(labels)):
              ind[int(labels[i])].append(i)
          # combined ind, 2*client_num + 2*client_num+1
          # TODO: make it based on total number of clients
          # currently each client has 4 labels (each label on 2 different clients)
          #  + ind[(self.client_num*2+2)%10] + ind[(self.client_num*2+3)%10]
          label_set = ind[self.client_num*2] + ind[self.client_num*2+1] + ind[(self.client_num*2+2)%10] + ind[(self.client_num*2+3)%10]
          subset = Subset(dataset, label_set)
          loader = DataLoader(
              subset,
              batch_size=self.batch_size,
              shuffle=True,
              num_workers=self.num_workers,
              drop_last=True,
              pin_memory=True)

          return loader

        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """
        CIFAR10 val set uses a subset of the training set for validation
        """
        transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def test_dataloader(self):
        """
        CIFAR10 test set uses the test split
        """
        transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, train=False, download=False, transform=transforms, **self.extra_args)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def default_transforms(self):
        cf10_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            cifar10_normalization()
        ])
        return cf10_transforms