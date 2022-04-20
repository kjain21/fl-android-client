from logging import root
import os
os.environ["PATH"] = ":".join([p for p in os.environ["PATH"].split(":")
                               if os.access(p, os.R_OK | os.X_OK)])
import io
# import matplotlib.pyplot as plt
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models import LightningMNISTClassifier, MNISTDataModule
from com.chaquo.python import Python

# class Client(pl.LightningModule):
#     def __init__(
#         self, model):
#         super().__init__()
#         self.model = model
    
#     def update_model(self, server):
#         self.load_state_dict(server.state_dict())

# def plot(x, y):
#     xa = [float(word) for word in x.split()]
#     ya = [float(word) for word in y.split()]

#     fig, ax = plt.subplots()
#     ax.plot(xa, ya)

#     f = io.BytesIO()
#     plt.savefig(f, format="png")
#     return f.getvalue()

import signal

def valid_signals():
    return {signal._int_to_enum(x, signal.Signals) for x in range(1, signal.NSIG)}
signal.valid_signals = valid_signals


class Client(pl.LightningModule):
    def __init__(
        self, model):
        super().__init__()
        self.model = model
    
    def update_model(self, server):
        self.load_state_dict(server.state_dict())
    
    def test_func(self):
        # logger = TensorBoardLogger(save_dir=params.logs_dir, name=client_name)
        current_mod = MNISTDataModule()
        # logger = TensorBoardLogger()
        os.environ["HOME"] = str(Python.getPlatform().getApplication().getFilesDir())
        root_dir =  str(Python.getPlatform().getApplication().getFilesDir())
        trainer_local = pl.Trainer(precision = 32, max_epochs=10, default_root_dir=root_dir)
        print("pre fit")
        trainer_local.fit(self.model, current_mod)
        print("post fit")
        return "Hello World 8"

def client_train():
    tc = Client(LightningMNISTClassifier())
    # import os.path

    # files_dir = str(Python.getPlatform().getApplication().getFilesDir())

    # name_of_file = "test_name"

    # completeName = os.path.join(files_dir, name_of_file+".txt")         

    # file1 = open(completeName, "w")

    # toFile = "random stuff"

    # file1.write(toFile)

    # file1.close()
    return tc.test_func()
    # return "Hello world 10"


# if __name__ == '__main__':
#     print()