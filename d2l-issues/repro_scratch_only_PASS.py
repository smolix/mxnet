"""Run ONLY the scratch_net training (no finetune_net first). If this
crashes, the trigger is not 'second' training — it's just multi-GPU
training of a fresh resnet18_v2 with real OpenCV-decoded images."""
import sys
import mxnet as mx
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

import matplotlib
matplotlib.use("Agg")
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))

normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

print("[INFO] building scratch net", flush=True)
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())

batch_size = 128
num_epochs = 5
train_iter = gluon.data.DataLoader(
    train_imgs.transform_first(train_augs), batch_size, shuffle=True)
test_iter = gluon.data.DataLoader(
    test_imgs.transform_first(test_augs), batch_size)
devices = d2l.try_all_gpus()
print(f"[INFO] devices={devices}", flush=True)
scratch_net.reset_ctx(devices)
scratch_net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(scratch_net.collect_params(), 'sgd',
                        {'learning_rate': 0.1, 'wd': 0.001})
print("[INFO] entering train_ch13", flush=True)
d2l.train_ch13(scratch_net, train_iter, test_iter, loss, trainer, num_epochs,
               devices)
print("[INFO] DONE", flush=True)
