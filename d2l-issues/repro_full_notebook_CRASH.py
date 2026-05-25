"""Load mxnet first (so its initialize.cc signal handlers register), then
re-install Python's faulthandler on top. faulthandler.enable() calls
sigaction(), which replaces whatever signal() handler mxnet installed.
Use chain=False so faulthandler dumps Python traceback then re-raises
SIGSEGV to terminate cleanly."""
import sys

# Load mxnet first so its static-init signal handlers run.
import mxnet as mx  # noqa: F401

# Now re-install faulthandler on top so we get a real Python+C traceback.
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)
import signal
# Explicitly tell faulthandler about ABRT (faulthandler covers SEGV/BUS/FPE/ILL by default,
# but ABRT is registered separately).
for sig_name in ('SIGABRT',):
    sig = getattr(signal, sig_name, None)
    if sig is not None:
        try:
            faulthandler.register(sig, file=sys.stderr, all_threads=True, chain=False)
        except Exception as e:
            print(f"could not register {sig_name}: {e}", file=sys.stderr)

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

print("[INFO] loading pretrained net", flush=True)
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)

finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
for p in finetune_net.output.collect_params().values():
    p.lr_mult = 10

def train_fine_tuning(net, learning_rate, label, batch_size=128, num_epochs=5):
    print(f"[INFO] {label}: building data iters", flush=True)
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    print(f"[INFO] {label}: devices={devices}", flush=True)
    net.reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    print(f"[INFO] {label}: entering train_ch13", flush=True)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
    print(f"[INFO] {label}: train_ch13 returned", flush=True)

train_fine_tuning(finetune_net, 0.01, "finetune")

print("[INFO] building scratch net", flush=True)
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1, "scratch")

print("[INFO] DONE", flush=True)
