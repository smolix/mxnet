"""Add the real ImageFolderDataset + DataLoader (OpenCV imdecode) to the
synthetic two-trainings repro. If this crashes, the trigger is the
DataLoader / OpenCV decode interaction with sequential train cycles."""
import os, sys, time
import mxnet as mx
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

from mxnet import autograd, gluon, init, np, npx
npx.set_np()

devices = [mx.gpu(i) for i in range(mx.context.num_gpus())]
print(f"devices={devices}", flush=True)

# Real hotdog data (uses OpenCV imdecode under the hood)
from d2l import mxnet as d2l
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

def make_finetune_net():
    pretrained = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
    net = gluon.model_zoo.vision.resnet18_v2(classes=2)
    net.features = pretrained.features
    net.output.initialize(init.Xavier())
    return net

def make_scratch_net():
    net = gluon.model_zoo.vision.resnet18_v2(classes=2)
    net.initialize(init=init.Xavier())
    return net

def split_batch(features, labels, devices):
    return d2l.split_batch(features, labels, devices)

def train_with_real_loader(net, lr, label, num_epochs=5, batch_size=128):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    net.reset_ctx(devices)
    net.hybridize()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': 0.001})
    t0 = time.time()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X_shards, y_shards = split_batch(X, y, devices)
            with autograd.record():
                losses = [loss_fn(net(x), y_) for x, y_ in zip(X_shards, y_shards)]
            for l in losses:
                l.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} train done",
              flush=True)
        # Evaluate-style forward pass
        for X, y in test_iter:
            X_shards, y_shards = split_batch(X, y, devices)
            preds = [net(x) for x in X_shards]
            for p in preds:
                _ = p.argmax(axis=1).asnumpy()
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} eval done",
              flush=True)

print("[INFO] making finetune_net", flush=True)
finetune_net = make_finetune_net()
print("[INFO] training finetune_net (5 epochs + eval each, real loader)", flush=True)
train_with_real_loader(finetune_net, 0.01, "finetune")

print("[INFO] making scratch_net", flush=True)
scratch_net = make_scratch_net()
print("[INFO] training scratch_net (5 epochs + eval each, real loader)", flush=True)
train_with_real_loader(scratch_net, 0.1, "scratch")

print("[INFO] DONE", flush=True)
