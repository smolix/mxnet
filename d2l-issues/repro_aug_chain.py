"""Test: ImageFolderDataset + ONLY the FIRST augmentation. Then progressively
add more. This bisects which augmentation (or combination) triggers the
crash.

We control which augs are active via an env var AUG_STAGE:
  STAGE=0: identity (baseline — should PASS, confirmed)
  STAGE=1: + RandomResizedCrop(224)
  STAGE=2: + RandomResizedCrop + RandomFlipLeftRight
  STAGE=3: + RandomResizedCrop + RandomFlipLeftRight + ToTensor
  STAGE=4: + full chain (Normalize too) — known CRASH
"""
import os, sys, time
import mxnet as mx
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

from mxnet import autograd, gluon, init, np, npx
npx.set_np()

STAGE = int(os.environ.get('AUG_STAGE', '1'))
print(f"AUG_STAGE = {STAGE}", flush=True)

devices = [mx.gpu(i) for i in range(mx.context.num_gpus())]
print(f"devices={devices}", flush=True)

from d2l import mxnet as d2l
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))

T = gluon.data.vision.transforms

# Build the augmentation pipeline incrementally
augs_list = []
if STAGE >= 1:
    augs_list.append(T.RandomResizedCrop(224))
if STAGE >= 2:
    augs_list.append(T.RandomFlipLeftRight())
if STAGE >= 3:
    augs_list.append(T.ToTensor())
if STAGE >= 4:
    augs_list.append(T.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]))

# We need a fixed-shape output for batch stacking. If we don't have
# ToTensor in the chain, also force HWC->CHW float so the trainer accepts.
# Add a finalizing identity-style step on top of whatever the user asked
# for, so we always get a (3, 224, 224) float32 tensor at the end.
def finalize(img):
    # If STAGE < 1, image is variable-sized HWC uint8 — resize first.
    if STAGE < 1:
        img = mx.image.imresize(img, 224, 224)
    # If STAGE < 3 (no ToTensor), do CHW + float conversion now.
    if STAGE < 3:
        img = img.astype('float32').transpose((2, 0, 1)) / 255.0
    return img

# Compose: augs + finalize
class _Pipe:
    def __init__(self, augs, finalize):
        self._augs = augs
        self._finalize = finalize
    def __call__(self, img):
        for aug in self._augs:
            img = aug(img)
        img = self._finalize(img)
        return img

pipe = _Pipe(augs_list, finalize)
# Test pipe
sample, _ = train_imgs[0]
sample = pipe(sample)
print(f"pipe output shape: {sample.shape}, dtype: {sample.dtype}", flush=True)


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


def train(net, lr, label, num_epochs=5, batch_size=128):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(pipe), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(pipe), batch_size)
    net.reset_ctx(devices)
    net.hybridize()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': 0.001})
    t0 = time.time()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X_shards, y_shards = d2l.split_batch(X, y, devices)
            with autograd.record():
                losses = [loss_fn(net(x), y_) for x, y_ in zip(X_shards, y_shards)]
            for l in losses:
                l.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} train done", flush=True)
        for X, y in test_iter:
            X_shards, y_shards = d2l.split_batch(X, y, devices)
            preds = [net(x) for x in X_shards]
            for p in preds:
                _ = p.argmax(axis=1).asnumpy()
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} eval done", flush=True)


print(f"[INFO] STAGE={STAGE}: training finetune_net", flush=True)
finetune_net = make_finetune_net()
train(finetune_net, 0.01, "finetune")

print(f"[INFO] STAGE={STAGE}: training scratch_net", flush=True)
scratch_net = make_scratch_net()
train(scratch_net, 0.1, "scratch")

print("[INFO] DONE", flush=True)
