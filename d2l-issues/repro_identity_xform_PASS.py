"""ImageFolderDataset with identity transform — OpenCV imdecode runs in
__getitem__ but NO per-item augmentation ops are applied. To make the
batch usable, we resize to (3, 224, 224) on GPU after split_and_load,
NOT in the per-item Python transform.

If this CRASHES: trigger is OpenCV imdecode (free of decoded buffer too
soon, or its interaction with engine on CPU NDArrays).
If this PASSES: trigger is the augmentation ops (mx.image transforms on
CPU NDArrays queued via the engine before split_and_load runs)."""
import os, sys, time
import mxnet as mx
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

from mxnet import autograd, gluon, init, np, npx
npx.set_np()

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


def identity_xform(img):
    """No per-item ops at all. Just resize on the CPU NDArray so the
    batch shapes are consistent for stacking. ImageFolderDataset returns
    HWC uint8 NDArrays of varying H,W — we MUST resize to a fixed shape
    so DataLoader can stack them. We do that with mx.image.imresize which
    is a single C++ op call (vs the multi-op augmentation chain)."""
    # imresize is a single op — minimal per-item engine traffic.
    img = mx.image.imresize(img, 224, 224)
    # Convert HWC uint8 → CHW float32 [0, 1] without going through ToTensor.
    img = img.astype('float32').transpose((2, 0, 1)) / 255.0
    return img


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


def train_with_identity(net, lr, label, num_epochs=5, batch_size=128):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(identity_xform), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(identity_xform), batch_size)
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
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} train done",
              flush=True)
        # eval
        for X, y in test_iter:
            X_shards, y_shards = d2l.split_batch(X, y, devices)
            preds = [net(x) for x in X_shards]
            for p in preds:
                _ = p.argmax(axis=1).asnumpy()
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} eval done",
              flush=True)


print("[INFO] making finetune_net", flush=True)
finetune_net = make_finetune_net()
print("[INFO] training finetune_net (identity xform on ImageFolderDataset)",
      flush=True)
train_with_identity(finetune_net, 0.01, "finetune")

print("[INFO] making scratch_net", flush=True)
scratch_net = make_scratch_net()
print("[INFO] training scratch_net (identity xform on ImageFolderDataset)",
      flush=True)
train_with_identity(scratch_net, 0.1, "scratch")

print("[INFO] DONE", flush=True)
