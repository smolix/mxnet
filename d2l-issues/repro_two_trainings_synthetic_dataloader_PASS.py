"""Use a synthetic Dataset (no OpenCV) inside Gluon DataLoader. If this
crashes, the issue is the DataLoader threading itself. If it doesn't,
the issue requires OpenCV imdecode."""
import os, sys, time
import mxnet as mx
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

from mxnet import autograd, gluon, init, np, npx
npx.set_np()

devices = [mx.gpu(i) for i in range(mx.context.num_gpus())]
print(f"devices={devices}", flush=True)

from d2l import mxnet as d2l

class SyntheticDataset(gluon.data.Dataset):
    def __init__(self, length=130, shape=(3, 224, 224), num_classes=2):
        self.length = length
        self.shape = shape
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return CPU NDArray + scalar label — exactly like an
        # ImageFolderDataset transformed item would look.
        x = mx.nd.random.uniform(shape=self.shape).astype('float32')
        y = mx.nd.array([idx % self.num_classes], dtype='float32').reshape(())
        return x, y

# Mimic the hotdog dataset shapes: 1000 train / 400 test, with last
# batch sizes that are divisible by 4 GPUs.
# 1000 train: 7 batches of 128 + 1 batch of 104 (104/4=26 ✓)
# 400 test:   3 batches of 128 + 1 batch of 16  (16/4=4 ✓)
train_dataset = SyntheticDataset(length=1000)
test_dataset = SyntheticDataset(length=400)

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

def train_with_synth_loader(net, lr, label, num_epochs=5, batch_size=128):
    train_iter = gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(test_dataset, batch_size)
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
print("[INFO] training finetune_net (synth dataset + DataLoader)", flush=True)
train_with_synth_loader(finetune_net, 0.01, "finetune")

print("[INFO] making scratch_net", flush=True)
scratch_net = make_scratch_net()
print("[INFO] training scratch_net (synth dataset + DataLoader)", flush=True)
train_with_synth_loader(scratch_net, 0.1, "scratch")

print("[INFO] DONE", flush=True)
