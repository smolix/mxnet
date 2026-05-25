"""Synthetic data on CPU then split_and_load to GPUs. Same code pattern as
d2l.train_batch_ch13. If this crashes, the trigger is the CPU→GPU
split_and_load pattern in combination with two sequential trainings."""
import os, sys, time
import mxnet as mx
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

from mxnet import autograd, gluon, init, np, npx
npx.set_np()

devices = [mx.gpu(i) for i in range(mx.context.num_gpus())]
print(f"devices={devices}", flush=True)

from d2l import mxnet as d2l

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

def train_with_cpu_transfer(net, lr, label, num_epochs=5, iters_per_epoch=20,
                             batch_size=128):
    net.reset_ctx(devices)
    net.hybridize()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': 0.001})
    t0 = time.time()
    for epoch in range(num_epochs):
        for i in range(iters_per_epoch):
            # NDArray on CPU first, then split_and_load to GPUs (same as
            # d2l.train_batch_ch13 with a real DataLoader feed)
            features = np.random.uniform(size=(batch_size, 3, 224, 224),
                                          ctx=mx.cpu())
            labels = np.random.randint(0, 2, size=(batch_size,),
                                        ctx=mx.cpu())
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            with autograd.record():
                losses = [loss_fn(net(x), y_) for x, y_ in zip(X_shards, y_shards)]
            for l in losses:
                l.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} train done",
              flush=True)
        # eval pattern
        for _ in range(4):
            features = np.random.uniform(size=(batch_size, 3, 224, 224),
                                          ctx=mx.cpu())
            labels = np.random.randint(0, 2, size=(batch_size,),
                                        ctx=mx.cpu())
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            preds = [net(x) for x in X_shards]
            for p in preds:
                _ = p.argmax(axis=1).asnumpy()
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} eval done",
              flush=True)

print("[INFO] making finetune_net", flush=True)
finetune_net = make_finetune_net()
print("[INFO] training finetune_net (5 epochs + eval each, CPU→GPU)", flush=True)
train_with_cpu_transfer(finetune_net, 0.01, "finetune")

print("[INFO] making scratch_net", flush=True)
scratch_net = make_scratch_net()
print("[INFO] training scratch_net (5 epochs + eval each, CPU→GPU)", flush=True)
train_with_cpu_transfer(scratch_net, 0.1, "scratch")

print("[INFO] DONE", flush=True)
