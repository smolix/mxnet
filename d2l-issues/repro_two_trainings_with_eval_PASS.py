"""Add 'evaluate-style' forward passes between training epochs to test
whether evaluate_accuracy_gpus is what triggers the crash. The d2l
train_ch13 loop calls evaluate_accuracy_gpus(test_iter) ONCE per epoch
(without autograd.record); we mimic that here without real data."""
import sys
import time
import mxnet as mx
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

from mxnet import autograd, gluon, init, np, npx
npx.set_np()

devices = [mx.gpu(i) for i in range(mx.context.num_gpus())]
print(f"devices={devices}", flush=True)

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

def train_with_eval(net, lr, label, num_epochs=5, iters_per_epoch=20,
                     batch_size=128):
    net.reset_ctx(devices)
    net.hybridize()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': 0.001})
    per_dev = batch_size // len(devices)
    t0 = time.time()
    for epoch in range(num_epochs):
        # Training loop
        for i in range(iters_per_epoch):
            xs = [np.random.uniform(size=(per_dev, 3, 224, 224), ctx=d)
                  for d in devices]
            ys = [np.random.randint(0, 2, size=(per_dev,), ctx=d) for d in devices]
            with autograd.record():
                losses = [loss_fn(net(x), y).sum() for x, y in zip(xs, ys)]
            for l in losses:
                l.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} train done", flush=True)
        # Evaluate-style forward pass (no autograd, no backward)
        # — mimics d2l.evaluate_accuracy_gpus
        for _ in range(4):  # 4 test batches
            xs = [np.random.uniform(size=(per_dev, 3, 224, 224), ctx=d)
                  for d in devices]
            preds = [net(x) for x in xs]
            # Force eval (matches metric.add(float(accuracy(...)), ...))
            for p in preds:
                _ = p.argmax(axis=1).asnumpy()
        mx.npx.waitall()
        print(f"[{time.time()-t0:6.1f}s] {label}: epoch {epoch} eval done", flush=True)

print("[INFO] making finetune_net", flush=True)
finetune_net = make_finetune_net()
print("[INFO] training finetune_net (5 epochs + eval each)", flush=True)
train_with_eval(finetune_net, 0.01, "finetune")

print("[INFO] making scratch_net", flush=True)
scratch_net = make_scratch_net()
print("[INFO] training scratch_net (5 epochs + eval each)", flush=True)
train_with_eval(scratch_net, 0.1, "scratch")

print("[INFO] DONE", flush=True)
