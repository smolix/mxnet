"""Minimal repro hypothesis: the d2l fine-tuning crash is triggered by
running two sequential train_fine_tuning calls in the same process. Each
call creates+destroys a Trainer (and its KVStore). If the second call
SEGVs even without real data / OpenCV / animator, the bug is in the
Trainer/KVStore re-init or in interaction with stale hybridized state."""
import sys
import time
import mxnet as mx
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)

from mxnet import autograd, gluon, init, np, npx
npx.set_np()

devices = [mx.gpu(i) for i in range(mx.context.num_gpus())]
print(f"devices={devices}", flush=True)

# Two distinct networks (matching d2l fine-tuning notebook structure)
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

def train_n_iters(net, lr, label, num_iters=200, batch_size=128):
    net.reset_ctx(devices)
    net.hybridize()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': 0.001})
    per_dev = batch_size // len(devices)
    t0 = time.time()
    for i in range(num_iters):
        xs = [np.random.uniform(size=(per_dev, 3, 224, 224), ctx=d) for d in devices]
        ys = [np.random.randint(0, 2, size=(per_dev,), ctx=d) for d in devices]
        with autograd.record():
            losses = [loss_fn(net(x), y).sum() for x, y in zip(xs, ys)]
        for l in losses:
            l.backward()
        trainer.step(batch_size, ignore_stale_grad=True)
        if i % 50 == 0:
            mx.npx.waitall()
            print(f"[{time.time()-t0:6.1f}s] {label}: iter {i}", flush=True)
    mx.npx.waitall()
    print(f"[{time.time()-t0:6.1f}s] {label}: completed {num_iters} iters", flush=True)

# 200 iters per training matches d2l fine-tuning's ~5 epochs × 8 iters (small dataset)
# but with batch_size=128 / 4 GPUs = 32 per dev — same as notebook
print("[INFO] making finetune_net", flush=True)
finetune_net = make_finetune_net()
print("[INFO] training finetune_net", flush=True)
train_n_iters(finetune_net, 0.01, "finetune", num_iters=200)

print("[INFO] making scratch_net", flush=True)
scratch_net = make_scratch_net()
print("[INFO] training scratch_net", flush=True)
train_n_iters(scratch_net, 0.1, "scratch", num_iters=200)

print("[INFO] DONE", flush=True)
