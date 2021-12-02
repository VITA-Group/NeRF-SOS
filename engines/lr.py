
import torch
import numpy as np

class LRScheduler:

    def __init__(self, optimizer, init_lr, decay_rate, decay_steps):
        if np.isscalar(init_lr):
            init_lr = [init_lr]
        self.init_lr = init_lr

        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.optimizer = optimizer

        assert len(self.optimizer.param_groups) == len(init_lr), \
            "Number of lr does not match number of param groups."

    def step(self, step):
        for lr, param_group in zip(self.init_lr, self.optimizer.param_groups):
            new_lrate = lr * (self.decay_rate ** (step / self.decay_steps))
            param_group['lr'] = new_lrate

# # schedule learning rate: scheme I
# def scheduler_1(iter):
#     return args.lr1_decay_rate ** (iter / args.lr1_decay_iter)

# # schedule learning rate: scheme II
# def scheduler_2(iter):
#     iter = iter + 1
#     if iter < args.lr2_warmup_iter:
#         return iter / args.lr2_warmup_iter
#     elif iter > args.lr2_decay_iter:
#         r = (iter - args.lr2_decay_iter) / (args.N_iters - args.lr2_decay_iter)
#         return (1. - args.lr2_scale) * math.exp(-r) + args.lr2_scale
#     else:
#         return 1.