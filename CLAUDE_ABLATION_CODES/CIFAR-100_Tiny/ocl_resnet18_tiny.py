# models/ocl_resnet18_tiny.py
"""
Supports four methods via --method flag:

  ours        : Full method — Meta-Learned Lambda + Amphibian Gating
  fixed_lambda: Ablation 1  — Fixed lambda (no meta-learning), Amphibian kept
  no_reg      : Ablation 2  — No regularization at all, Amphibian kept
  er          : Baseline    — Plain Experience Replay (no lambda, no gating)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_LAMBDA_MIN  = -13.8    # exp(-13.8) ~1e-6
LOG_LAMBDA_MAX  =  -3.0    # exp(-3.0)  ~0.05
FIXED_LAMBDA    =   0.001  # used when method == 'fixed_lambda'

LAYER_GROUPS = [
    ("conv1",  0),
    ("bn1",    0),
    ("layer1", 0),
    ("layer2", 1),
    ("layer3", 2),
    ("layer4", 3),
    ("fc",     4),
]
N_LAMBDA_GROUPS = 5

VALID_METHODS = ("ours", "fixed_lambda", "no_reg", "er")


class OCLResNet18Tiny(nn.Module):

    def __init__(self, num_classes=200, lr=0.05, no_meta=False,
                 dataset="tiny", lambda_lr=2e-3, method="ours"):
        super().__init__()
        assert method in VALID_METHODS, \
            f"method must be one of {VALID_METHODS}, got '{method}'"

        self.method  = method
        self.dataset = dataset
        # no_meta kept for backward compat — maps to fixed_lambda behaviour
        self.no_meta = no_meta or (method == "fixed_lambda")

        # ── Backbone ──────────────────────────────────────────────────────────
        base = resnet18(weights=None)
        if dataset == "cifar":
            base.conv1  = nn.Conv2d(3, 64, kernel_size=3,
                                    stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.net = base

        self.loss_fn = nn.CrossEntropyLoss()
        self.opt     = torch.optim.SGD(
            self.net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0
        )

        # ── Per-group log-lambdas (only optimised when method == 'ours') ──────
        self.log_lambdas = nn.Parameter(
            torch.full((N_LAMBDA_GROUPS,), -7.6, device=DEVICE)
        )
        self.lambda_opt = torch.optim.Adam(
            [self.log_lambdas], lr=lambda_lr, weight_decay=0.0
        )

        self.theta_old = None

    # ── Lambda helpers ────────────────────────────────────────────────────────

    @property
    def lambdas(self):
        return self.log_lambdas.clamp(LOG_LAMBDA_MIN, LOG_LAMBDA_MAX).exp()

    def _lambda_for(self, name):
        """
        ours        → learned per-layer lambda
        fixed_lambda→ scalar constant FIXED_LAMBDA for every layer
        no_reg / er → never called (reg loss is 0)
        """
        if self.method == "fixed_lambda":
            return torch.tensor(FIXED_LAMBDA, device=DEVICE)
        # ours
        for prefix, idx in LAYER_GROUPS:
            if name.startswith(prefix):
                return self.lambdas[idx]
        return torch.tensor(0.0, device=DEVICE)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def save_old_params(self):
        self.theta_old = {
            n: p.detach().clone()
            for n, p in self.net.named_parameters()
            if p.requires_grad
        }

    # ── Regularisation loss ───────────────────────────────────────────────────

    def _reg_loss(self, named_params=None):
        """
        Returns L2 anchoring loss.
        Zero when:  method == 'no_reg'
                    method == 'er'
                    theta_old is None (first task)
        """
        if self.method in ("no_reg", "er"):
            return torch.tensor(0.0, device=DEVICE)
        if self.theta_old is None:
            return torch.tensor(0.0, device=DEVICE)

        live = dict(self.net.named_parameters())
        reg  = torch.tensor(0.0, device=DEVICE)
        for name, p_old in self.theta_old.items():
            p = (named_params.get(name, live.get(name))
                 if named_params else live.get(name))
            if p is None:
                continue
            lam  = self._lambda_for(name)
            reg  = reg + (lam / 2.0) * (p - p_old).pow(2).sum()
        return reg

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        return self.net(x)

    # ── Observe: one gradient step ────────────────────────────────────────────

    def observe(self, x, y):
        self.train()
        logits = self.net(x)
        loss   = self.loss_fn(logits, y) + self._reg_loss()

        self.opt.zero_grad()
        loss.backward()

        # Amphibian gating — applied for all methods EXCEPT 'er'
        if self.method != "er":
            with torch.no_grad():
                for p in self.net.parameters():
                    if p.grad is None:
                        continue
                    std  = p.data.std().clamp(min=1e-6)
                    gate = torch.sigmoid((p.data / std).clamp(-2, 2))
                    p.grad.mul_(gate)

        self.opt.step()
        return loss.item()

    # ── Meta-update lambda (ours only) ────────────────────────────────────────

    def meta_update_lambda(self, x_in, y_in, x_mem, y_mem):
        """No-op for everything except method == 'ours'."""
        if self.method != "ours" or self.theta_old is None:
            return

        self.lambda_opt.zero_grad()
        named = {n: p for n, p in self.net.named_parameters()
                 if p.requires_grad}

        logits_in = self.net(x_in)
        loss_in   = self.loss_fn(logits_in, y_in) + self._reg_loss()

        grads = torch.autograd.grad(
            loss_in, list(named.values()),
            create_graph=True, allow_unused=True
        )

        lr = self.opt.param_groups[0]["lr"]
        vp = {}
        for (name, p), g in zip(named.items(), grads):
            if g is not None:
                std  = p.data.std().clamp(min=1e-6)
                gate = torch.sigmoid(p.data / std)
                vp[name] = p - lr * (g * gate)
            else:
                vp[name] = p

        meta_loss = self.loss_fn(self._forward_with(x_mem, vp), y_mem)
        meta_loss.backward()
        self.lambda_opt.step()

        with torch.no_grad():
            self.log_lambdas.clamp_(LOG_LAMBDA_MIN, LOG_LAMBDA_MAX)

    # ── Functional forward (used by meta-update only) ─────────────────────────

    def _forward_with(self, x, vp):
        net = self.net

        def _bn(inp, mod, prefix):
            w = vp.get(f"{prefix}.weight", mod.weight)
            b = vp.get(f"{prefix}.bias",   mod.bias)
            return F.batch_norm(inp, mod.running_mean, mod.running_var,
                                w, b, training=False, eps=mod.eps)

        def _basic_block(inp, block, prefix):
            identity = inp
            w1  = vp.get(f"{prefix}.conv1.weight", block.conv1.weight)
            out = F.conv2d(inp, w1, None,
                           stride=block.conv1.stride, padding=1)
            out = _bn(out, block.bn1, f"{prefix}.bn1")
            out = F.relu(out, inplace=False)
            w2  = vp.get(f"{prefix}.conv2.weight", block.conv2.weight)
            out = F.conv2d(out, w2, None, stride=1, padding=1)
            out = _bn(out, block.bn2, f"{prefix}.bn2")
            if block.downsample is not None:
                w_ds = vp.get(f"{prefix}.downsample.0.weight",
                              block.downsample[0].weight)
                identity = F.conv2d(inp, w_ds, None,
                                    stride=block.downsample[0].stride,
                                    padding=0)
                identity = _bn(identity, block.downsample[1],
                               f"{prefix}.downsample.1")
            return F.relu(out + identity, inplace=False)

        # Stem
        w_c1 = vp.get("conv1.weight", net.conv1.weight)
        if self.dataset == "cifar":
            out = F.conv2d(x, w_c1, None, stride=1, padding=1)
            out = _bn(out, net.bn1, "bn1")
            out = F.relu(out, inplace=False)
        else:
            out = F.conv2d(x, w_c1, None, stride=2, padding=3)
            out = _bn(out, net.bn1, "bn1")
            out = F.relu(out, inplace=False)
            out = F.max_pool2d(out, 3, stride=2, padding=1)

        for ln, lmod in [("layer1", net.layer1), ("layer2", net.layer2),
                          ("layer3", net.layer3), ("layer4", net.layer4)]:
            for i, block in enumerate(lmod):
                out = _basic_block(out, block, f"{ln}.{i}")

        out  = F.adaptive_avg_pool2d(out, (1, 1))
        out  = torch.flatten(out, 1)
        w_fc = vp.get("fc.weight", net.fc.weight)
        b_fc = vp.get("fc.bias",   net.fc.bias)
        return F.linear(out, w_fc, b_fc)
