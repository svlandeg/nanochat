"""
A nice and efficient mixed AdamW/Muon/SUMO Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon or SUMO.
Two versions are provided (SUMOAdamW/DistSUMOAdamW), for single GPU and distributed.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor
import math

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single GPU version of a SUMO-style optimizer (SUMOAdamW).
# Uses fused AdamW for non-matrix params and a SUMO-like update for 2D matrices.

class SUMOAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: SUMO for 2D matrix params, AdamW for others (single GPU).

    Param group schema:
        - 'params': list of parameters
        - 'kind': 'adamw' or 'muon'
        - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
        - For SUMO groups (kind == 'muon'):
            - 'lr': learning rate for matrix params
            - 'momentum': first-moment decay β (e.g. 0.95)
            - 'weight_decay': L2 weight decay in original space
            - Optional:
                - 'rank': low-rank dimension r (default: 16)
                - 'update_freq': subspace update frequency K (default: 100)
                - 'norm_growth_gamma': norm growth limiter γ (default: 1.1)
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # Reuse the fused AdamW kernel helpers from above
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            adamw_step_fused(
                p,
                grad,
                exp_avg,
                exp_avg_sq,
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_sumo(self, group: dict) -> None:
        params: list[Tensor] = group['params']
        if not params:
            return

        lr = group["lr"]
        beta = group.get("momentum", 0.95)
        weight_decay = group.get("weight_decay", 0.0)
        rank = int(group.get("rank", 16))
        update_freq = int(group.get("update_freq", 100))
        gamma = float(group.get("norm_growth_gamma", 1.1))

        for p in params:
            grad = p.grad
            if grad is None:
                continue

            # Fallback for non-matrix params: plain SGD with weight decay
            if grad.ndim != 2:
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)
                p.add_(grad, alpha=-lr)
                continue

            state = self.state[p]
            m, n = grad.shape

            if "step" not in state:
                state["step"] = 0
            state["step"] += 1

            eff_rank = max(1, min(rank, m, n))

            # Recompute subspace Q every update_freq steps (Block 1)
            recompute_q = ("Q" not in state) or (state["step"] % update_freq == 1)
            if recompute_q:
                try:
                    U, _, _ = torch.linalg.svd(grad, full_matrices=False)
                    state["Q"] = U[:, :eff_rank].to(dtype=grad.dtype, device=grad.device)
                except RuntimeError:
                    # If SVD fails and no previous Q, fall back to gradient descent
                    if "Q" not in state:
                        if weight_decay != 0.0:
                            p.add_(p, alpha=-lr * weight_decay)
                        p.add_(grad, alpha=-lr)
                        continue

            Q = state["Q"]

            # Project gradient to subspace: G_hat = Q^T G  (Block 2)
            G_hat = Q.transpose(-2, -1) @ grad  # (r, n)

            # First-order moment in subspace: M_t = β M_{t-1} + (1-β) G_hat
            if "M" not in state or state["M"].shape != G_hat.shape:
                state["M"] = torch.zeros_like(G_hat)
            M = state["M"]
            M.mul_(beta).add_(G_hat, alpha=(1.0 - beta))

            # Exact orthogonalization via SVD: O = U V^T
            try:
                U_m, _, Vh_m = torch.linalg.svd(M, full_matrices=False)
                O = (U_m @ Vh_m).to(dtype=M.dtype)
            except RuntimeError:
                O = M

            # Norm-growth limiter (Block 3)
            O_norm = torch.linalg.norm(O)
            prev_norm = state.get("O_norm", None)
            if prev_norm is not None and prev_norm > 0 and O_norm > 0:
                if (O_norm / prev_norm) > gamma:
                    O.mul_(gamma * prev_norm / (O_norm + 1e-8))
            state["O_norm"] = float(O_norm.item())

            # Full-space update (Block 4): ΔW = G - Q ( G_hat - O )
            delta = grad - Q @ (G_hat - O)

            # Shape-aware scaling (implicit layer-wise LR adaptation)
            scale = math.sqrt(float(max(m, n)))
            delta = delta / scale

            # Weight decay in original space
            if weight_decay != 0.0:
                p.add_(p, alpha=-lr * weight_decay)

            # Parameter update
            p.add_(delta, alpha=-lr)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                # Reuse 'muon' kind for matrix groups, but apply SUMO logic
                self._step_sumo(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistSUMOAdamW(torch.optim.Optimizer):
    """
    Distributed optimizer: SUMO-style updates for 2D matrix params, fused AdamW for others.

    This simplified version uses gradient all-reduce across ranks for each parameter:
      - For 'adamw' groups: all-reduce grads then apply fused AdamW as in MuonAdamW.
      - For 'muon' groups: all-reduce grads then apply the same SUMO-style update
        used in SUMOAdamW (low-rank moment orthogonalization via SVD).
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            adamw_step_fused(
                p,
                grad,
                exp_avg,
                exp_avg_sq,
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_sumo(self, group: dict) -> None:
        params: list[Tensor] = group['params']
        if not params:
            return

        lr = group["lr"]
        beta = group.get("momentum", 0.95)
        weight_decay = group.get("weight_decay", 0.0)
        rank = int(group.get("rank", 16))
        update_freq = int(group.get("update_freq", 100))
        gamma = float(group.get("norm_growth_gamma", 1.1))

        for p in params:
            grad = p.grad
            if grad is None:
                continue

            if grad.ndim != 2:
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)
                p.add_(grad, alpha=-lr)
                continue

            state = self.state[p]
            m, n = grad.shape

            if "step" not in state:
                state["step"] = 0
            state["step"] += 1

            eff_rank = max(1, min(rank, m, n))

            recompute_q = ("Q" not in state) or (state["step"] % update_freq == 1)
            if recompute_q:
                try:
                    U, _, _ = torch.linalg.svd(grad, full_matrices=False)
                    state["Q"] = U[:, :eff_rank].to(dtype=grad.dtype, device=grad.device)
                except RuntimeError:
                    if "Q" not in state:
                        if weight_decay != 0.0:
                            p.add_(p, alpha=-lr * weight_decay)
                        p.add_(grad, alpha=-lr)
                        continue

            Q = state["Q"]

            G_hat = Q.transpose(-2, -1) @ grad  # (r, n)

            if "M" not in state or state["M"].shape != G_hat.shape:
                state["M"] = torch.zeros_like(G_hat)
            M = state["M"]
            M.mul_(beta).add_(G_hat, alpha=(1.0 - beta))

            try:
                U_m, _, Vh_m = torch.linalg.svd(M, full_matrices=False)
                O = (U_m @ Vh_m).to(dtype=M.dtype)
            except RuntimeError:
                O = M

            O_norm = torch.linalg.norm(O)
            prev_norm = state.get("O_norm", None)
            if prev_norm is not None and prev_norm > 0 and O_norm > 0:
                if (O_norm / prev_norm) > gamma:
                    O.mul_(gamma * prev_norm / (O_norm + 1e-8))
            state["O_norm"] = float(O_norm.item())

            delta = grad - Q @ (G_hat - O)

            scale = math.sqrt(float(max(m, n)))
            delta = delta / scale

            if weight_decay != 0.0:
                p.add_(p, alpha=-lr * weight_decay)

            p.add_(delta, alpha=-lr)

    @torch.no_grad()
    def step(self):
        if not dist.is_initialized():
            raise RuntimeError("DistSUMOAdamW requires torch.distributed to be initialized")

        # Synchronize gradients across ranks
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # Apply local optimizer updates
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_sumo(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")
