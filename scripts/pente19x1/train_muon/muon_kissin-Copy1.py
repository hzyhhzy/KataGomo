import torch
import torch.distributed as dist

import logging

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
#     momentum.lerp_(grad, 1 - beta)
#     update = grad.lerp_(momentum, beta) if nesterov else momentum
#     if update.ndim == 4: # for the case of conv filters
#         update = update.view(len(update), -1)
#     update = zeropower_via_newtonschulz5(update, steps=ns_steps)
#     update *= max(1, grad.size(-2) / grad.size(-1))**0.5
#     return update
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    
    # 保存原始形状
    original_shape = update.shape
    
    if update.ndim == 4:  # 对于卷积滤波器的情况
        update = update.view(len(update), -1)
    
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    #logging.info(grad.shape)
    
    # 恢复原始形状
    if len(original_shape) == 4:
        update = update.view(original_shape)
    
    return update

def muon_update_kimi(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    
    # 保存原始形状
    original_shape = update.shape
    
    if update.ndim == 4:  # 对于卷积滤波器的情况
        update = update.view(len(update), -1)
    
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, max(grad.size()))**0.5
    
    # 恢复原始形状
    if len(original_shape) == 4:
        update = update.view(original_shape)
    
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single Opimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                #assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.99))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                #assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])










class MuonWithAuxAdamKissin(torch.optim.Optimizer):
    def __init__(self, param_groups):
        self.is_distributed = dist.is_initialized()
        for group in param_groups:
            # 确保所有参数组都有group_name
            group["group_name"] = group.get("group_name", "")
            group["lr"] = group.get("lr", 0.02 if group.get("use_muon", False) else 3e-4)
            
            # 设置默认学习率倍数 (Muon学习率 = adam_lr * muon_lr_multiplier)
            group["muon_lr_multiplier"] = group.get("muon_lr_multiplier", 20.0)
            # 设置WeightDecay (Muon WeightDecay = adam_wd * muon_wd_multiplier)
            group["muon_wd_multiplier"] = group.get("muon_wd_multiplier", 1.0)
            
            if "use_muon" not in group:
                group["use_muon"] = None
            
            if group["use_muon"] is not None:
                if group["use_muon"]:
                    group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                    # Muon参数组的默认值
                    group["momentum"] = group.get("momentum", 0.95)
                    group["weight_decay"] = group.get("weight_decay", 0)
                    assert set(group.keys()) <= set(["params", "lr", "momentum", "weight_decay", 
                                                    "use_muon", "group_name", "muon_lr_multiplier"])
                else:
                    # Adam参数组的默认值
                    group["betas"] = group.get("betas", (0.9, 0.99))
                    group["eps"] = group.get("eps", 1e-8)
                    group["weight_decay"] = group.get("weight_decay", 0)
                    assert set(group.keys()) <= set(["params", "lr", "betas", "eps", "weight_decay", 
                                                    "use_muon", "group_name", "muon_lr_multiplier"])
        super().__init__(param_groups, dict())

    def _auto_detect_muon(self, param, group_name: str) -> bool:
        """自动判断参数是否应该使用Muon优化"""
        
        # 根据组名判断
        if "output" in group_name.lower():
            #logging.info(f"{group_name} is output")
            return False
        if "gamma" in group_name.lower():
            #logging.info(f"{group_name} is output")
            return False
        if "normal" in group_name.lower():
            assert(group_name=="normal")
            #logging.info(f"{group_name} {param.shape} is normal")
            return True
        # 默认情况下根据参数维度判断
        #logging.info(group_name + " no output and normal")
        #logging.info(param.shape)
        #return param.ndim >= 2
        return False

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            # 初始化参数状态
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # 自动判断是否使用Muon
                    if group["use_muon"] is None:
                        state["use_muon"] = self._auto_detect_muon(p, group["group_name"])
                    else:
                        state["use_muon"] = group["use_muon"]
            
            if group.get("use_muon", None) is None:
                # 处理自动判断的情况
                params = [p for p in group["params"] if p.grad is not None and self.state[p]["use_muon"]]
                if params:
                    params = sorted(params, key=lambda x: x.size(), reverse=True)
                    if self.is_distributed:
                        params_pad = params + [torch.empty_like(params[-1])] * (len(params) % dist.get_world_size())
                    else:
                        params_pad = params
                
                # Muon参数处理 (使用20倍学习率)
                for base_i in range(len(params))[::dist.get_world_size() if self.is_distributed else 1]:
                    if not self.is_distributed or base_i + dist.get_rank() < len(params):
                        p = params[base_i + (dist.get_rank() if self.is_distributed else 0)]
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group.get("momentum", 0.95))
                        
                        # 确保update和p的形状一致
                        if update.shape != p.shape:
                            update = update.view_as(p)
                            
                        muon_lr = group["lr"] * group["muon_lr_multiplier"]
                        #logging.info(f"muon_lr {muon_lr}")
                        p.mul_(1 - muon_lr * group.get("weight_decay", 0))
                        p.add_(update, alpha=-muon_lr)
                    
                    if params and self.is_distributed:
                        dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
                
                # Adam参数处理 (使用原始学习率)
                beta1, beta2 = group.get("betas", (0.9, 0.99))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state["use_muon"]:
                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                           state["step"], group.get("betas", (0.9, 0.95)), group.get("eps", 1e-10))
                        p.mul_(1 - group["lr"] * group.get("weight_decay", 0))
                        p.add_(update, alpha=-group["lr"])
            else:
                # 原有逻辑保持不变 (显式指定use_muon的情况)
                if group["use_muon"]:
                    params = group["params"]
                    if self.is_distributed:
                        params_pad = params + [torch.empty_like(params[-1])] * (len(params) % dist.get_world_size())
                    else:
                        params_pad = params
                    for base_i in range(len(params))[::dist.get_world_size() if self.is_distributed else 1]:
                        if not self.is_distributed or base_i + dist.get_rank() < len(params):
                            p = params[base_i + (dist.get_rank() if self.is_distributed else 0)]
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)
                            update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                            
                            # 确保update和p的形状一致
                            if update.shape != p.shape:
                                update = update.view_as(p)
                            
                            muon_lr = group["lr"] * group["muon_lr_multiplier"]
                            p.mul_(1 - muon_lr * group["weight_decay"])
                            p.add_(update, alpha=-muon_lr)
                        if self.is_distributed:
                            dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
                else:
                    beta1, beta2 = group["betas"]
                    for p in group["params"]:
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                           state["step"], group["betas"], group["eps"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])



class MuonWithAuxAdamKimi(torch.optim.Optimizer):
    def __init__(self, param_groups, momentum_default=0.95):
        self.is_distributed = dist.is_initialized()
        for group in param_groups:
            # 确保所有参数组都有group_name
            group["group_name"] = group.get("group_name", "")
            group["lr"] = group.get("lr", 0.02 if group.get("use_muon", False) else 3e-4)
            
            # 设置默认学习率倍数 (Muon学习率 = adam_lr * muon_lr_multiplier)
            group["muon_lr_multiplier"] = group.get("muon_lr_multiplier", 1.0)
            
            if "use_muon" not in group:
                group["use_muon"] = None
            
            if group["use_muon"] is not None:
                if group["use_muon"]:
                    group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                    # Muon参数组的默认值
                    group["momentum"] = group.get("momentum", momentum_default)
                    group["weight_decay"] = group.get("weight_decay", 0)
                    assert set(group.keys()) <= set(["params", "lr", "momentum", "weight_decay", 
                                                    "use_muon", "group_name", "muon_lr_multiplier"])
                else:
                    # Adam参数组的默认值
                    group["betas"] = group.get("betas", (0.9, 0.999))
                    group["eps"] = group.get("eps", 1e-8)
                    group["weight_decay"] = group.get("weight_decay", 0)
                    assert set(group.keys()) <= set(["params", "lr", "betas", "eps", "weight_decay", 
                                                    "use_muon", "group_name", "muon_lr_multiplier"])
        super().__init__(param_groups, dict())

    def _auto_detect_muon(self, param, group_name: str) -> bool:
        """自动判断参数是否应该使用Muon优化"""
        # 根据组名判断
        if "output" in group_name.lower():
            return False
        if "gamma" in group_name.lower():
            #logging.info(f"{group_name} is output")
            return False
        if "normal" in group_name.lower():
            assert(group_name=="normal")
            return True
        # 默认情况下根据参数维度判断
        #return param.ndim >= 2
        
        return False

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            # 初始化参数状态
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # 自动判断是否使用Muon
                    if group["use_muon"] is None:
                        state["use_muon"] = self._auto_detect_muon(p, group["group_name"])
                    else:
                        state["use_muon"] = group["use_muon"]
            
            if group.get("use_muon", None) is None:
                # 处理自动判断的情况
                params = [p for p in group["params"] if p.grad is not None and self.state[p]["use_muon"]]
                if params:
                    params = sorted(params, key=lambda x: x.size(), reverse=True)
                    if self.is_distributed:
                        params_pad = params + [torch.empty_like(params[-1])] * (len(params) % dist.get_world_size())
                    else:
                        params_pad = params
                
                # Muon参数处理 (kimi,1倍学习率)
                for base_i in range(len(params))[::dist.get_world_size() if self.is_distributed else 1]:
                    if not self.is_distributed or base_i + dist.get_rank() < len(params):
                        p = params[base_i + (dist.get_rank() if self.is_distributed else 0)]
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update_kimi(p.grad, state["momentum_buffer"], beta=group.get("momentum", 0.95))
                        
                        # 确保update和p的形状一致
                        if update.shape != p.shape:
                            update = update.view_as(p)
                            
                        muon_lr = group["lr"] * group["muon_lr_multiplier"]
                        p.mul_(1 - muon_lr * group.get("weight_decay", 0))
                        p.add_(update, alpha=-muon_lr)
                    
                    if params and self.is_distributed:
                        dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
                
                # Adam参数处理 (使用原始学习率)
                beta1, beta2 = group.get("betas", (0.9, 0.99))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state["use_muon"]:
                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                           state["step"], group.get("betas", (0.9, 0.95)), group.get("eps", 1e-10))
                        p.mul_(1 - group["lr"] * group.get("weight_decay", 0))
                        p.add_(update, alpha=-group["lr"])
            else:
                # 原有逻辑保持不变 (显式指定use_muon的情况)
                if group["use_muon"]:
                    params = group["params"]
                    if self.is_distributed:
                        params_pad = params + [torch.empty_like(params[-1])] * (len(params) % dist.get_world_size())
                    else:
                        params_pad = params
                    for base_i in range(len(params))[::dist.get_world_size() if self.is_distributed else 1]:
                        if not self.is_distributed or base_i + dist.get_rank() < len(params):
                            p = params[base_i + (dist.get_rank() if self.is_distributed else 0)]
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)
                            update = muon_update_kimi(p.grad, state["momentum_buffer"], beta=group["momentum"])
                            
                            # 确保update和p的形状一致
                            if update.shape != p.shape:
                                update = update.view_as(p)
                            
                            muon_lr = group["lr"] * group["muon_lr_multiplier"]
                            p.mul_(1 - muon_lr * group["weight_decay"])
                            p.add_(update, alpha=-muon_lr)
                        if self.is_distributed:
                            dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
                else:
                    beta1, beta2 = group["betas"]
                    for p in group["params"]:
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                           state["step"], group["betas"], group["eps"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])