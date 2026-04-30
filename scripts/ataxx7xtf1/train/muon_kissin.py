import torch
import torch.distributed as dist

import logging

# Modified from "Kimi"'s Muon:  https://github.com/MoonshotAI/Moonlight
# Adapted for KataGo by LK (aka. loker404/Joe7/Kissin) and HZY (aka. Sigmoid/hzyhhzy)

@torch.compile
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
    #assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    assert G.ndim == 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() # this will be ignored if not supported
    #if torch.cuda.is_bf16_supported():
    #    X = G.bfloat16()
    #else:
    #    X = G.clone() # Fallback
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



def muon_update_kimi(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    
    # 保存原始形状
    original_shape = update.shape

    dims_gt_4 = sum(1 for s in original_shape if s > 4)
    if dims_gt_4 <= 1:
        raise ValueError(f"Muon 形状检查失败: original_shape {original_shape} 中大于4的维度只有 {dims_gt_4} 个 (必须 > 1)。这通常意味着该参数不适合使用 Muon。")

    #print(update.shape)
    if update.ndim == 4:  # 对于卷积滤波器的情况
        update = update.view(len(update), -1)
        #print(update.shape)
    if update.shape[0] <= 4 or update.shape[1] <= 4 :
        raise ValueError(f"Muon 形状检查失败: original_shape {original_shape} 被reshape成 {update.shape} ")

    
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, max(grad.size()))**0.5
    
    # 恢复原始形状
    if len(original_shape) == 4:
        update = update.view(original_shape)
    else:
        assert original_shape==update.shape, f"original_shape={original_shape}, update.shape={update.shape}"
    
    return update




def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)



class MuonWithAuxAdamKimi(torch.optim.Optimizer):
    def __init__(self, param_groups, momentum_default=0.95):
        self.is_distributed = dist.is_initialized()
        self.momentum_default=momentum_default
        for group in param_groups:
            # 确保所有参数组都有group_name
            #group["group_name"] = group.get("group_name", "")
            group["lr"] = 0 #会被update_and_return_lr_and_wd()覆盖
            
            
            # 设置默认学习率倍数 (adam学习率 = lr / muon_lr_multiplier)
            group["muon_lr_multiplier"] = group.get("muon_lr_multiplier", 8.0)
            
            if "use_muon" not in group:
                group["use_muon"] = self.is_muon_group(group["group_name"])
            
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # Muon参数组的默认值
                group["momentum"] = group.get("momentum", momentum_default)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) <= set(["params", "lr", "momentum", "weight_decay", 
                                                "use_muon", "group_name", "muon_lr_multiplier"])
            else:
                # Adam参数组的默认值
                group["betas"] = group.get("betas", (self.momentum_default,  0.995))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) <= set(["params", "lr", "betas", "eps", "weight_decay", 
                                                "use_muon", "group_name", "muon_lr_multiplier"])
        super().__init__(param_groups, dict())

    def is_muon_group(self, group_name: str) -> bool:
        """自动判断参数是否应该使用Muon优化"""
        # 根据组名判断
        if "output" in group_name.lower():
            return False
        if "gamma" in group_name.lower():
            #logging.info(f"{group_name} is output")
            return False
        if "noreg" in group_name.lower():
            #logging.info(f"{group_name} is output")
            return False
        if "normal" in group_name.lower():
            assert group_name=="normal" or group_name=="normal_attn", f"Unknown group_name: {group_name}, you should add it to is_muon_group()"
            return True
        assert False, f"Unknown group_name: {group_name}, you should add it to is_muon_group()"
        # 默认情况下根据参数维度判断
        #return param.ndim >= 2
        
        return False

    def _local_muon_state_for_checkpoint(self, optimizer_state_dict):
        local_muon_state = {}
        if not self.is_distributed:
            return local_muon_state
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        for group in optimizer_state_dict["param_groups"]:
            if not group.get("use_muon", False):
                continue
            for local_index, param_id in enumerate(group["params"]):
                if local_index % world_size != rank:
                    continue
                if param_id not in optimizer_state_dict["state"]:
                    continue
                state = optimizer_state_dict["state"][param_id]
                if "momentum_buffer" not in state:
                    continue
                local_muon_state[param_id] = {
                    "momentum_buffer": state["momentum_buffer"].detach().cpu()
                }
        return local_muon_state

    def state_dict_for_checkpoint(self):
        optimizer_state_dict = super().state_dict()
        if not self.is_distributed:
            return optimizer_state_dict
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_muon_state = self._local_muon_state_for_checkpoint(optimizer_state_dict)
        gathered_muon_state = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_muon_state, gathered_muon_state, dst=0)
        if rank != 0:
            return None
        for rank_state in gathered_muon_state:
            if rank_state is None:
                continue
            for param_id, state in rank_state.items():
                optimizer_state_dict["state"][param_id] = state
        return optimizer_state_dict

    def load_state_dict_for_checkpoint(self, state_dict):
        if not self.is_distributed:
            super().load_state_dict(state_dict)
            return
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        muon_non_local_param_ids = set()
        for group in state_dict["param_groups"]:
            if not group.get("use_muon", False):
                continue
            for local_index, param_id in enumerate(group["params"]):
                if local_index % world_size != rank:
                    muon_non_local_param_ids.add(param_id)
        filtered_state = dict(state_dict["state"])
        for param_id in muon_non_local_param_ids:
            if param_id in filtered_state:
                del filtered_state[param_id]
        filtered_state_dict = {
            "state": filtered_state,
            "param_groups": state_dict["param_groups"],
        }
        super().load_state_dict(filtered_state_dict)


    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            muon_lr = group["lr"] 
            adam_lr = group["lr"] / group["muon_lr_multiplier"]

        
            # 原有逻辑保持不变 (显式指定use_muon的情况)
            if group["use_muon"]:
                params = group["params"]
                if self.is_distributed:
                    num_to_pad = dist.get_world_size() - (len(params) - 1) % dist.get_world_size()
                    params_pad = params + [torch.empty_like(params[-1])] * num_to_pad
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
                        assert update.shape == p.shape
                        #if update.shape != p.shape:
                        #    update = update.view_as(p)
                        
                        p.mul_(1 - muon_lr * group["weight_decay"])
                        p.add_(update, alpha=-muon_lr)
                    if self.is_distributed:
                        dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                if "betas" not in group:
                    group["betas"] = group.get("betas", (self.momentum_default,  0.995))
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    
                    if "step" not in state:
                        state["step"] = 100000 
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                        state["step"], group["betas"], group["eps"])
                    p.mul_(1 - adam_lr * group["weight_decay"])
                    p.add_(update, alpha=-adam_lr)
