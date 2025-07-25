import math
import torch # 假设使用 PyTorch 张量

class PadamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, # 默认 eps 修改为 1e-6
                 weight_decay=1e-2, p=0.25, amsgrad=False): # 默认 p 修改为 0.25
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < p :
            raise ValueError(f"Invalid p value: {p}. It should be greater than 0.")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, p=p, amsgrad=amsgrad)
        super(PadamW, self).__init__(params, defaults) # 类名修改为 PadamW

    def __setstate__(self, state):
        super(PadamW, self).__setstate__(state) # 类名修改为 PadamW
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = [] # For AMSGrad
            state_steps = []
            beta1, beta2 = group['betas']
            p_val = group['p'] # 从 group 中获取 p 值
            eps_val = group['eps'] # 从 group 中获取 eps 值

            for param in group['params']:
                if param.grad is not None:
                    if param.grad.is_sparse:
                        raise RuntimeError('PadamW does not support sparse gradients, please consider SparseAdam instead')
                    params_with_grad.append(param)
                    grads.append(param.grad)

                    state = self.state[param]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state['step'] += 1
                    state_steps.append(state['step'])


            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]

                # Decoupled weight decay (AdamW style)
                if group['weight_decay'] != 0:
                    param.mul_(1 - group['lr'] * group['weight_decay'] / (1 - beta1) )

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2) # Use .conj() for complex gradients

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Corrected first moment estimate
                corrected_exp_avg = exp_avg / bias_correction1

                # Corrected second raw moment estimate for the denominator
                if group['amsgrad']:
                    max_exp_avg_sq = max_exp_avg_sqs[i]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    # Bias correction is applied to the max_exp_avg_sq
                    v_hat_eff = max_exp_avg_sq / bias_correction2
                    denom = v_hat_eff.pow(p_val).add_(eps_val)
                else:
                    v_hat = exp_avg_sq / bias_correction2
                    denom = v_hat.pow(p_val).add_(eps_val) # 使用 group['eps'] 和 group['p']

                # Parameter update
                update_val = corrected_exp_avg / denom
                param.add_(update_val, alpha=-group['lr'])

        return loss