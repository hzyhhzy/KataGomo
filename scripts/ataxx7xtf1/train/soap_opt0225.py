import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain

# Parts of the code are modifications of Pytorch's AdamW optimizer
# Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py
MIN_MATRIX_DIM = 15 # min dims of matrix to apply SOAP

class SOAP(optim.Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
        max_precond_dim (`int`, *optional*, defaults to 10000):
            Maximum dimension of the preconditioner.
            Set to 10000, so that we exclude most common vocab sizes while including layers.
        merge_dims (`bool`, *optional*, defaults to `False`):
            Whether or not to merge dimensions of the preconditioner.
        precondition_1d (`bool`, *optional*, defaults to `False`):
            Whether or not to precondition 1D gradients.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer. 
            Helps at large precondition_frequency (~100 in our experiments), 
            but hurts performance at small precondition_frequency (~10 in our experiments).
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias correction in Adam.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float= -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int=10,
        max_precond_dim: int=10000, # 
        merge_dims: bool = False, # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format
        
    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor based on the two largest dimensions.
        """
        
        shape = grad.shape
        if len(shape) < 2:
            return grad.reshape(-1)
        
        # Find the two largest dimensions and their indices
        sorted_indices = sorted(range(len(shape)), key=lambda i: shape[i], reverse=True)
        idx1, idx2 = sorted_indices[0], sorted_indices[1]
        max1, max2 = shape[idx1], shape[idx2]
        
        if max1 > max_precond_dim or max2 < MIN_MATRIX_DIM:
            return grad.reshape(-1)
        
        first_idx = min(idx1, idx2)
        
        # Merge [0, first_idx] and [first_idx + 1, last]
        dim1 = 1
        for i in range(first_idx + 1):
            dim1 *= shape[i]
            
        dim2 = 1
        for i in range(first_idx + 1, len(shape)):
            dim2 *= shape[i]
            
        return grad.reshape(dim1, dim2)               

    @torch.no_grad()
    def step(self, closure = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            # Cache frequently accessed group parameters
            merge_dims = group["merge_dims"]
            max_precond_dim = group['max_precond_dim']
            lr = group["lr"]
            betas = group["betas"]
            beta1, beta2 = betas
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            normalize_grads = group["normalize_grads"]
            correct_bias = group["correct_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if 'Q' not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group['precondition_frequency'],
                        precondition_1d=group['precondition_1d'],
                        shampoo_beta=(group['shampoo_beta'] if group['shampoo_beta'] >= 0 else betas[1]),
                        max_precond_dim=max_precond_dim,
                        merge_dims=merge_dims,
                    )
                    self.update_preconditioner(grad, state,
                                               max_precond_dim=max_precond_dim,
                                               merge_dims=merge_dims,
                                               precondition_1d=group["precondition_1d"])
                    continue # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = self.project(grad, state, merge_dims=merge_dims,
                                              max_precond_dim=max_precond_dim)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                step_count = state["step"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

                denom = exp_avg_sq.sqrt().add_(eps)

                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                # exp_avg_projected = self.project(exp_avg, state, merge_dims=merge_dims,
                #                                  max_precond_dim=max_precond_dim)
                exp_avg_projected = exp_avg

                step_size = lr
                if correct_bias:
                    # Optimized: compute bias correction factors
                    bias_correction1 = 1.0 - beta1 ** step_count
                    bias_correction2 = 1.0 - beta2 ** step_count
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                norm_grad = self.project_back(exp_avg_projected / denom, state, merge_dims=merge_dims,
                                                 max_precond_dim=max_precond_dim)

                if normalize_grads:
                    norm_grad = norm_grad / (1e-30+torch.mean(norm_grad**2)**0.5)

                p.add_(norm_grad, alpha=-step_size)

                # From AdamW code: Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if weight_decay > 0.0:
                    p.add_(p, alpha=(-lr * weight_decay))

                # Update is done after the gradient step to avoid using current gradients in the projection.
                self.update_preconditioner(grad, state,
                                               max_precond_dim=max_precond_dim,
                                               merge_dims=merge_dims,
                                               precondition_1d=group["precondition_1d"])

        return loss
    
    def init_preconditioner(self, grad, state, precondition_frequency=10,
                            shampoo_beta=0.95, max_precond_dim=10000, precondition_1d=False,
                            merge_dims=False):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state['GG'] = [] # Will hold all the preconditioner matrices (L and R in the paper).
        state['_merge_dims_cache'] = {}  # Cache for merge_dims results
        state['_original_shape'] = grad.shape  # Store original shape

        if merge_dims:
            grad = self.merge_dims(grad, max_precond_dim)
            # Cache the merge_dims result
            state['_merge_dims_cache']['merged'] = True
            state['_merge_dims_cache']['merged_shape'] = grad.shape
                    
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:

            for sh in grad.shape:
                if sh > max_precond_dim:
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device))

        state['Q'] = None # Will hold all the eigenbases of the preconditioner.
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta          
        
    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        Optimized to use cached merge_dims results when available.
        Further optimized to use matmul instead of tensordot for better GPU Tensor Core utilization.
        """
        original_shape = grad.shape
        if merge_dims:
            # Use cached merge_dims result if available
            if '_merge_dims_cache' in state and state['_merge_dims_cache'].get('merged', False):
                grad = grad.reshape(state['_merge_dims_cache']['merged_shape'])
            else:
                grad = self.merge_dims(grad, max_precond_dim)

        for mat in state['Q']:
            if len(mat) > 0:
                # Optimized: use matmul instead of tensordot for better GPU performance
                # tensordot(grad, mat, dims=[[0], [0]]) computes dot product along grad's dim 0 and mat's dim 0
                # For 2D grad (d0, d1) and mat (d0, d0), result is (d1, d0)
                # For multi-dimensional grad (d0, d1, d2, ...) and mat (d0, d0), result is (d1, d2, ..., d0)
                if grad.dim() == 2:
                    # grad is (d0, d1), mat is (d0, d0), result should be (d1, d0)
                    # tensordot(grad, mat, dims=[[0], [0]]) = grad.T @ mat
                    grad = grad.T @ mat
                else:
                    assert(False)
                    # Reshape to (dim0, -1), compute matmul, then reshape back
                    # tensordot(grad, mat, dims=[[0], [0]]) where grad is (d0, d1, d2, ...) and mat is (d0, d0)
                    # Result shape is (d1, d2, ..., d0)
                    # Equivalent to: reshape grad to (d0, -1), compute grad_2d.T @ mat, then reshape to (d1, d2, ..., d0)
                    grad_shape = grad.shape
                    grad_2d = grad.reshape(grad_shape[0], -1)  # (d0, d1*d2*...)
                    grad_2d = grad_2d.T @ mat  # (d1*d2*..., d0) @ (d0, d0) = (d1*d2*..., d0)
                    grad = grad_2d.reshape(*grad_shape[1:], mat.shape[1])
            else:
                assert(len(grad.shape)==1)
                if(len(grad.shape)>1):
                    print(original_shape,grad.shape,len(state['Q']),mat)
                    permute_order = list(range(1, len(grad.shape))) + [0]
                    grad = grad.permute(permute_order)

        if merge_dims:
            grad = grad.reshape(original_shape)
        return grad
        
    def update_preconditioner(self, grad, state,
                              max_precond_dim=10000, merge_dims=False, precondition_1d=False):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        Optimized to only reproject exp_avg when Q is actually updated and use cached merge_dims.
        Further optimized to use matmul instead of tensordot for better GPU Tensor Core utilization.
        """
        # Check if Q will be updated
        q_will_update = state['Q'] is None or (state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0)

        # Only project back exp_avg if Q will be updated
        if q_will_update and state["Q"] is not None:
            state["exp_avg"] = self.project_back(state["exp_avg"], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)

        if merge_dims:
            # Use cached merge_dims result if available
            if '_merge_dims_cache' in state and state['_merge_dims_cache'].get('merged', False):
                grad = grad.reshape(state['_merge_dims_cache']['merged_shape'])
            else:
                grad = self.merge_dims(grad, max_precond_dim)
                
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                # Optimized: use outer product directly
                state['GG'][0].lerp_(grad.outer(grad), 1-state['shampoo_beta'])
        else:
            for idx, sh in enumerate(grad.shape):
                if sh <= max_precond_dim:
                    # Reshape: (dim_idx, other_dims_product)
                    target_grad = grad.permute(idx, *range(idx), *range(idx+1, len(grad.shape)))
                    target_grad = target_grad.reshape(sh, -1)
                    # Compute outer product: (sh, sh) = (sh, other_dims) @ (other_dims, sh)
                    outer_product = target_grad @ target_grad.T
                    state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])

        if state['Q'] is None:
            state['Q'] = self.get_orthogonal_matrix(state['GG'])
        elif state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)

        # Only project exp_avg if Q was updated
        if q_will_update and state["step"] > 0:
            state["exp_avg"] = self.project(state["exp_avg"], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim) 

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient back to the original space.
        Optimized to use cached merge_dims results when available.
        Further optimized to use matmul instead of tensordot for better GPU Tensor Core utilization.
        """
        original_shape = grad.shape
        if merge_dims:
            # Use cached merge_dims result if available
            if '_merge_dims_cache' in state and state['_merge_dims_cache'].get('merged', False):
                grad = grad.reshape(state['_merge_dims_cache']['merged_shape'])
            else:
                grad = self.merge_dims(grad, max_precond_dim)

        for mat in state['Q']:
            if len(mat) > 0:
                # Optimized: use matmul instead of tensordot for better GPU performance
                # tensordot(grad, mat, dims=[[0], [1]]) computes dot product along grad's dim 0 and mat's dim 1
                # For 2D grad (d0, d1) and mat (d0, d0), result is (d1, d0)
                # For multi-dimensional grad (d0, d1, d2, ...) and mat (d0, d0), result is (d1, d2, ..., d0)
                if grad.dim() == 2:
                    # grad is (d0, d1), mat is (d0, d0), result should be (d1, d0)
                    # tensordot(grad, mat, dims=[[0], [1]]) = grad.T @ mat.T
                    grad = grad.T @ mat.T
                else:
                    # Reshape to (dim0, -1), compute matmul, then reshape back
                    # tensordot(grad, mat, dims=[[0], [1]]) where grad is (d0, d1, d2, ...) and mat is (d0, d0)
                    # Result shape is (d1, d2, ..., d0)
                    # Equivalent to: reshape grad to (d0, -1), compute grad_2d.T @ mat.T, then reshape to (d1, d2, ..., d0)
                    grad_shape = grad.shape
                    grad_2d = grad.reshape(grad_shape[0], -1)  # (d0, d1*d2*...)
                    grad_2d = grad_2d.T @ mat.T  # (d1*d2*..., d0) @ (d0, d0).T = (d1*d2*..., d0) @ (d0, d0) = (d1*d2*..., d0)
                    grad = grad_2d.reshape(*grad_shape[1:], mat.shape[0])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            grad = grad.reshape(original_shape)
        return grad
        

    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        Optimized to avoid unnecessary type conversions.
        """
        final = []
        for m in mat:
            if len(m) == 0:
                final.append([])
                continue

            # Preserve original dtype and device
            original_dtype = m.dtype
            original_device = m.device

            # Only convert if not float32
            if m.dtype != torch.float32:
                m_float = m.float()
            else:
                m_float = m

            # Add small regularization for numerical stability
            m_reg = m_float + 1e-30 * torch.eye(m_float.shape[0], device=m_float.device)

            try:
                _, Q = torch.linalg.eigh(m_reg)
            except RuntimeError:
                # Fall back to float64 if eigh fails
                m_reg = m_reg.double() + 1e-30 * torch.eye(m_float.shape[0], device=m_float.device, dtype=torch.float64)
                _, Q = torch.linalg.eigh(m_reg)
                Q = Q.float()

            Q = torch.flip(Q, [1])

            # Convert back to original dtype and device
            if original_dtype != torch.float32:
                Q = Q.to(original_device).to(original_dtype)

            final.append(Q)
        return final
        

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration
        followed by torch.linalg.qr decomposition.
        Optimized to avoid unnecessary type conversions and reduce redundant computations.
        Further optimized to use more efficient matrix multiplication.
        """
        precond_list = state['GG']
        orth_list = state['Q']

        # Handle exp_avg_sq reshape once
        orig_shape = state['exp_avg_sq'].shape
        if merge_dims:
            # Use cached merge_dims result if available
            if '_merge_dims_cache' in state and state['_merge_dims_cache'].get('merged', False):
                exp_avg_sq = state['exp_avg_sq'].reshape(state['_merge_dims_cache']['merged_shape'])
            else:
                exp_avg_sq = self.merge_dims(state['exp_avg_sq'], max_precond_dim)
        else:
            exp_avg_sq = state['exp_avg_sq']

        final = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if len(m) == 0:
                final.append([])
                continue

            # Preserve original dtype and device
            original_dtype = m.dtype
            original_device = m.device

            # Only convert if not float32
            if m.dtype != torch.float32:
                m_float = m.float()
                o_float = o.float()
            else:
                m_float = m
                o_float = o

            # Estimate eigenvalues and sort
            # Optimized: use torch.diag for eigenvalue estimation
            est_eig = torch.diag(o_float.T @ m_float @ o_float)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o_sorted = o_float[:, sort_idx]

            # Power iteration + QR decomposition
            # Optimized: matmul is already used here
            power_iter = m_float @ o_sorted
            Q, _ = torch.linalg.qr(power_iter)

            # Convert back to original dtype and device
            if original_dtype != torch.float32:
                Q = Q.to(original_device).to(original_dtype)

            final.append(Q)

        # Restore exp_avg_sq shape
        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state['exp_avg_sq'] = exp_avg_sq
        return final
    
    