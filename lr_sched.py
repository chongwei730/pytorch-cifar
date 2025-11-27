# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random as _rand
import random
import torch
import numpy as np
from torch.optim import Adam, SGD
from typing import Iterable, List, Tuple, Optional
from torch.nn.utils import vector_to_parameters, parameters_to_vector

class CustomReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.7, patience=7, 
                 threshold=1e-4, threshold_mode='rel', cooldown=0, 
                 min_lr=1e-6, eps=1e-8, verbose=False, injection_prob=0.2, 
                 injection_mode='random', injection_factor=1.5, lr=0.05, warmup_epochs=5):
        

        super().__init__(optimizer, mode, factor, patience, threshold, 
                         threshold_mode, cooldown, min_lr, eps, verbose)

        
        # Store custom attributes specific to this class
        self.injection_prob = injection_prob
        self.injection_factor = injection_factor
        self.injection_mode = injection_mode  # 'fixed' or 'long_tail'
        self.original_lrs = None  # Store original learning rates during injection
        self.min_lr = min_lr  # Store minimum learning rate
        self.warmup_epochs = warmup_epochs
        self.base_max_lrs = [lr for g in optimizer.param_groups]
        self.patience = self.patience
        self.lr = lr
        self.factor = factor

        # Predefine a long-tail distribution for injection factors
        self.injection_distribution = self._generate_long_tail_distribution()

    def _generate_long_tail_distribution(self):
        """Generate a long-tail distribution for injection factors."""
        distribution = [1.0] * 80 + [random.uniform(1.0, 2.0) for _ in range(20)]
        random.shuffle(distribution)
        return distribution

    def _expand_injection_distribution(self):
        """Expand the upper bound of the injection distribution."""
        current_max = max(self.injection_distribution)
        increment = self.factor
        distribution = [1.0] * 80 + [random.uniform(1.0, current_max+increment) for _ in range(20)]
        self.injection_distribution = distribution
        current_max = max(self.injection_distribution)
        print(f"[Injection Dist] Expanded with: {current_max}")


    def restore_original_lr(self):
        """Restore the original learning rates after injection."""
        if self.original_lrs is not None:
            for param_group, original_lr in zip(self.optimizer.param_groups, self.original_lrs):
                param_group['lr'] = original_lr
            self.original_lrs = None


    def step(self, metrics, epoch=None):
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            return
        # if metrics.item() == 0.0:
        #     return


        # Restore original learning rates if they were modified
        self.restore_original_lr()
 

        # Perform the regular ReduceLROnPlateau step
        if self.num_bad_epochs >= self.patience:
            self._expand_injection_distribution()



        super().step(metrics, epoch)

        # current_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        # if all(abs(clr - blr) < 1e-7 for clr, blr in zip(current_lrs, self.base_max_lrs)):
        #     print("Current learning rates equal to base max learning rates. Skip step.")
        #     return


        # Probabilistically inject a large learning rate
        if self.injection_mode == 'fixed':
            if _rand.random() < self.injection_prob:
                self.original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]  # Save original learning rates
                injection_factor = self.injection_factor
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * injection_factor
                self.num_bad_epochs = self.patience
            return self.original_lrs
            
        elif self.injection_mode == 'random':
            self.original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]  # Save original learning rates
            injection_factor = random.choice(self.injection_distribution)  # Sample from the predefined distribution
            # print("epoch", epoch)
            # print("distribution max:", max(self.injection_distribution))
            # print("patience:", self.patience, "num_bad_epochs",self.num_bad_epochs)
            # print("injection factor", injection_factor)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * injection_factor


            if injection_factor > 1.0:
                self.num_bad_epochs = 0 
            return self.original_lrs
        else:
            return

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random as _rand
import random
import torch
import numpy as np
from torch.optim import Adam, SGD
from typing import Iterable, List, Tuple, Optional
from torch.nn.utils import vector_to_parameters, parameters_to_vector
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random as _rand
import random
import torch
import numpy as np
from typing import Iterable, List, Tuple, Optional
import matplotlib.pyplot as plt




class CustomReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.7, patience=7, 
                 threshold=1e-4, threshold_mode='rel', cooldown=0, 
                 min_lr=1e-6, eps=1e-8, verbose=False, injection_prob=0.2, 
                 injection_mode='random', injection_factor=1.5, lr=0.05, warmup_epochs=5):
        

        super().__init__(optimizer, mode, factor, patience, threshold, 
                         threshold_mode, cooldown, min_lr, eps, verbose)

        
        # Store custom attributes specific to this class
        self.injection_prob = injection_prob
        self.injection_factor = injection_factor
        self.injection_mode = injection_mode  # 'fixed' or 'long_tail'
        self.original_lrs = None  # Store original learning rates during injection
        self.min_lr = min_lr  # Store minimum learning rate
        self.warmup_epochs = warmup_epochs
        self.base_max_lrs = [lr for g in optimizer.param_groups]
        self.patience = self.patience
        self.lr = lr
        self.factor = factor

        # Predefine a long-tail distribution for injection factors
        self.injection_distribution = self._generate_long_tail_distribution()

    def _generate_long_tail_distribution(self):
        """Generate a long-tail distribution for injection factors."""
        distribution = [1.0] * 80 + [random.uniform(1.0, 2.0) for _ in range(20)]
        random.shuffle(distribution)
        return distribution

    def _expand_injection_distribution(self):
        """Expand the upper bound of the injection distribution."""
        current_max = max(self.injection_distribution)
        increment = self.factor
        distribution = [1.0] * 80 + [random.uniform(1.0, current_max+increment) for _ in range(20)]
        self.injection_distribution = distribution
        current_max = max(self.injection_distribution)
        print(f"[Injection Dist] Expanded with: {current_max}")


    def restore_original_lr(self):
        """Restore the original learning rates after injection."""
        if self.original_lrs is not None:
            for param_group, original_lr in zip(self.optimizer.param_groups, self.original_lrs):
                param_group['lr'] = original_lr
            self.original_lrs = None


    def step(self, metrics, epoch=None):
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            return
        # if metrics.item() == 0.0:
        #     return


        # Restore original learning rates if they were modified
        self.restore_original_lr()
 

        # Perform the regular ReduceLROnPlateau step
        if self.num_bad_epochs >= self.patience:
            self._expand_injection_distribution()



        super().step(metrics, epoch)

        # current_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        # if all(abs(clr - blr) < 1e-7 for clr, blr in zip(current_lrs, self.base_max_lrs)):
        #     print("Current learning rates equal to base max learning rates. Skip step.")
        #     return


        # Probabilistically inject a large learning rate
        if self.injection_mode == 'fixed':
            if _rand.random() < self.injection_prob:
                self.original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]  # Save original learning rates
                injection_factor = self.injection_factor
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * injection_factor
                self.num_bad_epochs = self.patience
            return self.original_lrs
            
        elif self.injection_mode == 'random':
            self.original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]  # Save original learning rates
            injection_factor = random.choice(self.injection_distribution)  # Sample from the predefined distribution
            # print("epoch", epoch)
            # print("distribution max:", max(self.injection_distribution))
            # print("patience:", self.patience, "num_bad_epochs",self.num_bad_epochs)
            # print("injection factor", injection_factor)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * injection_factor


            if injection_factor > 1.0:
                self.num_bad_epochs = 0 
            return self.original_lrs
        else:
            return



class LineSearchScheduler():
    def __init__(self, optimizer, start_lr, model, num_search=16, optimizer_type="SGD", injection=True, search_mode="backtrack"):
        """
        num_search: maximum number of searches
        start_lr: maximum LR to start if backtrack/ minimum LR to start if forward
        optimizer_type: Option: SGD, SGD_momentum, Adam
        """



        self.optimizer = optimizer
        self.num_search = num_search
        self.start_lr = start_lr
        self.model = model
        self.optimizer_type = optimizer_type
        self.injection=injection
        self.prev_fvals = deque(maxlen=2)
        self.prev_alpha = start_lr
        self.search_mode = search_mode
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.start_lr
        self.paras = [p for p in self.model.parameters() if p.requires_grad]
        self.injection_distribution = self._generate_long_tail_distribution()
    

    def _unflatten_like(self, flat_np):
        flat = torch.from_numpy(flat_np).to(self.paras[0].device, dtype=self.paras[0].dtype)
        outs, off = [], 0
        for p in self.paras:
            n = p.numel()
            outs.append(flat[off:off+n].view_as(p))
            off += n
        return outs
    
    def _generate_long_tail_distribution(self):
        distribution = [1.0] * 80 + [random.uniform(1.0, 2.0) for _ in range(20)]
        random.shuffle(distribution)
        return distribution
    
    def state_dict(self):
        return {
            'last_lr': self.prev_alpha,
        }
    

    def _sgd_momentum_direction_from_state(self, grads, fallback_to_neg_grad=True):
        dirs = []
        pg0 = self.optimizer.param_groups[0] if len(self.optimizer.param_groups) > 0 else {}
        momentum = pg0.get("momentum", 0.0)   

        grads_tensors = self._unflatten_like(grads)
        for p, g in zip(self.paras, grads_tensors):
            st = self.optimizer.state.get(p, None)
            if st is not None and "momentum_buffer" in st:
                v = st["momentum_buffer"]
                g = torch.zeros_like(p) if g is None else g
                v_t = momentum * v + g
                update = -v_t
                dirs.append(update.view(-1))
            else:
                if fallback_to_neg_grad:
                    g = -grads
                    dirs.append(g.reshape(-1))
                else:
                    dirs.append(torch.zeros_like(p).reshape(-1))

        return torch.cat(dirs).detach().cpu().numpy()
    
    
    def _adam_direction_from_state(self, grads, fallback_to_neg_grad=True, use_bias_correction=True):
        dirs = []
        pg0 = self.optimizer.param_groups[0] if len(self.optimizer.param_groups) > 0 else {}
        eps = pg0.get("eps", 1e-8)
        beta1, beta2 = pg0.get("betas", (0.9, 0.999))
        grads_tensors = self._unflatten_like(grads)


        for p, g in zip(self.paras, grads_tensors):
            dev = p.device
            st = self.optimizer.state.get(p, None)
            if st is not None and "exp_avg" in st and "exp_avg_sq" in st and st.get("step", 0) > 0:
                m = st["exp_avg"]
                v = st["exp_avg_sq"]
                t = st["step"]

                g = torch.zeros_like(p) if g is None else g
                m_t = beta1 * m + (1 - beta1) * g
                v_t = beta2 * v + (1 - beta2) * (g * g)

                # Calculate bias-corrected first and second moment estimates
                if use_bias_correction:
                    m_hat = m_t / (1.0 - (beta1 ** t))
                    v_hat = v_t / (1.0 - (beta2 ** t))
                else:
                    m_hat = m_t
                    v_hat = v_t

                # Compute the update direction
                update = - m_hat / (v_hat.sqrt() + eps)

                dirs.append(update.view(-1))
            else:
                if fallback_to_neg_grad:
                    update = - g / (g.abs() + eps)
                    dirs.append(update.view(-1))
                else:
                    dirs.append(torch.zeros_like(p).reshape(-1))



        return torch.cat(dirs).detach().cpu().numpy()

    
    # def restore(self, xk_t, bn_states, was_training):
    #     _vector_to_params_(self.model, xk_t)
    #     for m, rm, rv in bn_states:
    #             m.running_mean.copy_(rm)
    #             m.running_var.copy_(rv)
    #     if was_training:
    #             self.model.train()
    


    def step(self, loss_fn, direction=None, condition="strong_wolfe", c1=0.6, c2=0.9, factor=0.5, amax=50.0, step=0, interval=100):
        """
        condition: Line Search condition. Option: armijo, strong_wolfe
        search_mode: Option: backtracking, forward, interpolate
        factor: used for searching. (growing/shrinking LR)
        c1: parameter for armijo rule
        c2: parameter for wolfe-condition
        interval: perform line search every {interval} steps.
        """
        assert_finite_params(self.model)

        if step % interval != 0:
            if self.injection:
                self.original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]  # Save original learning rates
                injection_factor = random.choice(self.injection_distribution)  # Sample from the predefined distribution
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.prev_alpha * injection_factor
                print(param_group['lr'])
                return
            else:
                return


        xk_t = _params_to_vector(self.model, self.paras).detach()
        xk = xk_t.cpu().numpy()
        # was_training = self.model.training
        # bn_states = []
        # for m in self.model.modules():
        #     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        #         bn_states.append((m, m.running_mean.clone(), m.running_var.clone()))
        # self.model.eval() 




        def f_at_x(x_np):
            x_t = torch.from_numpy(x_np).to(xk_t.device, dtype=xk_t.dtype)
            _vector_to_params_(self.model, x_t, paras=self.paras)

            # for pg in self.optimizer.param_groups:
            #     wd = pg.get("weight_decay", 0.0)
            #     if wd != 0:
            #         for p in pg["params"]:
            #             if p.requires_grad:
            #                 p.data.add_(p.data, alpha=-pg["lr"] * wd)
            with torch.no_grad():
                val = loss_fn()
            _vector_to_params_(self.model, xk_t, paras=self.paras)
            return val

        def fprime_at(x_np):
            x_t = torch.from_numpy(x_np).to(xk_t.device, dtype=xk_t.dtype)
            _vector_to_params_(self.model, x_t, paras=self.paras)
            # for pg in self.optimizer.param_groups:
            #     wd = pg.get("weight_decay", 0.0)
            #     if wd != 0:
            #         for p in pg["params"]:
            #             if p.requires_grad:
            #                 p.data.add_(p.data, alpha=-pg["lr"] * wd)
            loss = loss_fn()
            grads = get_grad_list(self.paras)
            flat = []
            for p, g in zip(self.paras, grads):
                if g is None:
                    flat.append(torch.zeros_like(p).reshape(-1))
                else:
                    flat.append(g.reshape(-1))
            g = torch.cat(flat).detach().cpu().numpy()
            _vector_to_params_(self.model, xk_t, paras=self.paras) 
            return loss, g
        
        f0, gk = fprime_at(xk)  
        if direction is None:
            if self.optimizer_type == "Adam":
                pk_np = self._adam_direction_from_state(grads=gk, fallback_to_neg_grad=True, use_bias_correction=True)
            elif self.optimizer_type == "SGD_momentum":
                pk_np = self._sgd_momentum_direction_from_state(grads=gk, fallback_to_neg_grad=True)
            elif self.optimizer_type == "SGD":
                pk_np = -gk
        


        if float(np.dot(gk, pk_np)) >= 0:
            print("Ascent")
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.optimizer.state[p]
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['step'] = torch.tensor(0., device=p.device, dtype=p.dtype)
                    else:
                        if 'exp_avg' in state:
                            state['exp_avg'].zero_()
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'].zero_()
                        state['step'] = torch.tensor(0., device=p.device, dtype=p.dtype)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 1e-6
            return  



        # self.restore(xk_t, bn_states, was_training)

        if condition == "strong_wolfe":
            old_fval = self.prev_fvals[-1] if len(self.prev_fvals) >= 1 else None
            old_old_fval = self.prev_fvals[-2] if len(self.prev_fvals) >= 2 else None
            alpha, fc, gc, phi_star, old_fval, derphi_star = line_search_wolfe2(
                f=f_at_x,
                myfprime=fprime_at, 
                xk=xk,
                pk=pk_np,
                gfk=gk,
                old_fval=old_fval,
                old_old_fval=old_old_fval,
                args=(),
                c1=c1,
                c2=c2,
                amax=float(amax),
            )

        elif condition == "armijo":
            alpha, fc, _ = line_search_armijo(
                    f=f_at_x,
                    xk=xk,
                    pk=pk_np,
                    gfk=gk,
                    old_fval=f0,
                    args=(),
                    c1=c1,
                    alpha0=self.prev_alpha,
                    num_search=self.num_search,
                    step=step,
                    search_mode=self.search_mode,
                    factor=factor
                )

        if alpha is None or not np.isfinite(alpha) or alpha <= 0:
            current_lr = self.optimizer.param_groups[0]["lr"]
            alpha = float(current_lr if np.isfinite(current_lr) and current_lr > 0 else self.start_lr)

        self.prev_fvals.append(f0)

        # self.restore(xk_t, bn_states, was_training)
        
        print(f"[LineSearchScheduler] alpha={alpha:.6g}, fc={fc}")
        
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = alpha


        self.prev_alpha = alpha





            
def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=0.8, c2=0.9, amax=50):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient (can be None).
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.

    Returns
    -------
    alpha0 : float
        Alpha for which ``x_new = x0 + alpha * pk``.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """
    fc = [0]
    gc = [0]
    gval = [None]

    def phi(alpha):
        fc[0] += 1
        return f(xk + alpha * pk, *args)
    
    

    if isinstance(myfprime, tuple):
        def derphi(alpha):
            fc[0] += len(xk)+1
            eps = myfprime[1]
            fprime = myfprime[0]
            newargs = (f,eps) + args
            gval[0] = fprime(xk+alpha*pk, *newargs)  # store for later use
 
            return np.dot(gval[0], pk)
    else:
        fprime = myfprime
        def derphi(alpha):
            gc[0] += 1
            gval[0] = fprime(xk+alpha*pk)[1]  # store for later use
            # print(xk, alpha, pk )
            # print("VAL", fprime(xk+alpha*pk))
            return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    alpha_star, phi_star, old_fval, derphi_star = \
                scalar_search_wolfe2(phi, derphi, old_fval, old_old_fval,
                                     derphi0, c1, c2, amax)

    if derphi_star is not None:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star

def assert_finite_params(model):
    for n,p in model.named_parameters():
        if not torch.isfinite(p).all():
            print(f"[BAD PARAM AT LOAD] {n} has non-finite values ({p.dtype})")
            raise RuntimeError("Non-finite parameter at start")

def scalar_search_wolfe2(phi, derphi=None, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=50):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable f(x,*args)
        Objective scalar function.

    derphi : callable f'(x,*args), optional
        Objective function derivative (can be None)
    phi0 : float, optional
        Value of phi at s=0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value of derphi at s=0
    args : tuple
        Additional arguments passed to objective function.
    c1 : float
        Parameter for Armijo condition rule.
    c2 : float
        Parameter for curvature condition rule.

    Returns
    -------
    alpha_star : float
        Best alpha
    phi_star
        phi at alpha_star
    phi0
        phi at 0
    derphi_star
        derphi at alpha_star

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].
    """

    if phi0 is None:
        phi0 = phi(0.)

    if derphi0 is None and derphi is not None:
        derphi0 = derphi(0.)

    alpha0 = 0
    if old_phi0 is not None:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    if alpha1 == 0:
        # This shouldn't happen. Perhaps the increment has slipped below
        # machine precision?  For now, set the return variables skip the
        # useless while loop, and raise warnflag=2 due to possible imprecision.
        alpha_star = None
        phi_star = phi0
        phi0 = old_phi0
        derphi_star = None

    phi_a1 = phi(alpha1)
    #derphi_a1 = derphi(alpha1)  evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0



    i = 1
    maxiter = 10
    for i in range(maxiter):
        if alpha1 == 0:
            break
        if (phi_a1 > phi0 + c1*alpha1*derphi0) or \
           ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2)
            break

        alpha2 = 2 * alpha1   # increase by factor of two on each iteration
        i = i + 1
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None

    return alpha_star, phi_star, phi0, derphi_star




def _cubicmin(a,fa,fpa,b,fb,c,fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found return None

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    C = fpa
    D = fa
    db = b-a
    dc = c-a
    if (db == 0) or (dc == 0) or (b==c): return None
    denom = (db*dc)**2 * (db-dc)
    d1 = np.empty((2,2))
    d1[0,0] = dc**2
    d1[0,1] = -db**2
    d1[1,0] = -dc**3
    d1[1,1] = db**3
    [A,B] = np.dot(d1, np.asarray([fb-fa-C*db,fc-fa-C*dc]).flatten())
    A /= denom
    B /= denom
    radical = B*B-3*A*C
    if radical < 0:  return None
    if (A == 0): return None
    xmin = a + (-B + np.sqrt(radical))/(3*A)
    return xmin


def _quadmin(a,fa,fpa,b,fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    D = fa
    C = fpa
    db = b-a*1.0
    if (db==0): return None
    B = (fb-D-C*db)/(db*db)
    if (B <= 0): return None
    xmin = a  - C / (2.0*B)
    return xmin

def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2):
    """
    Part of the optimization algorithm in `scalar_search_wolfe2`.
    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while 1:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here.  Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi-a_lo;
        if dalpha < 0: a,b = a_hi,a_lo
        else: a,b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval) then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is stil too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1*dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if (i==0) or (a_j is None) or (a_j > b-cchk) or (a_j < a+cchk):
            qchk = delta2*dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2*derphi0:
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            a_star = a_j
            val_star = phi_aj
            valprime_star = None
            break
    return a_star, val_star, valprime_star




def _normalize_params(model, paras: Optional[Iterable]=None) -> List[torch.nn.Parameter]:
    if paras is None:
        return [p for p in model.parameters() if p.requires_grad]

    normed = []
    for item in paras:
        if isinstance(item, tuple):
            p = item[1]
        else:
            p = item
        if not isinstance(p, torch.nn.Parameter):
            raise TypeError(f"Expected torch.nn.Parameter, got {type(p)}")
        normed.append(p)
    return normed


def _params_to_vector(model, paras: Optional[Iterable]=None) -> torch.Tensor:
    params = _normalize_params(model, paras)
    if not params:
        first = next(model.parameters(), None)
        device = first.device if first is not None else torch.device("cpu")
        return torch.empty(0, device=device)

    with torch.no_grad():
        flats = [p.detach().reshape(-1) for p in params]
        return torch.cat(flats)


def _vector_to_params_(model, vec: torch.Tensor, paras: Optional[Iterable]=None) -> None:
    params = _normalize_params(model, paras)
    total = sum(p.numel() for p in params)
    if vec.numel() != total:
        raise ValueError(f"Size mismatch: vec has {vec.numel()} elements, "
                         f"but params require {total}.")

    with torch.no_grad():
        offset = 0
        for p in params:
            numel = p.numel()
            chunk = vec[offset:offset + numel].view_as(p)
            if chunk.device != p.device or chunk.dtype != p.dtype:
                chunk = chunk.to(device=p.device, dtype=p.dtype)
            p.copy_(chunk)
            offset += numel


def get_grad_list(params):
    return [p.grad for p in params]


def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1, num_search=16, step=0, search_mode="backtrack", factor=0.5):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Parameters
    ----------
    f : callable
        Function to be minimized.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    gfk : array_like
        Gradient of `f` at point `xk`.
    old_fval : float
        Value of `f` at point `xk`.
    args : tuple, optional
        Optional arguments.
    c1 : float, optional
        Value to control stopping criterion.
    alpha0 : scalar, optional
        Value of `alpha` at start of the optimization.

    Returns
    -------
    alpha
    f_count
    f_val_at_alpha

    Notes
    -----
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    """
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        value = f(xk + alpha1*pk, *args)
        return value

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    derphi0 = np.dot(gfk, pk)
    
    if search_mode == "backtrack":
        # alpha, phi1 = search_backtracking_visual(phi, phi0, derphi0, c1=c1,
        #                                 alpha=alpha0, shrink=factor, plot_path=f"backtracking_{step}.png")
        alpha, phi1 = search_backtracking(phi, phi0, derphi0, c1=c1,
                                        alpha=alpha0, shrink=factor, num_search=num_search)
    elif search_mode == "forward":
        alpha, phi1 = search_forward(phi, phi0, derphi0, c1=c1,
                                           alpha=alpha0, grow=factor, shrink=factor/2, amax=1, num_search=num_search)
    elif search_mode == "bisection":
         alpha, phi1 = search_bisection(phi, phi0, derphi0, c1=c1,
                                           old_alpha=alpha0, grow=factor, shrink=factor/2, amax=1, num_search=num_search)
    else:
        alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
                                        alpha0=alpha0)
    return alpha, fc[0], phi1


# def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
#     """
#     Compatibility wrapper for `line_search_armijo`
#     """
#     r = line_search_armijo(f, xk, pk, gfk, old_fval, args=args, c1=c1,
#                            alpha0=alpha0)
#     return r[0], r[1], 0, r[2]


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """Minimize over alpha, the function ``phi(alpha)``.

    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    alpha > 0 is assumed to be a descent direction.

    Returns
    -------
    alpha
    phi1

    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1


def search_forward(phi, phi0, derphi0, c1, alpha, grow, shrink, amax, num_search):

    # Try expanding
    phi_a = phi(alpha)
    count = 0
    while phi_a <= phi0 + c1 * alpha * derphi0 and alpha < amax and count < num_search:
        alpha *= grow
        phi_a = phi(alpha)
        count += 1

    # Overshoot → shrink until good
    while phi_a > phi0 + c1 * alpha * derphi0:
        alpha *= shrink
        phi_a = phi(alpha)

    return alpha, phi_a

def search_bisection(phi, phi0, derphi0, c1,
                     old_alpha, grow=2.0, shrink=0.5,
                     amax=1, num_search=10):

    alpha = old_alpha
    phi_a = phi(alpha)

    armijo_old = phi_a <= phi0 + c1 * alpha * derphi0
    print(armijo_old)

    if armijo_old:
        for _ in range(num_search): 
            print(_)
        
            new_alpha = alpha / grow
            if new_alpha >= amax:
                break

            new_phi = phi(new_alpha)


            if new_phi > phi0 + c1 * new_alpha * derphi0:
                break

    
            alpha = new_alpha
            phi_a = new_phi

        return alpha, phi_a


    else:
        for _ in range(num_search): 
            print(_)
   
            new_alpha = alpha * grow
            new_phi = phi(new_alpha)

        
            if new_phi <= phi0 + c1 * new_alpha * derphi0:
                return new_alpha, new_phi

            alpha = new_alpha
            phi_a = new_phi

        return alpha, phi_a

def search_backtracking(phi, phi0, derphi0, c1, alpha, shrink, num_search):
    phi_a = phi(alpha)
    count = 0
    while phi_a > phi0 + c1 * alpha * derphi0 and count < num_search:
        alpha *= shrink
        phi_a = phi(alpha)
        count += 1
    return alpha, phi_a




def search_backtracking_visual(
    phi, phi0, derphi0,
    c1, alpha, shrink,
    plot_path="backtracking_ls.png",
    t_min=0.0, t_max=1.0, num_points=20
):
    explored = []  

    # --- Backtracking loop ---
    phi_a = phi(alpha)
    explored.append((alpha, phi_a))

    while phi_a > phi0 + c1 * alpha * derphi0:
        alpha *= shrink
        phi_a = phi(alpha)
        explored.append((alpha, phi_a))

    chosen_alpha, chosen_phi = alpha, phi_a


    t_vals = np.linspace(t_min, t_max, num_points)
    phi_vals_list = []
    for t in t_vals:
        value = phi(t)   
        phi_vals_list.append(value)
    phi_vals = np.array(phi_vals_list)


    armijo_line = phi0 + c1 * t_vals * derphi0


    plt.figure(figsize=(8, 6))


    plt.plot(t_vals, phi_vals, label="phi(t)", linewidth=2)


    plt.plot(t_vals, armijo_line, "--", label="Armijo line", linewidth=2)


    for i, (a, v) in enumerate(explored):
        plt.scatter(a, v, color="red", s=60)
        if i == 0:
            plt.annotate("init", (a, v), textcoords="offset points", xytext=(5, 5))
        else:
            plt.annotate(f"bt {i}", (a, v), textcoords="offset points", xytext=(5, 5))

    
    plt.scatter(chosen_alpha, chosen_phi, color="blue", s=120, marker="x", label="chosen alpha")

    # labels
    plt.xlabel("t (step size)")
    plt.ylabel("phi(t)")
    plt.title("Backtracking Line Search Visualization")
    plt.grid(True)
    plt.legend()

    # Save
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return chosen_alpha, chosen_phi, 