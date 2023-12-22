from __future__ import annotations



def forward(self, L_stack0_ : torch.Tensor):
    l_stack0_ = L_stack0_
    
    # File: /workspace/youkaichao/code/pytorch/benchmarks/dynamo/timm_models.py:325, code: return reduce_to_scalar_loss(pred) / 1000.0
    truediv = l_stack0_ / 1000.0;  l_stack0_ = None
    return (truediv,)
    