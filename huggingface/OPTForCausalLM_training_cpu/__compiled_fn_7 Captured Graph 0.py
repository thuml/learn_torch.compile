from __future__ import annotations



def forward(self, L_attention_mask_ : torch.Tensor):
    l_attention_mask_ = L_attention_mask_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:111, code: attention_mask = attention_mask.long()
    attention_mask = l_attention_mask_.long();  l_attention_mask_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:114, code: positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
    cumsum = torch.cumsum(attention_mask, dim = 1)
    type_as = cumsum.type_as(attention_mask);  cumsum = None
    mul = type_as * attention_mask;  type_as = attention_mask = None
    long_1 = mul.long();  mul = None
    positions = long_1 - 1;  long_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:117, code: positions = positions[:, past_key_values_length:]
    positions_1 = positions[(slice(None, None, None), slice(0, None, None))];  positions = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:119, code: return super().forward(positions + self.offset)
    add = positions_1 + 2;  positions_1 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:164, code: input, self.weight, self.padding_idx, self.max_norm,
    l__self___weight = self.L__self___weight
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding = torch.nn.functional.embedding(add, l__self___weight, None, None, 2.0, False, False);  add = l__self___weight = None
    return (embedding,)
    