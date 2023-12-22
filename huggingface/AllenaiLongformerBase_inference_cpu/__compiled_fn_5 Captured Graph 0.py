from __future__ import annotations



def forward(self, L_attention_mask_ : torch.Tensor):
    l_attention_mask_ = L_attention_mask_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1287, code: is_index_masked = attention_mask < 0
    is_index_masked = l_attention_mask_ < 0
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1288, code: is_index_global_attn = attention_mask > 0
    is_index_global_attn = l_attention_mask_ > 0;  l_attention_mask_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1291, code: is_global_attn = is_index_global_attn.flatten().any().item()
    flatten = is_index_global_attn.flatten()
    any_1 = flatten.any();  flatten = None
    return (any_1, is_index_masked, is_index_global_attn)
    