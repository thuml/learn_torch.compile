from __future__ import annotations



def forward(self, L_attention_mask_ : torch.Tensor):
    l_attention_mask_ = L_attention_mask_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:74, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask = torch.full((2048, 2048), -3.4028234663852886e+38, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:75, code: mask_cond = torch.arange(mask.size(-1), device=device)
    mask_cond = torch.arange(2048, device = device(type='cuda', index=0))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:76, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add = mask_cond + 1
    view = add.view(2048, 1);  add = None
    lt = mask_cond < view;  mask_cond = view = None
    masked_fill_ = mask.masked_fill_(lt, 0);  lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:77, code: mask = mask.to(dtype)
    mask_1 = mask.to(torch.float32);  mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:81, code: return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))];  mask_1 = None
    combined_attention_mask = getitem.expand(1, 1, 2048, 2048);  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:91, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    getitem_1 = l_attention_mask_[(slice(None, None, None), None, None, slice(None, None, None))];  l_attention_mask_ = None
    expand_1 = getitem_1.expand(1, 1, 2048, 2048);  getitem_1 = None
    expanded_mask = expand_1.to(torch.float32);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:93, code: inverted_mask = 1.0 - expanded_mask
    inverted_mask = 1.0 - expanded_mask;  expanded_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:95, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    to_2 = inverted_mask.to(torch.bool)
    masked_fill = inverted_mask.masked_fill(to_2, -3.4028234663852886e+38);  inverted_mask = to_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:547, code: expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
    expanded_attn_mask = masked_fill.to(device(type='cuda', index=0));  masked_fill = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:551, code: expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    combined_attention_mask_1 = expanded_attn_mask + combined_attention_mask;  expanded_attn_mask = combined_attention_mask = None
    return (combined_attention_mask_1,)
    