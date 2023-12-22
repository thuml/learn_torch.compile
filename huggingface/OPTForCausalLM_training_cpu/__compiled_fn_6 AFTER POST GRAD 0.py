from __future__ import annotations



def forward(self, arg0_1: "f32[1, 2048]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:91, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    unsqueeze_2: "f32[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(arg0_1, 1);  arg0_1 = None
    unsqueeze_3: "f32[1, 1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
    expand_1: "f32[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 1, 2048, 2048]);  unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:93, code: inverted_mask = 1.0 - expanded_mask
    sub: "f32[1, 1, 2048, 2048]" = torch.ops.aten.sub.Tensor(1.0, expand_1);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:95, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    convert_element_type: "b8[1, 1, 2048, 2048]" = torch.ops.prims.convert_element_type.default(sub, torch.bool)
    full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[1, 1, 2048, 2048]" = torch.ops.aten.where.self(convert_element_type, full_default_2, sub);  convert_element_type = full_default_2 = sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:75, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:76, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add: "i64[2048]" = torch.ops.aten.add.Tensor(iota, 1)
    view: "i64[2048, 1]" = torch.ops.aten.reshape.default(add, [2048, 1]);  add = None
    lt: "b8[2048, 2048]" = torch.ops.aten.lt.Tensor(iota, view);  iota = view = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:74, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:76, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where: "f32[2048, 2048]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:551, code: expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    unsqueeze_4: "f32[1, 2048, 2048]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_5: "f32[1, 1, 2048, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
    expand_2: "f32[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(unsqueeze_5, [1, 1, 2048, 2048]);  unsqueeze_5 = None
    add_1: "f32[1, 1, 2048, 2048]" = torch.ops.aten.add.Tensor(where_1, expand_2);  where_1 = expand_2 = None
    return (add_1,)
    