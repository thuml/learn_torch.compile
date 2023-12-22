from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[2050, 768]"; primals_2: "f32[1, 2048]"; tangents_1: "f32[1, 2048, 768]"; 

    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:111, code: attention_mask = attention_mask.long()
    convert_element_type: "i64[1, 2048]" = torch.ops.prims.convert_element_type.default(primals_2, torch.int64);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:114, code: positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
    cumsum: "i64[1, 2048]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
    mul: "i64[1, 2048]" = torch.ops.aten.mul.Tensor(cumsum, convert_element_type);  cumsum = convert_element_type = None
    sub: "i64[1, 2048]" = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:117, code: positions = positions[:, past_key_values_length:]
    slice_1: "i64[1, 2048]" = torch.ops.aten.slice.Tensor(sub, 0, 0, 9223372036854775807);  sub = None
    slice_2: "i64[1, 2048]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:119, code: return super().forward(positions + self.offset)
    add: "i64[1, 2048]" = torch.ops.aten.add.Tensor(slice_2, 2);  slice_2 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding: "f32[1, 2048, 768]" = torch.ops.aten.embedding.default(primals_1, add);  primals_1 = None
    eq: "b8[1, 2048]" = torch.ops.aten.eq.Scalar(add, -1)
    unsqueeze: "b8[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[1, 2048, 768]" = torch.ops.aten.where.self(unsqueeze, scalar_tensor, tangents_1);  unsqueeze = scalar_tensor = tangents_1 = None
    full: "f32[2050, 768]" = torch.ops.aten.full.default([2050, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[2050, 768]" = torch.ops.aten._unsafe_index_put.default(full, [add], where, True);  full = add = where = None
    return pytree.tree_unflatten([embedding, _unsafe_index_put, None], self._out_spec)
    