from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1026, 768]"; tangents_1: "f32[1, 1024, 768]"; 

    primals_1, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:129, code: positions = torch.arange(
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:131, code: ).expand(bsz, -1)
    expand: "i64[1, 1024]" = torch.ops.aten.expand.default(iota, [1, -1]);  iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:133, code: return super().forward(positions + self.offset)
    add: "i64[1, 1024]" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_1, add);  primals_1 = None
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(add, -1)
    unsqueeze: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze, scalar_tensor, tangents_1);  unsqueeze = scalar_tensor = tangents_1 = None
    full: "f32[1026, 768]" = torch.ops.aten.full.default([1026, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1026, 768]" = torch.ops.aten._unsafe_index_put.default(full, [add], where, True);  full = add = where = None
    return pytree.tree_unflatten([embedding, _unsafe_index_put], self._out_spec)
    