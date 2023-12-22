from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[512, 512]"; tangents_1: "f32[128, 512]"; 

    primals_1, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:119, code: positions = torch.arange(
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding: "f32[128, 512]" = torch.ops.aten.embedding.default(primals_1, iota);  primals_1 = None
    eq: "b8[128]" = torch.ops.aten.eq.Scalar(iota, -1)
    unsqueeze: "b8[128, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[128, 512]" = torch.ops.aten.where.self(unsqueeze, scalar_tensor, tangents_1);  unsqueeze = scalar_tensor = tangents_1 = None
    full: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 512]" = torch.ops.aten._unsafe_index_put.default(full, [iota], where, True);  full = iota = where = None
    return pytree.tree_unflatten([embedding, _unsafe_index_put], self._out_spec)
    