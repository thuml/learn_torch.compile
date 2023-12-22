from __future__ import annotations



def forward(self, add: "i64[1, 2048]", tangents_1: "f32[1, 2048, 768]"):
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    eq: "b8[1, 2048]" = torch.ops.aten.eq.Scalar(add, -1)
    unsqueeze: "b8[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[1, 2048, 768]" = torch.ops.aten.where.self(unsqueeze, full_default, tangents_1);  unsqueeze = full_default = tangents_1 = None
    full_default_1: "f32[2050, 768]" = torch.ops.aten.full.default([2050, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[2050, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_1, [add], where, True);  full_default_1 = add = where = None
    return [_unsafe_index_put, None]
    