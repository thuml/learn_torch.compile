from __future__ import annotations



def forward(self, add: "i64[1, 256]", tangents_1: "f32[1, 256, 1024]"):
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    full_default: "b8[1, 256, 1]" = torch.ops.aten.full.default([1, 256, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[1, 256, 1024]" = torch.ops.aten.where.self(full_default, full_default_1, tangents_1);  full_default = full_default_1 = tangents_1 = None
    full_default_2: "f32[514, 1024]" = torch.ops.aten.full.default([514, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[514, 1024]" = torch.ops.aten._unsafe_index_put.default(full_default_2, [add], where, True);  full_default_2 = add = where = None
    return [_unsafe_index_put]
    