from __future__ import annotations



def forward(self, primals_1: "f32[128, 2560]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:122, code: positions = torch.arange(
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding: "f32[128, 2560]" = torch.ops.aten.embedding.default(primals_1, iota);  primals_1 = None
    return [embedding, iota]
    