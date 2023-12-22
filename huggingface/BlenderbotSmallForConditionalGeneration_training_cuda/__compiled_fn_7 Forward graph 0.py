from __future__ import annotations



def forward(self, primals_1: "f32[512, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:119, code: positions = torch.arange(
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding: "f32[128, 512]" = torch.ops.aten.embedding.default(primals_1, iota);  primals_1 = None
    return [embedding, iota]
    