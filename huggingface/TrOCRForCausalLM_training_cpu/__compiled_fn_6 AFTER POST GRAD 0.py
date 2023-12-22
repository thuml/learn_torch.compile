from __future__ import annotations



def forward(self, primals_1: "f32[514, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:94, code: positions = torch.arange(
    iota: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:96, code: ).expand(bsz, -1)
    expand: "i64[1, 256]" = torch.ops.aten.expand.default(iota, [1, -1]);  iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:98, code: return super().forward(positions + self.offset)
    add: "i64[1, 256]" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding: "f32[1, 256, 1024]" = torch.ops.aten.embedding.default(primals_1, add);  primals_1 = None
    return [embedding, add]
    