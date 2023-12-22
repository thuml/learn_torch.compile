from __future__ import annotations



def forward(self):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:95, code: past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    l__self___weight = self.L__self___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:94, code: positions = torch.arange(
    arange = torch.arange(0, 256, dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:96, code: ).expand(bsz, -1)
    positions = arange.expand(1, -1);  arange = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:98, code: return super().forward(positions + self.offset)
    add = positions + 2;  positions = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding = torch.nn.functional.embedding(add, l__self___weight, None, None, 2.0, False, False);  add = l__self___weight = None
    return (embedding,)
    