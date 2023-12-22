from __future__ import annotations



def forward(self):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:123, code: past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    l__self___weight = self.L__self___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:122, code: positions = torch.arange(
    positions = torch.arange(0, 128, dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding = torch.nn.functional.embedding(positions, l__self___weight, None, None, 2.0, False, False);  positions = l__self___weight = None
    return (embedding,)
    