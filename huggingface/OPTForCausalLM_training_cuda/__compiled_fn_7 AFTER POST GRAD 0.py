from __future__ import annotations



def forward(self, primals_1: "f32[2050, 768]", primals_2: "f32[1, 2048]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:111, code: attention_mask = attention_mask.long()
    convert_element_type: "i64[1, 2048]" = torch.ops.prims.convert_element_type.default(primals_2, torch.int64);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:114, code: positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
    cumsum: "i64[1, 2048]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
    mul: "i64[1, 2048]" = torch.ops.aten.mul.Tensor(cumsum, convert_element_type);  cumsum = convert_element_type = None
    sub: "i64[1, 2048]" = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:119, code: return super().forward(positions + self.offset)
    add: "i64[1, 2048]" = torch.ops.aten.add.Tensor(sub, 2);  sub = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding: "f32[1, 2048, 768]" = torch.ops.aten.embedding.default(primals_1, add);  primals_1 = None
    return [embedding, add]
    