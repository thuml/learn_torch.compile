from __future__ import annotations



def forward(self, arg0_1: "f32[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1288, code: is_index_global_attn = attention_mask > 0
    gt: "b8[1, 1024]" = torch.ops.aten.gt.Scalar(arg0_1, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1291, code: is_global_attn = is_index_global_attn.flatten().any().item()
    view: "b8[1024]" = torch.ops.aten.reshape.default(gt, [1024])
    any_1: "b8[]" = torch.ops.aten.any.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1287, code: is_index_masked = attention_mask < 0
    lt: "b8[1, 1024]" = torch.ops.aten.lt.Scalar(arg0_1, 0);  arg0_1 = None
    return (any_1, lt, gt)
    