from __future__ import annotations



def forward(self, arg0_1: "f32[2050, 1024]", arg1_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:205, code: position_ids += self.offset
    add: "i64[1, 128]" = torch.ops.aten.add.Tensor(arg1_1, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:212, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view_1: "i64[128]" = torch.ops.aten.view.default(add, [-1])
    index: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg0_1, [view_1]);  arg0_1 = view_1 = None
    view_2: "f32[1, 128, 1024]" = torch.ops.aten.view.default(index, [1, 128, 1024]);  index = None
    alias: "f32[1, 128, 1024]" = torch.ops.aten.alias.default(view_2);  view_2 = None
    alias_1: "f32[1, 128, 1024]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # No stacktrace found for following nodes
    copy_: "i64[1, 128]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = None
    return (alias_1,)
    