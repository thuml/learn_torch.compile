from __future__ import annotations



def forward(self, arg0_1: "f32[1026, 256]", arg1_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:144, code: mask = input_ids.ne(padding_idx).int()
    ne: "b8[1, 128]" = torch.ops.aten.ne.Scalar(arg1_1, 1);  arg1_1 = None
    convert_element_type: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:145, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum: "i64[1, 128]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
    convert_element_type_1: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
    add: "i32[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
    mul: "i32[1, 128]" = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:146, code: return incremental_indices.long() + padding_idx
    convert_element_type_2: "i64[1, 128]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
    add_1: "i64[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:130, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
    view: "i64[128]" = torch.ops.aten.reshape.default(add_1, [-1]);  add_1 = None
    index: "f32[128, 256]" = torch.ops.aten.index.Tensor(arg0_1, [view]);  arg0_1 = view = None
    view_1: "f32[1, 128, 256]" = torch.ops.aten.reshape.default(index, [1, 128, -1]);  index = None
    alias: "f32[1, 128, 256]" = torch.ops.aten.alias.default(view_1);  view_1 = None
    alias_1: "f32[1, 128, 256]" = torch.ops.aten.alias.default(alias);  alias = None
    return (alias_1,)
    