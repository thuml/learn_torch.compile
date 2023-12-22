from __future__ import annotations



def forward(self, arg0_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:77, code: shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    full: "i64[1, 1024]" = torch.ops.aten.full.default([1, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:78, code: shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    slice_1: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 9223372036854775807);  arg0_1 = None
    slice_2: "i64[1, 1023]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, -1);  slice_1 = None
    clone: "i64[1, 1023]" = torch.ops.aten.clone.default(slice_2);  slice_2 = None
    slice_3: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_4: "i64[1, 1023]" = torch.ops.aten.slice.Tensor(slice_3, 1, 1, 9223372036854775807);  slice_3 = None
    copy: "i64[1, 1023]" = torch.ops.aten.copy.default(slice_4, clone);  slice_4 = clone = None
    slice_5: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_scatter: "i64[1, 1024]" = torch.ops.aten.slice_scatter.default(slice_5, copy, 1, 1, 9223372036854775807);  slice_5 = copy = None
    slice_scatter_1: "i64[1, 1024]" = torch.ops.aten.slice_scatter.default(full, slice_scatter, 0, 0, 9223372036854775807);  full = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:79, code: shifted_input_ids[:, 0] = decoder_start_token_id
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    slice_9: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 9223372036854775807)
    select_1: "i64[1]" = torch.ops.aten.select.int(slice_9, 1, 0);  slice_9 = None
    copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, lift_fresh_copy);  select_1 = lift_fresh_copy = None
    slice_10: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 9223372036854775807)
    select_scatter: "i64[1, 1024]" = torch.ops.aten.select_scatter.default(slice_10, copy_1, 1, 0);  slice_10 = copy_1 = None
    slice_scatter_2: "i64[1, 1024]" = torch.ops.aten.slice_scatter.default(slice_scatter_1, select_scatter, 0, 0, 9223372036854775807);  slice_scatter_1 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:84, code: shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(slice_scatter_2, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where: "i64[1, 1024]" = torch.ops.aten.where.self(eq, scalar_tensor, slice_scatter_2);  eq = scalar_tensor = slice_scatter_2 = None
    return (where,)
    