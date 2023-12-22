from __future__ import annotations



def forward(self, arg0_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:65, code: prev_output_tokens = input_ids.clone()
    clone: "i64[1, 1024]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:70, code: prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(clone, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[1, 1024]" = torch.ops.aten.where.self(eq, scalar_tensor, clone);  eq = scalar_tensor = clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:72, code: index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    ne: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(where, 1)
    sum_1: "i64[1]" = torch.ops.aten.sum.dim_IntList(ne, [1]);  ne = None
    sub: "i64[1]" = torch.ops.aten.sub.Tensor(sum_1, 1);  sum_1 = None
    unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(sub, -1);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:73, code: decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    gather: "i64[1, 1]" = torch.ops.aten.gather.default(where, 1, unsqueeze);  unsqueeze = None
    
    # No stacktrace found for following nodes
    squeeze: "i64[]" = torch.ops.aten.squeeze.default(gather);  gather = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:74, code: prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    slice_3: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(where, 0, 0, 9223372036854775807)
    slice_4: "i64[1, 1023]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, -1);  slice_3 = None
    clone_1: "i64[1, 1023]" = torch.ops.aten.clone.default(slice_4);  slice_4 = None
    slice_7: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(where, 0, 0, 9223372036854775807)
    slice_8: "i64[1, 1023]" = torch.ops.aten.slice.Tensor(slice_7, 1, 1, 9223372036854775807);  slice_7 = None
    copy: "i64[1, 1023]" = torch.ops.aten.copy.default(slice_8, clone_1);  slice_8 = clone_1 = None
    slice_9: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(where, 0, 0, 9223372036854775807)
    slice_scatter: "i64[1, 1024]" = torch.ops.aten.slice_scatter.default(slice_9, copy, 1, 1, 9223372036854775807);  slice_9 = copy = None
    slice_scatter_1: "i64[1, 1024]" = torch.ops.aten.slice_scatter.default(where, slice_scatter, 0, 0, 9223372036854775807);  where = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:75, code: prev_output_tokens[:, 0] = decoder_start_tokens
    slice_13: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 9223372036854775807)
    select_1: "i64[1]" = torch.ops.aten.select.int(slice_13, 1, 0);  slice_13 = None
    copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, squeeze);  select_1 = squeeze = None
    slice_14: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 9223372036854775807)
    select_scatter: "i64[1, 1024]" = torch.ops.aten.select_scatter.default(slice_14, copy_1, 1, 0);  slice_14 = copy_1 = None
    slice_scatter_2: "i64[1, 1024]" = torch.ops.aten.slice_scatter.default(slice_scatter_1, select_scatter, 0, 0, 9223372036854775807);  slice_scatter_1 = select_scatter = None
    return (slice_scatter_2,)
    