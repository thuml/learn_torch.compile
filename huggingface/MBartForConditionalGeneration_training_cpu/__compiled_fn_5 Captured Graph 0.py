from __future__ import annotations



def forward(self, L_labels_ : torch.Tensor):
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:66, code: prev_output_tokens = input_ids.clone()
    decoder_input_ids = l_labels_.clone();  l_labels_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:71, code: prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    eq = decoder_input_ids == -100
    masked_fill_ = decoder_input_ids.masked_fill_(eq, 1);  eq = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:73, code: index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    ne = decoder_input_ids.ne(1)
    sum_1 = ne.sum(dim = 1);  ne = None
    sub = sum_1 - 1;  sum_1 = None
    index_of_eos = sub.unsqueeze(-1);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:74, code: decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    gather = decoder_input_ids.gather(1, index_of_eos);  index_of_eos = None
    decoder_start_tokens = gather.squeeze();  gather = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:75, code: prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    getitem = decoder_input_ids[(slice(None, None, None), slice(None, -1, None))]
    clone_1 = getitem.clone();  getitem = None
    decoder_input_ids[(slice(None, None, None), slice(1, None, None))] = clone_1;  setitem = decoder_input_ids;  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:76, code: prev_output_tokens[:, 0] = decoder_start_tokens
    decoder_input_ids[(slice(None, None, None), 0)] = decoder_start_tokens;  setitem_1 = decoder_input_ids;  decoder_start_tokens = None
    return (decoder_input_ids,)
    