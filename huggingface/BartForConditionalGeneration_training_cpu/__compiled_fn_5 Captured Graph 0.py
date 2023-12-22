from __future__ import annotations



def forward(self, L_labels_ : torch.Tensor):
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:77, code: shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    decoder_input_ids = l_labels_.new_zeros((1, 1024))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:78, code: shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    getitem = l_labels_[(slice(None, None, None), slice(None, -1, None))];  l_labels_ = None
    clone = getitem.clone();  getitem = None
    decoder_input_ids[(slice(None, None, None), slice(1, None, None))] = clone;  setitem = decoder_input_ids;  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:79, code: shifted_input_ids[:, 0] = decoder_start_token_id
    decoder_input_ids[(slice(None, None, None), 0)] = 2;  setitem_1 = decoder_input_ids
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:84, code: shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    eq = decoder_input_ids == -100
    masked_fill_ = decoder_input_ids.masked_fill_(eq, 1);  eq = None
    return (decoder_input_ids,)
    