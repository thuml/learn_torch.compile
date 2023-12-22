from __future__ import annotations



def forward(self, L_input_ids_ : torch.Tensor):
    l_input_ids_ = L_input_ids_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:144, code: mask = input_ids.ne(padding_idx).int()
    ne = l_input_ids_.ne(1);  l_input_ids_ = None
    mask = ne.int();  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:145, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum = torch.cumsum(mask, dim = 1)
    type_as = cumsum.type_as(mask);  cumsum = None
    add = type_as + 0;  type_as = None
    incremental_indices = add * mask;  add = mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:146, code: return incremental_indices.long() + padding_idx
    long = incremental_indices.long();  incremental_indices = None
    add_1 = long + 1;  long = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:121, code: position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
    position_ids = add_1.to(device(type='cuda', index=0));  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:127, code: if max_pos > self.weights.size(0):
    l__self___weights = self.L__self___weights
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:130, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
    view = position_ids.view(-1);  position_ids = None
    index_select = l__self___weights.index_select(0, view);  l__self___weights = view = None
    view_1 = index_select.view(1, 128, -1);  index_select = None
    detach = view_1.detach();  view_1 = None
    return (detach,)
    