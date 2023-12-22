from __future__ import annotations



def forward(self, L_input_ids_ : torch.Tensor):
    input_ids = L_input_ids_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1720, code: attention_mask = torch.ones(input_shape, device=device)
    attention_mask = torch.ones((1, 1024), device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1722, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    token_type_ids = torch.zeros((1, 1024), dtype = torch.int64, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))];  attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:927, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask_1 = extended_attention_mask.to(dtype = torch.float32);  extended_attention_mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub = 1.0 - extended_attention_mask_1;  extended_attention_mask_1 = None
    extended_attention_mask_2 = sub * -3.4028234663852886e+38;  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1739, code: extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
    extended_attention_mask_3 = extended_attention_mask_2[(slice(None, None, None), 0, 0, slice(None, None, None))];  extended_attention_mask_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:428, code: mask = input_ids.ne(padding_idx).int()
    ne = input_ids.ne(1)
    mask = ne.int();  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:429, code: incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    cumsum = torch.cumsum(mask, dim = 1)
    type_as = cumsum.type_as(mask);  cumsum = None
    incremental_indices = type_as * mask;  type_as = mask = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:430, code: return incremental_indices.long() + padding_idx
    long = incremental_indices.long();  incremental_indices = None
    add = long + 1;  long = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:457, code: position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
    position_ids = add.to(device(type='cpu'));  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:470, code: inputs_embeds = self.word_embeddings(input_ids)
    inputs_embeds = self.L__self___embeddings_word_embeddings(input_ids);  input_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:471, code: position_embeddings = self.position_embeddings(position_ids)
    position_embeddings = self.L__self___embeddings_position_embeddings(position_ids);  position_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:472, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    token_type_embeddings = self.L__self___embeddings_token_type_embeddings(token_type_ids);  token_type_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:474, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add_1 = inputs_embeds + position_embeddings;  inputs_embeds = position_embeddings = None
    embeddings = add_1 + token_type_embeddings;  add_1 = token_type_embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:475, code: embeddings = self.LayerNorm(embeddings)
    embeddings_1 = self.L__self___embeddings_LayerNorm(embeddings);  embeddings = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:476, code: embeddings = self.dropout(embeddings)
    embedding_output = self.L__self___embeddings_dropout(embeddings_1);  embeddings_1 = None
    return (embedding_output, extended_attention_mask_3)
    