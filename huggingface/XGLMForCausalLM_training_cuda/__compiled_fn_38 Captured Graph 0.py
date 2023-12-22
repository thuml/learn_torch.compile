from __future__ import annotations



def forward(self, L_stack0_last_hidden_state : torch.Tensor, L_labels_ : torch.Tensor):
    l_stack0_last_hidden_state = L_stack0_last_hidden_state
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:828, code: logits = self.lm_head(outputs[0])
    logits = self.L__self___lm_head(l_stack0_last_hidden_state);  l_stack0_last_hidden_state = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:833, code: shift_labels = labels.new_zeros(labels.shape)
    shift_labels = l_labels_.new_zeros((1, 128))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:834, code: shift_labels[:, :-1] = labels[:, 1:].clone()
    getitem = l_labels_[(slice(None, None, None), slice(1, None, None))];  l_labels_ = None
    clone = getitem.clone();  getitem = None
    shift_labels[(slice(None, None, None), slice(None, -1, None))] = clone;  setitem = shift_labels;  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:835, code: shift_labels[:, -1] = self.config.pad_token_id
    shift_labels[(slice(None, None, None), -1)] = 1;  setitem_1 = shift_labels
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:838, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view = logits.view(-1, 256008)
    view_1 = shift_labels.view(-1);  shift_labels = None
    loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (loss, logits)
    