from __future__ import annotations



def forward(self, L_stack0_last_hidden_state : torch.Tensor, L_labels_ : torch.Tensor):
    l_stack0_last_hidden_state = L_stack0_last_hidden_state
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:956, code: logits = self.lm_head(outputs[0]).contiguous()
    l__self___lm_head = self.L__self___lm_head(l_stack0_last_hidden_state);  l_stack0_last_hidden_state = None
    logits = l__self___lm_head.contiguous();  l__self___lm_head = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:961, code: labels = labels.to(logits.device)
    labels = l_labels_.to(device(type='cpu'));  l_labels_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:963, code: shift_logits = logits[..., :-1, :].contiguous()
    getitem = logits[(Ellipsis, slice(None, -1, None), slice(None, None, None))]
    shift_logits = getitem.contiguous();  getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:964, code: shift_labels = labels[..., 1:].contiguous()
    getitem_1 = labels[(Ellipsis, slice(1, None, None))];  labels = None
    shift_labels = getitem_1.contiguous();  getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:967, code: loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view = shift_logits.view(-1, 50272);  shift_logits = None
    view_1 = shift_labels.view(-1);  shift_labels = None
    loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (loss, logits)
    