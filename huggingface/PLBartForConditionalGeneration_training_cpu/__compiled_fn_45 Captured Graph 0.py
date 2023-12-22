from __future__ import annotations



def forward(self, L_stack0_last_hidden_state : torch.Tensor, L_labels_ : torch.Tensor):
    l_stack0_last_hidden_state = L_stack0_last_hidden_state
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1344, code: lm_logits = self.lm_head(outputs[0])
    lm_logits = self.L__self___lm_head(l_stack0_last_hidden_state);  l_stack0_last_hidden_state = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1345, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
    l__self___final_logits_bias = self.L__self___final_logits_bias
    to = l__self___final_logits_bias.to(device(type='cpu'));  l__self___final_logits_bias = None
    lm_logits_1 = lm_logits + to;  lm_logits = to = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1350, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view = lm_logits_1.view(-1, 50005)
    view_1 = l_labels_.view(-1);  l_labels_ = None
    masked_lm_loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (masked_lm_loss, lm_logits_1)
    