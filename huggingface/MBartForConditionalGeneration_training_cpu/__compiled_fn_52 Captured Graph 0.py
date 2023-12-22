from __future__ import annotations



def forward(self, L_stack0_last_hidden_state : torch.Tensor, L_labels_ : torch.Tensor):
    l_stack0_last_hidden_state = L_stack0_last_hidden_state
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1374, code: lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    l__self___lm_head = self.L__self___lm_head(l_stack0_last_hidden_state);  l_stack0_last_hidden_state = None
    l__self___final_logits_bias = self.L__self___final_logits_bias
    lm_logits = l__self___lm_head + l__self___final_logits_bias;  l__self___lm_head = l__self___final_logits_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1379, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view = lm_logits.view(-1, 50265)
    view_1 = l_labels_.view(-1);  l_labels_ = None
    masked_lm_loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (masked_lm_loss, lm_logits)
    