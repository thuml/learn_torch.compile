from __future__ import annotations



def forward(self, L_stack0_last_hidden_state : torch.Tensor, L_labels_ : torch.Tensor):
    l_stack0_last_hidden_state = L_stack0_last_hidden_state
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:978, code: logits = self.output_projection(outputs[0])
    logits = self.L__self___output_projection(l_stack0_last_hidden_state);  l_stack0_last_hidden_state = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/trocr/modeling_trocr.py:983, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view = logits.view(-1, 50265)
    view_1 = l_labels_.view(-1);  l_labels_ = None
    loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (loss, logits)
    