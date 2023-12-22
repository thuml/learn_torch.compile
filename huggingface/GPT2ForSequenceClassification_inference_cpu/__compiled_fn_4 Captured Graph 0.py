from __future__ import annotations



def forward(self, L_pooled_logits_ : torch.Tensor, L_labels_ : torch.Tensor):
    l_pooled_logits_ = L_pooled_logits_
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1477, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
    view = l_pooled_logits_.view(-1, 2);  l_pooled_logits_ = None
    view_1 = l_labels_.view(-1);  l_labels_ = None
    loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (loss,)
    