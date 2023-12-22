from __future__ import annotations



def forward(self, L_stack0_last_hidden_state : torch.Tensor, L_labels_ : torch.Tensor):
    sequence_output = L_stack0_last_hidden_state
    l_labels_ = L_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1397, code: x = self.dense(features)
    x = self.L__self___lm_head_dense(sequence_output);  sequence_output = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    x_1 = torch._C._nn.gelu(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1399, code: x = self.layer_norm(x)
    x_2 = self.L__self___lm_head_layer_norm(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1402, code: x = self.decoder(x)
    prediction_scores = self.L__self___lm_head_decoder(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1863, code: labels = labels.to(prediction_scores.device)
    labels = l_labels_.to(device(type='cuda', index=0));  l_labels_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1864, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view = prediction_scores.view(-1, 50265)
    view_1 = labels.view(-1);  labels = None
    masked_lm_loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (masked_lm_loss, prediction_scores)
    