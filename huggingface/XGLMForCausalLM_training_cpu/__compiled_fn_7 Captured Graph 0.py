from __future__ import annotations



def forward(self, L_position_ids_ : torch.Tensor):
    l_position_ids_ = L_position_ids_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:205, code: position_ids += self.offset
    l_position_ids_ += 2;  position_ids = l_position_ids_;  l_position_ids_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:209, code: if max_pos > self.weights.size(0):
    l__self___weights = self.L__self___weights
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:212, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view = position_ids.view(-1);  position_ids = None
    index_select = l__self___weights.index_select(0, view);  l__self___weights = view = None
    view_1 = index_select.view(1, 128, 1024);  index_select = None
    detach = view_1.detach();  view_1 = None
    return (detach,)
    