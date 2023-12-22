from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x = self.L__mod___patch_embed_proj(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten = x.flatten(2);  x = None
    x_1 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_3 = self.L__mod___patch_embed_norm(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:362, code: x = x + self.pos_embed
    l__mod___pos_embed = self.L__mod___pos_embed
    x_4 = x_3 + l__mod___pos_embed;  x_3 = l__mod___pos_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:363, code: x = self.pos_drop(x)
    x_5 = self.L__mod___pos_drop(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:364, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    l__mod___cls_token = self.L__mod___cls_token
    cls_tokens = l__mod___cls_token.expand(8, -1, -1);  l__mod___cls_token = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_0_norm1_weight = self.L__mod___blocks_0_norm1_weight
    l__mod___blocks_0_norm1_bias = self.L__mod___blocks_0_norm1_bias
    x_6 = torch.nn.functional.layer_norm(x_5, (768,), l__mod___blocks_0_norm1_weight, l__mod___blocks_0_norm1_bias, 1e-06);  l__mod___blocks_0_norm1_weight = l__mod___blocks_0_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange = torch.arange(14)
    view = arange.view(1, -1);  arange = None
    arange_1 = torch.arange(14)
    view_1 = arange_1.view(-1, 1);  arange_1 = None
    ind = view - view_1;  view = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx = ind.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave = ind.repeat_interleave(14, dim = 0);  ind = None
    indy = repeat_interleave.repeat_interleave(14, dim = 1);  repeat_interleave = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_1 = indx ** 2
    pow_2 = indy ** 2
    indd = pow_1 + pow_2;  pow_1 = pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze = indd.unsqueeze(0);  indd = None
    rel_indices[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze;  setitem = rel_indices;  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_1 = indy.unsqueeze(0);  indy = None
    rel_indices[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_1;  setitem_1 = rel_indices;  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_2 = indx.unsqueeze(0);  indx = None
    rel_indices[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_2;  setitem_2 = rel_indices;  unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to = rel_indices.to(device(type='cpu'));  rel_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_0_attn_qk = self.L__mod___blocks_0_attn_qk(x_6)
    reshape = l__mod___blocks_0_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_0_attn_qk = None
    qk = reshape.permute(2, 0, 3, 1, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q = qk[0]
    k = qk[1];  qk = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score = to.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_0_attn_pos_proj = self.L__mod___blocks_0_attn_pos_proj(pos_score);  pos_score = None
    pos_score_1 = l__mod___blocks_0_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_0_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_1 = k.transpose(-2, -1);  k = None
    matmul = q @ transpose_1;  q = transpose_1 = None
    patch_score = matmul * 0.14433756729740643;  matmul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_1 = patch_score.softmax(dim = -1);  patch_score = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_2 = pos_score_1.softmax(dim = -1);  pos_score_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_0_attn_gating_param = self.L__mod___blocks_0_attn_gating_param
    gating = l__mod___blocks_0_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_0_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid = torch.sigmoid(gating)
    sub_1 = 1.0 - sigmoid;  sigmoid = None
    mul_1 = sub_1 * patch_score_1;  sub_1 = patch_score_1 = None
    sigmoid_1 = torch.sigmoid(gating);  gating = None
    mul_2 = sigmoid_1 * pos_score_2;  sigmoid_1 = pos_score_2 = None
    attn = mul_1 + mul_2;  mul_1 = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_1 = attn.sum(dim = -1)
    unsqueeze_3 = sum_1.unsqueeze(-1);  sum_1 = None
    attn /= unsqueeze_3;  attn_1 = attn;  attn = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_3 = self.L__mod___blocks_0_attn_attn_drop(attn_1);  attn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_0_attn_v = self.L__mod___blocks_0_attn_v(x_6);  x_6 = None
    reshape_1 = l__mod___blocks_0_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_0_attn_v = None
    v = reshape_1.permute(0, 2, 1, 3);  reshape_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_1 = attn_3 @ v;  attn_3 = v = None
    transpose_2 = matmul_1.transpose(1, 2);  matmul_1 = None
    x_7 = transpose_2.reshape(8, 196, 768);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_8 = self.L__mod___blocks_0_attn_proj(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_9 = self.L__mod___blocks_0_attn_proj_drop(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_0_drop_path = self.L__mod___blocks_0_drop_path(x_9);  x_9 = None
    x_10 = x_5 + l__mod___blocks_0_drop_path;  x_5 = l__mod___blocks_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_0_norm2_weight = self.L__mod___blocks_0_norm2_weight
    l__mod___blocks_0_norm2_bias = self.L__mod___blocks_0_norm2_bias
    x_11 = torch.nn.functional.layer_norm(x_10, (768,), l__mod___blocks_0_norm2_weight, l__mod___blocks_0_norm2_bias, 1e-06);  l__mod___blocks_0_norm2_weight = l__mod___blocks_0_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_12 = self.L__mod___blocks_0_mlp_fc1(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_13 = self.L__mod___blocks_0_mlp_act(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_14 = self.L__mod___blocks_0_mlp_drop1(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_15 = self.L__mod___blocks_0_mlp_norm(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_16 = self.L__mod___blocks_0_mlp_fc2(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_17 = self.L__mod___blocks_0_mlp_drop2(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_0_drop_path_1 = self.L__mod___blocks_0_drop_path(x_17);  x_17 = None
    x_19 = x_10 + l__mod___blocks_0_drop_path_1;  x_10 = l__mod___blocks_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_1_norm1_weight = self.L__mod___blocks_1_norm1_weight
    l__mod___blocks_1_norm1_bias = self.L__mod___blocks_1_norm1_bias
    x_20 = torch.nn.functional.layer_norm(x_19, (768,), l__mod___blocks_1_norm1_weight, l__mod___blocks_1_norm1_bias, 1e-06);  l__mod___blocks_1_norm1_weight = l__mod___blocks_1_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_1 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_2 = torch.arange(14)
    view_3 = arange_2.view(1, -1);  arange_2 = None
    arange_3 = torch.arange(14)
    view_4 = arange_3.view(-1, 1);  arange_3 = None
    ind_1 = view_3 - view_4;  view_3 = view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_1 = ind_1.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_2 = ind_1.repeat_interleave(14, dim = 0);  ind_1 = None
    indy_1 = repeat_interleave_2.repeat_interleave(14, dim = 1);  repeat_interleave_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_3 = indx_1 ** 2
    pow_4 = indy_1 ** 2
    indd_1 = pow_3 + pow_4;  pow_3 = pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_4 = indd_1.unsqueeze(0);  indd_1 = None
    rel_indices_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_4;  setitem_3 = rel_indices_1;  unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_5 = indy_1.unsqueeze(0);  indy_1 = None
    rel_indices_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_5;  setitem_4 = rel_indices_1;  unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_6 = indx_1.unsqueeze(0);  indx_1 = None
    rel_indices_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_6;  setitem_5 = rel_indices_1;  unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_1 = rel_indices_1.to(device(type='cpu'));  rel_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_1_attn_qk = self.L__mod___blocks_1_attn_qk(x_20)
    reshape_3 = l__mod___blocks_1_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_1_attn_qk = None
    qk_1 = reshape_3.permute(2, 0, 3, 1, 4);  reshape_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_1 = qk_1[0]
    k_1 = qk_1[1];  qk_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_3 = to_1.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_1_attn_pos_proj = self.L__mod___blocks_1_attn_pos_proj(pos_score_3);  pos_score_3 = None
    pos_score_4 = l__mod___blocks_1_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_1_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_3 = k_1.transpose(-2, -1);  k_1 = None
    matmul_2 = q_1 @ transpose_3;  q_1 = transpose_3 = None
    patch_score_2 = matmul_2 * 0.14433756729740643;  matmul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_3 = patch_score_2.softmax(dim = -1);  patch_score_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_5 = pos_score_4.softmax(dim = -1);  pos_score_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_1_attn_gating_param = self.L__mod___blocks_1_attn_gating_param
    gating_1 = l__mod___blocks_1_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_1_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_2 = torch.sigmoid(gating_1)
    sub_3 = 1.0 - sigmoid_2;  sigmoid_2 = None
    mul_4 = sub_3 * patch_score_3;  sub_3 = patch_score_3 = None
    sigmoid_3 = torch.sigmoid(gating_1);  gating_1 = None
    mul_5 = sigmoid_3 * pos_score_5;  sigmoid_3 = pos_score_5 = None
    attn_4 = mul_4 + mul_5;  mul_4 = mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_2 = attn_4.sum(dim = -1)
    unsqueeze_7 = sum_2.unsqueeze(-1);  sum_2 = None
    attn_4 /= unsqueeze_7;  attn_5 = attn_4;  attn_4 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_7 = self.L__mod___blocks_1_attn_attn_drop(attn_5);  attn_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_attn_v = self.L__mod___blocks_1_attn_v(x_20);  x_20 = None
    reshape_4 = l__mod___blocks_1_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_1_attn_v = None
    v_1 = reshape_4.permute(0, 2, 1, 3);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_3 = attn_7 @ v_1;  attn_7 = v_1 = None
    transpose_4 = matmul_3.transpose(1, 2);  matmul_3 = None
    x_21 = transpose_4.reshape(8, 196, 768);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_22 = self.L__mod___blocks_1_attn_proj(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_23 = self.L__mod___blocks_1_attn_proj_drop(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_1_drop_path = self.L__mod___blocks_1_drop_path(x_23);  x_23 = None
    x_24 = x_19 + l__mod___blocks_1_drop_path;  x_19 = l__mod___blocks_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_1_norm2_weight = self.L__mod___blocks_1_norm2_weight
    l__mod___blocks_1_norm2_bias = self.L__mod___blocks_1_norm2_bias
    x_25 = torch.nn.functional.layer_norm(x_24, (768,), l__mod___blocks_1_norm2_weight, l__mod___blocks_1_norm2_bias, 1e-06);  l__mod___blocks_1_norm2_weight = l__mod___blocks_1_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_26 = self.L__mod___blocks_1_mlp_fc1(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_27 = self.L__mod___blocks_1_mlp_act(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_28 = self.L__mod___blocks_1_mlp_drop1(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_29 = self.L__mod___blocks_1_mlp_norm(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_30 = self.L__mod___blocks_1_mlp_fc2(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_31 = self.L__mod___blocks_1_mlp_drop2(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_1_drop_path_1 = self.L__mod___blocks_1_drop_path(x_31);  x_31 = None
    x_33 = x_24 + l__mod___blocks_1_drop_path_1;  x_24 = l__mod___blocks_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_2_norm1_weight = self.L__mod___blocks_2_norm1_weight
    l__mod___blocks_2_norm1_bias = self.L__mod___blocks_2_norm1_bias
    x_34 = torch.nn.functional.layer_norm(x_33, (768,), l__mod___blocks_2_norm1_weight, l__mod___blocks_2_norm1_bias, 1e-06);  l__mod___blocks_2_norm1_weight = l__mod___blocks_2_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_2 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_4 = torch.arange(14)
    view_6 = arange_4.view(1, -1);  arange_4 = None
    arange_5 = torch.arange(14)
    view_7 = arange_5.view(-1, 1);  arange_5 = None
    ind_2 = view_6 - view_7;  view_6 = view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_2 = ind_2.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_4 = ind_2.repeat_interleave(14, dim = 0);  ind_2 = None
    indy_2 = repeat_interleave_4.repeat_interleave(14, dim = 1);  repeat_interleave_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_5 = indx_2 ** 2
    pow_6 = indy_2 ** 2
    indd_2 = pow_5 + pow_6;  pow_5 = pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_8 = indd_2.unsqueeze(0);  indd_2 = None
    rel_indices_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_8;  setitem_6 = rel_indices_2;  unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_9 = indy_2.unsqueeze(0);  indy_2 = None
    rel_indices_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_9;  setitem_7 = rel_indices_2;  unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_10 = indx_2.unsqueeze(0);  indx_2 = None
    rel_indices_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_10;  setitem_8 = rel_indices_2;  unsqueeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_2 = rel_indices_2.to(device(type='cpu'));  rel_indices_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_attn_qk = self.L__mod___blocks_2_attn_qk(x_34)
    reshape_6 = l__mod___blocks_2_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_2_attn_qk = None
    qk_2 = reshape_6.permute(2, 0, 3, 1, 4);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_2 = qk_2[0]
    k_2 = qk_2[1];  qk_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_6 = to_2.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_2_attn_pos_proj = self.L__mod___blocks_2_attn_pos_proj(pos_score_6);  pos_score_6 = None
    pos_score_7 = l__mod___blocks_2_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_2_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_5 = k_2.transpose(-2, -1);  k_2 = None
    matmul_4 = q_2 @ transpose_5;  q_2 = transpose_5 = None
    patch_score_4 = matmul_4 * 0.14433756729740643;  matmul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_5 = patch_score_4.softmax(dim = -1);  patch_score_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_8 = pos_score_7.softmax(dim = -1);  pos_score_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_2_attn_gating_param = self.L__mod___blocks_2_attn_gating_param
    gating_2 = l__mod___blocks_2_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_2_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_4 = torch.sigmoid(gating_2)
    sub_5 = 1.0 - sigmoid_4;  sigmoid_4 = None
    mul_7 = sub_5 * patch_score_5;  sub_5 = patch_score_5 = None
    sigmoid_5 = torch.sigmoid(gating_2);  gating_2 = None
    mul_8 = sigmoid_5 * pos_score_8;  sigmoid_5 = pos_score_8 = None
    attn_8 = mul_7 + mul_8;  mul_7 = mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_3 = attn_8.sum(dim = -1)
    unsqueeze_11 = sum_3.unsqueeze(-1);  sum_3 = None
    attn_8 /= unsqueeze_11;  attn_9 = attn_8;  attn_8 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_11 = self.L__mod___blocks_2_attn_attn_drop(attn_9);  attn_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_attn_v = self.L__mod___blocks_2_attn_v(x_34);  x_34 = None
    reshape_7 = l__mod___blocks_2_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_2_attn_v = None
    v_2 = reshape_7.permute(0, 2, 1, 3);  reshape_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_5 = attn_11 @ v_2;  attn_11 = v_2 = None
    transpose_6 = matmul_5.transpose(1, 2);  matmul_5 = None
    x_35 = transpose_6.reshape(8, 196, 768);  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_36 = self.L__mod___blocks_2_attn_proj(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_37 = self.L__mod___blocks_2_attn_proj_drop(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_2_drop_path = self.L__mod___blocks_2_drop_path(x_37);  x_37 = None
    x_38 = x_33 + l__mod___blocks_2_drop_path;  x_33 = l__mod___blocks_2_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_2_norm2_weight = self.L__mod___blocks_2_norm2_weight
    l__mod___blocks_2_norm2_bias = self.L__mod___blocks_2_norm2_bias
    x_39 = torch.nn.functional.layer_norm(x_38, (768,), l__mod___blocks_2_norm2_weight, l__mod___blocks_2_norm2_bias, 1e-06);  l__mod___blocks_2_norm2_weight = l__mod___blocks_2_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_40 = self.L__mod___blocks_2_mlp_fc1(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_41 = self.L__mod___blocks_2_mlp_act(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_42 = self.L__mod___blocks_2_mlp_drop1(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_43 = self.L__mod___blocks_2_mlp_norm(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_44 = self.L__mod___blocks_2_mlp_fc2(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_45 = self.L__mod___blocks_2_mlp_drop2(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_2_drop_path_1 = self.L__mod___blocks_2_drop_path(x_45);  x_45 = None
    x_47 = x_38 + l__mod___blocks_2_drop_path_1;  x_38 = l__mod___blocks_2_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_3_norm1_weight = self.L__mod___blocks_3_norm1_weight
    l__mod___blocks_3_norm1_bias = self.L__mod___blocks_3_norm1_bias
    x_48 = torch.nn.functional.layer_norm(x_47, (768,), l__mod___blocks_3_norm1_weight, l__mod___blocks_3_norm1_bias, 1e-06);  l__mod___blocks_3_norm1_weight = l__mod___blocks_3_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_3 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_6 = torch.arange(14)
    view_9 = arange_6.view(1, -1);  arange_6 = None
    arange_7 = torch.arange(14)
    view_10 = arange_7.view(-1, 1);  arange_7 = None
    ind_3 = view_9 - view_10;  view_9 = view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_3 = ind_3.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_6 = ind_3.repeat_interleave(14, dim = 0);  ind_3 = None
    indy_3 = repeat_interleave_6.repeat_interleave(14, dim = 1);  repeat_interleave_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_7 = indx_3 ** 2
    pow_8 = indy_3 ** 2
    indd_3 = pow_7 + pow_8;  pow_7 = pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_12 = indd_3.unsqueeze(0);  indd_3 = None
    rel_indices_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_12;  setitem_9 = rel_indices_3;  unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_13 = indy_3.unsqueeze(0);  indy_3 = None
    rel_indices_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_13;  setitem_10 = rel_indices_3;  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_14 = indx_3.unsqueeze(0);  indx_3 = None
    rel_indices_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_14;  setitem_11 = rel_indices_3;  unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_3 = rel_indices_3.to(device(type='cpu'));  rel_indices_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_3_attn_qk = self.L__mod___blocks_3_attn_qk(x_48)
    reshape_9 = l__mod___blocks_3_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_3_attn_qk = None
    qk_3 = reshape_9.permute(2, 0, 3, 1, 4);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_3 = qk_3[0]
    k_3 = qk_3[1];  qk_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_9 = to_3.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_3_attn_pos_proj = self.L__mod___blocks_3_attn_pos_proj(pos_score_9);  pos_score_9 = None
    pos_score_10 = l__mod___blocks_3_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_3_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_7 = k_3.transpose(-2, -1);  k_3 = None
    matmul_6 = q_3 @ transpose_7;  q_3 = transpose_7 = None
    patch_score_6 = matmul_6 * 0.14433756729740643;  matmul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_7 = patch_score_6.softmax(dim = -1);  patch_score_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_11 = pos_score_10.softmax(dim = -1);  pos_score_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_3_attn_gating_param = self.L__mod___blocks_3_attn_gating_param
    gating_3 = l__mod___blocks_3_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_3_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_6 = torch.sigmoid(gating_3)
    sub_7 = 1.0 - sigmoid_6;  sigmoid_6 = None
    mul_10 = sub_7 * patch_score_7;  sub_7 = patch_score_7 = None
    sigmoid_7 = torch.sigmoid(gating_3);  gating_3 = None
    mul_11 = sigmoid_7 * pos_score_11;  sigmoid_7 = pos_score_11 = None
    attn_12 = mul_10 + mul_11;  mul_10 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_4 = attn_12.sum(dim = -1)
    unsqueeze_15 = sum_4.unsqueeze(-1);  sum_4 = None
    attn_12 /= unsqueeze_15;  attn_13 = attn_12;  attn_12 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_15 = self.L__mod___blocks_3_attn_attn_drop(attn_13);  attn_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_3_attn_v = self.L__mod___blocks_3_attn_v(x_48);  x_48 = None
    reshape_10 = l__mod___blocks_3_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_3_attn_v = None
    v_3 = reshape_10.permute(0, 2, 1, 3);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_7 = attn_15 @ v_3;  attn_15 = v_3 = None
    transpose_8 = matmul_7.transpose(1, 2);  matmul_7 = None
    x_49 = transpose_8.reshape(8, 196, 768);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_50 = self.L__mod___blocks_3_attn_proj(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_51 = self.L__mod___blocks_3_attn_proj_drop(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_3_drop_path = self.L__mod___blocks_3_drop_path(x_51);  x_51 = None
    x_52 = x_47 + l__mod___blocks_3_drop_path;  x_47 = l__mod___blocks_3_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_3_norm2_weight = self.L__mod___blocks_3_norm2_weight
    l__mod___blocks_3_norm2_bias = self.L__mod___blocks_3_norm2_bias
    x_53 = torch.nn.functional.layer_norm(x_52, (768,), l__mod___blocks_3_norm2_weight, l__mod___blocks_3_norm2_bias, 1e-06);  l__mod___blocks_3_norm2_weight = l__mod___blocks_3_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_54 = self.L__mod___blocks_3_mlp_fc1(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_55 = self.L__mod___blocks_3_mlp_act(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_56 = self.L__mod___blocks_3_mlp_drop1(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_57 = self.L__mod___blocks_3_mlp_norm(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_58 = self.L__mod___blocks_3_mlp_fc2(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_59 = self.L__mod___blocks_3_mlp_drop2(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_3_drop_path_1 = self.L__mod___blocks_3_drop_path(x_59);  x_59 = None
    x_61 = x_52 + l__mod___blocks_3_drop_path_1;  x_52 = l__mod___blocks_3_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_4_norm1_weight = self.L__mod___blocks_4_norm1_weight
    l__mod___blocks_4_norm1_bias = self.L__mod___blocks_4_norm1_bias
    x_62 = torch.nn.functional.layer_norm(x_61, (768,), l__mod___blocks_4_norm1_weight, l__mod___blocks_4_norm1_bias, 1e-06);  l__mod___blocks_4_norm1_weight = l__mod___blocks_4_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_4 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_8 = torch.arange(14)
    view_12 = arange_8.view(1, -1);  arange_8 = None
    arange_9 = torch.arange(14)
    view_13 = arange_9.view(-1, 1);  arange_9 = None
    ind_4 = view_12 - view_13;  view_12 = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_4 = ind_4.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_8 = ind_4.repeat_interleave(14, dim = 0);  ind_4 = None
    indy_4 = repeat_interleave_8.repeat_interleave(14, dim = 1);  repeat_interleave_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_9 = indx_4 ** 2
    pow_10 = indy_4 ** 2
    indd_4 = pow_9 + pow_10;  pow_9 = pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_16 = indd_4.unsqueeze(0);  indd_4 = None
    rel_indices_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_16;  setitem_12 = rel_indices_4;  unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_17 = indy_4.unsqueeze(0);  indy_4 = None
    rel_indices_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_17;  setitem_13 = rel_indices_4;  unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_18 = indx_4.unsqueeze(0);  indx_4 = None
    rel_indices_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_18;  setitem_14 = rel_indices_4;  unsqueeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_4 = rel_indices_4.to(device(type='cpu'));  rel_indices_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_4_attn_qk = self.L__mod___blocks_4_attn_qk(x_62)
    reshape_12 = l__mod___blocks_4_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_4_attn_qk = None
    qk_4 = reshape_12.permute(2, 0, 3, 1, 4);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_4 = qk_4[0]
    k_4 = qk_4[1];  qk_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_12 = to_4.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_4_attn_pos_proj = self.L__mod___blocks_4_attn_pos_proj(pos_score_12);  pos_score_12 = None
    pos_score_13 = l__mod___blocks_4_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_4_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_9 = k_4.transpose(-2, -1);  k_4 = None
    matmul_8 = q_4 @ transpose_9;  q_4 = transpose_9 = None
    patch_score_8 = matmul_8 * 0.14433756729740643;  matmul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_9 = patch_score_8.softmax(dim = -1);  patch_score_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_14 = pos_score_13.softmax(dim = -1);  pos_score_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_4_attn_gating_param = self.L__mod___blocks_4_attn_gating_param
    gating_4 = l__mod___blocks_4_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_4_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_8 = torch.sigmoid(gating_4)
    sub_9 = 1.0 - sigmoid_8;  sigmoid_8 = None
    mul_13 = sub_9 * patch_score_9;  sub_9 = patch_score_9 = None
    sigmoid_9 = torch.sigmoid(gating_4);  gating_4 = None
    mul_14 = sigmoid_9 * pos_score_14;  sigmoid_9 = pos_score_14 = None
    attn_16 = mul_13 + mul_14;  mul_13 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_5 = attn_16.sum(dim = -1)
    unsqueeze_19 = sum_5.unsqueeze(-1);  sum_5 = None
    attn_16 /= unsqueeze_19;  attn_17 = attn_16;  attn_16 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_19 = self.L__mod___blocks_4_attn_attn_drop(attn_17);  attn_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_4_attn_v = self.L__mod___blocks_4_attn_v(x_62);  x_62 = None
    reshape_13 = l__mod___blocks_4_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_4_attn_v = None
    v_4 = reshape_13.permute(0, 2, 1, 3);  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_9 = attn_19 @ v_4;  attn_19 = v_4 = None
    transpose_10 = matmul_9.transpose(1, 2);  matmul_9 = None
    x_63 = transpose_10.reshape(8, 196, 768);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_64 = self.L__mod___blocks_4_attn_proj(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_65 = self.L__mod___blocks_4_attn_proj_drop(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_4_drop_path = self.L__mod___blocks_4_drop_path(x_65);  x_65 = None
    x_66 = x_61 + l__mod___blocks_4_drop_path;  x_61 = l__mod___blocks_4_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_4_norm2_weight = self.L__mod___blocks_4_norm2_weight
    l__mod___blocks_4_norm2_bias = self.L__mod___blocks_4_norm2_bias
    x_67 = torch.nn.functional.layer_norm(x_66, (768,), l__mod___blocks_4_norm2_weight, l__mod___blocks_4_norm2_bias, 1e-06);  l__mod___blocks_4_norm2_weight = l__mod___blocks_4_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_68 = self.L__mod___blocks_4_mlp_fc1(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_69 = self.L__mod___blocks_4_mlp_act(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_70 = self.L__mod___blocks_4_mlp_drop1(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_71 = self.L__mod___blocks_4_mlp_norm(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_72 = self.L__mod___blocks_4_mlp_fc2(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_73 = self.L__mod___blocks_4_mlp_drop2(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_4_drop_path_1 = self.L__mod___blocks_4_drop_path(x_73);  x_73 = None
    x_75 = x_66 + l__mod___blocks_4_drop_path_1;  x_66 = l__mod___blocks_4_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_5_norm1_weight = self.L__mod___blocks_5_norm1_weight
    l__mod___blocks_5_norm1_bias = self.L__mod___blocks_5_norm1_bias
    x_76 = torch.nn.functional.layer_norm(x_75, (768,), l__mod___blocks_5_norm1_weight, l__mod___blocks_5_norm1_bias, 1e-06);  l__mod___blocks_5_norm1_weight = l__mod___blocks_5_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_5 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_10 = torch.arange(14)
    view_15 = arange_10.view(1, -1);  arange_10 = None
    arange_11 = torch.arange(14)
    view_16 = arange_11.view(-1, 1);  arange_11 = None
    ind_5 = view_15 - view_16;  view_15 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_5 = ind_5.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_10 = ind_5.repeat_interleave(14, dim = 0);  ind_5 = None
    indy_5 = repeat_interleave_10.repeat_interleave(14, dim = 1);  repeat_interleave_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_11 = indx_5 ** 2
    pow_12 = indy_5 ** 2
    indd_5 = pow_11 + pow_12;  pow_11 = pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_20 = indd_5.unsqueeze(0);  indd_5 = None
    rel_indices_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_20;  setitem_15 = rel_indices_5;  unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_21 = indy_5.unsqueeze(0);  indy_5 = None
    rel_indices_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_21;  setitem_16 = rel_indices_5;  unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_22 = indx_5.unsqueeze(0);  indx_5 = None
    rel_indices_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_22;  setitem_17 = rel_indices_5;  unsqueeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_5 = rel_indices_5.to(device(type='cpu'));  rel_indices_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_5_attn_qk = self.L__mod___blocks_5_attn_qk(x_76)
    reshape_15 = l__mod___blocks_5_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_5_attn_qk = None
    qk_5 = reshape_15.permute(2, 0, 3, 1, 4);  reshape_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_5 = qk_5[0]
    k_5 = qk_5[1];  qk_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_15 = to_5.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_5_attn_pos_proj = self.L__mod___blocks_5_attn_pos_proj(pos_score_15);  pos_score_15 = None
    pos_score_16 = l__mod___blocks_5_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_5_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_11 = k_5.transpose(-2, -1);  k_5 = None
    matmul_10 = q_5 @ transpose_11;  q_5 = transpose_11 = None
    patch_score_10 = matmul_10 * 0.14433756729740643;  matmul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_11 = patch_score_10.softmax(dim = -1);  patch_score_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_17 = pos_score_16.softmax(dim = -1);  pos_score_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_5_attn_gating_param = self.L__mod___blocks_5_attn_gating_param
    gating_5 = l__mod___blocks_5_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_5_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_10 = torch.sigmoid(gating_5)
    sub_11 = 1.0 - sigmoid_10;  sigmoid_10 = None
    mul_16 = sub_11 * patch_score_11;  sub_11 = patch_score_11 = None
    sigmoid_11 = torch.sigmoid(gating_5);  gating_5 = None
    mul_17 = sigmoid_11 * pos_score_17;  sigmoid_11 = pos_score_17 = None
    attn_20 = mul_16 + mul_17;  mul_16 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_6 = attn_20.sum(dim = -1)
    unsqueeze_23 = sum_6.unsqueeze(-1);  sum_6 = None
    attn_20 /= unsqueeze_23;  attn_21 = attn_20;  attn_20 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_23 = self.L__mod___blocks_5_attn_attn_drop(attn_21);  attn_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_5_attn_v = self.L__mod___blocks_5_attn_v(x_76);  x_76 = None
    reshape_16 = l__mod___blocks_5_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_5_attn_v = None
    v_5 = reshape_16.permute(0, 2, 1, 3);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_11 = attn_23 @ v_5;  attn_23 = v_5 = None
    transpose_12 = matmul_11.transpose(1, 2);  matmul_11 = None
    x_77 = transpose_12.reshape(8, 196, 768);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_78 = self.L__mod___blocks_5_attn_proj(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_79 = self.L__mod___blocks_5_attn_proj_drop(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_5_drop_path = self.L__mod___blocks_5_drop_path(x_79);  x_79 = None
    x_80 = x_75 + l__mod___blocks_5_drop_path;  x_75 = l__mod___blocks_5_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_5_norm2_weight = self.L__mod___blocks_5_norm2_weight
    l__mod___blocks_5_norm2_bias = self.L__mod___blocks_5_norm2_bias
    x_81 = torch.nn.functional.layer_norm(x_80, (768,), l__mod___blocks_5_norm2_weight, l__mod___blocks_5_norm2_bias, 1e-06);  l__mod___blocks_5_norm2_weight = l__mod___blocks_5_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_82 = self.L__mod___blocks_5_mlp_fc1(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_83 = self.L__mod___blocks_5_mlp_act(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_84 = self.L__mod___blocks_5_mlp_drop1(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_85 = self.L__mod___blocks_5_mlp_norm(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_86 = self.L__mod___blocks_5_mlp_fc2(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_87 = self.L__mod___blocks_5_mlp_drop2(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_5_drop_path_1 = self.L__mod___blocks_5_drop_path(x_87);  x_87 = None
    x_89 = x_80 + l__mod___blocks_5_drop_path_1;  x_80 = l__mod___blocks_5_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_6_norm1_weight = self.L__mod___blocks_6_norm1_weight
    l__mod___blocks_6_norm1_bias = self.L__mod___blocks_6_norm1_bias
    x_90 = torch.nn.functional.layer_norm(x_89, (768,), l__mod___blocks_6_norm1_weight, l__mod___blocks_6_norm1_bias, 1e-06);  l__mod___blocks_6_norm1_weight = l__mod___blocks_6_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_6 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_12 = torch.arange(14)
    view_18 = arange_12.view(1, -1);  arange_12 = None
    arange_13 = torch.arange(14)
    view_19 = arange_13.view(-1, 1);  arange_13 = None
    ind_6 = view_18 - view_19;  view_18 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_6 = ind_6.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_12 = ind_6.repeat_interleave(14, dim = 0);  ind_6 = None
    indy_6 = repeat_interleave_12.repeat_interleave(14, dim = 1);  repeat_interleave_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_13 = indx_6 ** 2
    pow_14 = indy_6 ** 2
    indd_6 = pow_13 + pow_14;  pow_13 = pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_24 = indd_6.unsqueeze(0);  indd_6 = None
    rel_indices_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_24;  setitem_18 = rel_indices_6;  unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_25 = indy_6.unsqueeze(0);  indy_6 = None
    rel_indices_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_25;  setitem_19 = rel_indices_6;  unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_26 = indx_6.unsqueeze(0);  indx_6 = None
    rel_indices_6[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_26;  setitem_20 = rel_indices_6;  unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_6 = rel_indices_6.to(device(type='cpu'));  rel_indices_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_6_attn_qk = self.L__mod___blocks_6_attn_qk(x_90)
    reshape_18 = l__mod___blocks_6_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_6_attn_qk = None
    qk_6 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_6 = qk_6[0]
    k_6 = qk_6[1];  qk_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_18 = to_6.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_6_attn_pos_proj = self.L__mod___blocks_6_attn_pos_proj(pos_score_18);  pos_score_18 = None
    pos_score_19 = l__mod___blocks_6_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_6_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_13 = k_6.transpose(-2, -1);  k_6 = None
    matmul_12 = q_6 @ transpose_13;  q_6 = transpose_13 = None
    patch_score_12 = matmul_12 * 0.14433756729740643;  matmul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_13 = patch_score_12.softmax(dim = -1);  patch_score_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_20 = pos_score_19.softmax(dim = -1);  pos_score_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_6_attn_gating_param = self.L__mod___blocks_6_attn_gating_param
    gating_6 = l__mod___blocks_6_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_6_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_12 = torch.sigmoid(gating_6)
    sub_13 = 1.0 - sigmoid_12;  sigmoid_12 = None
    mul_19 = sub_13 * patch_score_13;  sub_13 = patch_score_13 = None
    sigmoid_13 = torch.sigmoid(gating_6);  gating_6 = None
    mul_20 = sigmoid_13 * pos_score_20;  sigmoid_13 = pos_score_20 = None
    attn_24 = mul_19 + mul_20;  mul_19 = mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_7 = attn_24.sum(dim = -1)
    unsqueeze_27 = sum_7.unsqueeze(-1);  sum_7 = None
    attn_24 /= unsqueeze_27;  attn_25 = attn_24;  attn_24 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_27 = self.L__mod___blocks_6_attn_attn_drop(attn_25);  attn_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_6_attn_v = self.L__mod___blocks_6_attn_v(x_90);  x_90 = None
    reshape_19 = l__mod___blocks_6_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_6_attn_v = None
    v_6 = reshape_19.permute(0, 2, 1, 3);  reshape_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_13 = attn_27 @ v_6;  attn_27 = v_6 = None
    transpose_14 = matmul_13.transpose(1, 2);  matmul_13 = None
    x_91 = transpose_14.reshape(8, 196, 768);  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_92 = self.L__mod___blocks_6_attn_proj(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_93 = self.L__mod___blocks_6_attn_proj_drop(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_6_drop_path = self.L__mod___blocks_6_drop_path(x_93);  x_93 = None
    x_94 = x_89 + l__mod___blocks_6_drop_path;  x_89 = l__mod___blocks_6_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_6_norm2_weight = self.L__mod___blocks_6_norm2_weight
    l__mod___blocks_6_norm2_bias = self.L__mod___blocks_6_norm2_bias
    x_95 = torch.nn.functional.layer_norm(x_94, (768,), l__mod___blocks_6_norm2_weight, l__mod___blocks_6_norm2_bias, 1e-06);  l__mod___blocks_6_norm2_weight = l__mod___blocks_6_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_96 = self.L__mod___blocks_6_mlp_fc1(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_97 = self.L__mod___blocks_6_mlp_act(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_98 = self.L__mod___blocks_6_mlp_drop1(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_99 = self.L__mod___blocks_6_mlp_norm(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_100 = self.L__mod___blocks_6_mlp_fc2(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_101 = self.L__mod___blocks_6_mlp_drop2(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_6_drop_path_1 = self.L__mod___blocks_6_drop_path(x_101);  x_101 = None
    x_103 = x_94 + l__mod___blocks_6_drop_path_1;  x_94 = l__mod___blocks_6_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_7_norm1_weight = self.L__mod___blocks_7_norm1_weight
    l__mod___blocks_7_norm1_bias = self.L__mod___blocks_7_norm1_bias
    x_104 = torch.nn.functional.layer_norm(x_103, (768,), l__mod___blocks_7_norm1_weight, l__mod___blocks_7_norm1_bias, 1e-06);  l__mod___blocks_7_norm1_weight = l__mod___blocks_7_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_7 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_14 = torch.arange(14)
    view_21 = arange_14.view(1, -1);  arange_14 = None
    arange_15 = torch.arange(14)
    view_22 = arange_15.view(-1, 1);  arange_15 = None
    ind_7 = view_21 - view_22;  view_21 = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_7 = ind_7.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_14 = ind_7.repeat_interleave(14, dim = 0);  ind_7 = None
    indy_7 = repeat_interleave_14.repeat_interleave(14, dim = 1);  repeat_interleave_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_15 = indx_7 ** 2
    pow_16 = indy_7 ** 2
    indd_7 = pow_15 + pow_16;  pow_15 = pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_28 = indd_7.unsqueeze(0);  indd_7 = None
    rel_indices_7[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_28;  setitem_21 = rel_indices_7;  unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_29 = indy_7.unsqueeze(0);  indy_7 = None
    rel_indices_7[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_29;  setitem_22 = rel_indices_7;  unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_30 = indx_7.unsqueeze(0);  indx_7 = None
    rel_indices_7[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_30;  setitem_23 = rel_indices_7;  unsqueeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_7 = rel_indices_7.to(device(type='cpu'));  rel_indices_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_7_attn_qk = self.L__mod___blocks_7_attn_qk(x_104)
    reshape_21 = l__mod___blocks_7_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_7_attn_qk = None
    qk_7 = reshape_21.permute(2, 0, 3, 1, 4);  reshape_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_7 = qk_7[0]
    k_7 = qk_7[1];  qk_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_21 = to_7.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_7_attn_pos_proj = self.L__mod___blocks_7_attn_pos_proj(pos_score_21);  pos_score_21 = None
    pos_score_22 = l__mod___blocks_7_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_7_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_15 = k_7.transpose(-2, -1);  k_7 = None
    matmul_14 = q_7 @ transpose_15;  q_7 = transpose_15 = None
    patch_score_14 = matmul_14 * 0.14433756729740643;  matmul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_15 = patch_score_14.softmax(dim = -1);  patch_score_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_23 = pos_score_22.softmax(dim = -1);  pos_score_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_7_attn_gating_param = self.L__mod___blocks_7_attn_gating_param
    gating_7 = l__mod___blocks_7_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_7_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_14 = torch.sigmoid(gating_7)
    sub_15 = 1.0 - sigmoid_14;  sigmoid_14 = None
    mul_22 = sub_15 * patch_score_15;  sub_15 = patch_score_15 = None
    sigmoid_15 = torch.sigmoid(gating_7);  gating_7 = None
    mul_23 = sigmoid_15 * pos_score_23;  sigmoid_15 = pos_score_23 = None
    attn_28 = mul_22 + mul_23;  mul_22 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_8 = attn_28.sum(dim = -1)
    unsqueeze_31 = sum_8.unsqueeze(-1);  sum_8 = None
    attn_28 /= unsqueeze_31;  attn_29 = attn_28;  attn_28 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_31 = self.L__mod___blocks_7_attn_attn_drop(attn_29);  attn_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_7_attn_v = self.L__mod___blocks_7_attn_v(x_104);  x_104 = None
    reshape_22 = l__mod___blocks_7_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_7_attn_v = None
    v_7 = reshape_22.permute(0, 2, 1, 3);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_15 = attn_31 @ v_7;  attn_31 = v_7 = None
    transpose_16 = matmul_15.transpose(1, 2);  matmul_15 = None
    x_105 = transpose_16.reshape(8, 196, 768);  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_106 = self.L__mod___blocks_7_attn_proj(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_107 = self.L__mod___blocks_7_attn_proj_drop(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_7_drop_path = self.L__mod___blocks_7_drop_path(x_107);  x_107 = None
    x_108 = x_103 + l__mod___blocks_7_drop_path;  x_103 = l__mod___blocks_7_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_7_norm2_weight = self.L__mod___blocks_7_norm2_weight
    l__mod___blocks_7_norm2_bias = self.L__mod___blocks_7_norm2_bias
    x_109 = torch.nn.functional.layer_norm(x_108, (768,), l__mod___blocks_7_norm2_weight, l__mod___blocks_7_norm2_bias, 1e-06);  l__mod___blocks_7_norm2_weight = l__mod___blocks_7_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_110 = self.L__mod___blocks_7_mlp_fc1(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_111 = self.L__mod___blocks_7_mlp_act(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_112 = self.L__mod___blocks_7_mlp_drop1(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_113 = self.L__mod___blocks_7_mlp_norm(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_114 = self.L__mod___blocks_7_mlp_fc2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_115 = self.L__mod___blocks_7_mlp_drop2(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_7_drop_path_1 = self.L__mod___blocks_7_drop_path(x_115);  x_115 = None
    x_117 = x_108 + l__mod___blocks_7_drop_path_1;  x_108 = l__mod___blocks_7_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_8_norm1_weight = self.L__mod___blocks_8_norm1_weight
    l__mod___blocks_8_norm1_bias = self.L__mod___blocks_8_norm1_bias
    x_118 = torch.nn.functional.layer_norm(x_117, (768,), l__mod___blocks_8_norm1_weight, l__mod___blocks_8_norm1_bias, 1e-06);  l__mod___blocks_8_norm1_weight = l__mod___blocks_8_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_8 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_16 = torch.arange(14)
    view_24 = arange_16.view(1, -1);  arange_16 = None
    arange_17 = torch.arange(14)
    view_25 = arange_17.view(-1, 1);  arange_17 = None
    ind_8 = view_24 - view_25;  view_24 = view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_8 = ind_8.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_16 = ind_8.repeat_interleave(14, dim = 0);  ind_8 = None
    indy_8 = repeat_interleave_16.repeat_interleave(14, dim = 1);  repeat_interleave_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_17 = indx_8 ** 2
    pow_18 = indy_8 ** 2
    indd_8 = pow_17 + pow_18;  pow_17 = pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_32 = indd_8.unsqueeze(0);  indd_8 = None
    rel_indices_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_32;  setitem_24 = rel_indices_8;  unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_33 = indy_8.unsqueeze(0);  indy_8 = None
    rel_indices_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_33;  setitem_25 = rel_indices_8;  unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_34 = indx_8.unsqueeze(0);  indx_8 = None
    rel_indices_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_34;  setitem_26 = rel_indices_8;  unsqueeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_8 = rel_indices_8.to(device(type='cpu'));  rel_indices_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_8_attn_qk = self.L__mod___blocks_8_attn_qk(x_118)
    reshape_24 = l__mod___blocks_8_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_8_attn_qk = None
    qk_8 = reshape_24.permute(2, 0, 3, 1, 4);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_8 = qk_8[0]
    k_8 = qk_8[1];  qk_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_24 = to_8.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_8_attn_pos_proj = self.L__mod___blocks_8_attn_pos_proj(pos_score_24);  pos_score_24 = None
    pos_score_25 = l__mod___blocks_8_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_8_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_17 = k_8.transpose(-2, -1);  k_8 = None
    matmul_16 = q_8 @ transpose_17;  q_8 = transpose_17 = None
    patch_score_16 = matmul_16 * 0.14433756729740643;  matmul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_17 = patch_score_16.softmax(dim = -1);  patch_score_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_26 = pos_score_25.softmax(dim = -1);  pos_score_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_8_attn_gating_param = self.L__mod___blocks_8_attn_gating_param
    gating_8 = l__mod___blocks_8_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_8_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_16 = torch.sigmoid(gating_8)
    sub_17 = 1.0 - sigmoid_16;  sigmoid_16 = None
    mul_25 = sub_17 * patch_score_17;  sub_17 = patch_score_17 = None
    sigmoid_17 = torch.sigmoid(gating_8);  gating_8 = None
    mul_26 = sigmoid_17 * pos_score_26;  sigmoid_17 = pos_score_26 = None
    attn_32 = mul_25 + mul_26;  mul_25 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_9 = attn_32.sum(dim = -1)
    unsqueeze_35 = sum_9.unsqueeze(-1);  sum_9 = None
    attn_32 /= unsqueeze_35;  attn_33 = attn_32;  attn_32 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_35 = self.L__mod___blocks_8_attn_attn_drop(attn_33);  attn_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_8_attn_v = self.L__mod___blocks_8_attn_v(x_118);  x_118 = None
    reshape_25 = l__mod___blocks_8_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_8_attn_v = None
    v_8 = reshape_25.permute(0, 2, 1, 3);  reshape_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_17 = attn_35 @ v_8;  attn_35 = v_8 = None
    transpose_18 = matmul_17.transpose(1, 2);  matmul_17 = None
    x_119 = transpose_18.reshape(8, 196, 768);  transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_120 = self.L__mod___blocks_8_attn_proj(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_121 = self.L__mod___blocks_8_attn_proj_drop(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_8_drop_path = self.L__mod___blocks_8_drop_path(x_121);  x_121 = None
    x_122 = x_117 + l__mod___blocks_8_drop_path;  x_117 = l__mod___blocks_8_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_8_norm2_weight = self.L__mod___blocks_8_norm2_weight
    l__mod___blocks_8_norm2_bias = self.L__mod___blocks_8_norm2_bias
    x_123 = torch.nn.functional.layer_norm(x_122, (768,), l__mod___blocks_8_norm2_weight, l__mod___blocks_8_norm2_bias, 1e-06);  l__mod___blocks_8_norm2_weight = l__mod___blocks_8_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_124 = self.L__mod___blocks_8_mlp_fc1(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_125 = self.L__mod___blocks_8_mlp_act(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_126 = self.L__mod___blocks_8_mlp_drop1(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_127 = self.L__mod___blocks_8_mlp_norm(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_128 = self.L__mod___blocks_8_mlp_fc2(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_129 = self.L__mod___blocks_8_mlp_drop2(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_8_drop_path_1 = self.L__mod___blocks_8_drop_path(x_129);  x_129 = None
    x_131 = x_122 + l__mod___blocks_8_drop_path_1;  x_122 = l__mod___blocks_8_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_9_norm1_weight = self.L__mod___blocks_9_norm1_weight
    l__mod___blocks_9_norm1_bias = self.L__mod___blocks_9_norm1_bias
    x_132 = torch.nn.functional.layer_norm(x_131, (768,), l__mod___blocks_9_norm1_weight, l__mod___blocks_9_norm1_bias, 1e-06);  l__mod___blocks_9_norm1_weight = l__mod___blocks_9_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:121, code: rel_indices = torch.zeros(1, num_patches, num_patches, 3)
    rel_indices_9 = torch.zeros(1, 196, 196, 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:122, code: ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
    arange_18 = torch.arange(14)
    view_27 = arange_18.view(1, -1);  arange_18 = None
    arange_19 = torch.arange(14)
    view_28 = arange_19.view(-1, 1);  arange_19 = None
    ind_9 = view_27 - view_28;  view_27 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:123, code: indx = ind.repeat(img_size, img_size)
    indx_9 = ind_9.repeat(14, 14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:124, code: indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
    repeat_interleave_18 = ind_9.repeat_interleave(14, dim = 0);  ind_9 = None
    indy_9 = repeat_interleave_18.repeat_interleave(14, dim = 1);  repeat_interleave_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:125, code: indd = indx ** 2 + indy ** 2
    pow_19 = indx_9 ** 2
    pow_20 = indy_9 ** 2
    indd_9 = pow_19 + pow_20;  pow_19 = pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:126, code: rel_indices[:, :, :, 2] = indd.unsqueeze(0)
    unsqueeze_36 = indd_9.unsqueeze(0);  indd_9 = None
    rel_indices_9[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = unsqueeze_36;  setitem_27 = rel_indices_9;  unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:127, code: rel_indices[:, :, :, 1] = indy.unsqueeze(0)
    unsqueeze_37 = indy_9.unsqueeze(0);  indy_9 = None
    rel_indices_9[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = unsqueeze_37;  setitem_28 = rel_indices_9;  unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:128, code: rel_indices[:, :, :, 0] = indx.unsqueeze(0)
    unsqueeze_38 = indx_9.unsqueeze(0);  indx_9 = None
    rel_indices_9[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = unsqueeze_38;  setitem_29 = rel_indices_9;  unsqueeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:130, code: return rel_indices.to(device)
    to_9 = rel_indices_9.to(device(type='cpu'));  rel_indices_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_9_attn_qk = self.L__mod___blocks_9_attn_qk(x_132)
    reshape_27 = l__mod___blocks_9_attn_qk.reshape(8, 196, 2, 16, 48);  l__mod___blocks_9_attn_qk = None
    qk_9 = reshape_27.permute(2, 0, 3, 1, 4);  reshape_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    q_9 = qk_9[0]
    k_9 = qk_9[1];  qk_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:84, code: pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score_27 = to_9.expand(8, -1, -1, -1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    l__mod___blocks_9_attn_pos_proj = self.L__mod___blocks_9_attn_pos_proj(pos_score_27);  pos_score_27 = None
    pos_score_28 = l__mod___blocks_9_attn_pos_proj.permute(0, 3, 1, 2);  l__mod___blocks_9_attn_pos_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    transpose_19 = k_9.transpose(-2, -1);  k_9 = None
    matmul_18 = q_9 @ transpose_19;  q_9 = transpose_19 = None
    patch_score_18 = matmul_18 * 0.14433756729740643;  matmul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    patch_score_19 = patch_score_18.softmax(dim = -1);  patch_score_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    pos_score_29 = pos_score_28.softmax(dim = -1);  pos_score_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    l__mod___blocks_9_attn_gating_param = self.L__mod___blocks_9_attn_gating_param
    gating_9 = l__mod___blocks_9_attn_gating_param.view(1, -1, 1, 1);  l__mod___blocks_9_attn_gating_param = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_18 = torch.sigmoid(gating_9)
    sub_19 = 1.0 - sigmoid_18;  sigmoid_18 = None
    mul_28 = sub_19 * patch_score_19;  sub_19 = patch_score_19 = None
    sigmoid_19 = torch.sigmoid(gating_9);  gating_9 = None
    mul_29 = sigmoid_19 * pos_score_29;  sigmoid_19 = pos_score_29 = None
    attn_36 = mul_28 + mul_29;  mul_28 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    sum_10 = attn_36.sum(dim = -1)
    unsqueeze_39 = sum_10.unsqueeze(-1);  sum_10 = None
    attn_36 /= unsqueeze_39;  attn_37 = attn_36;  attn_36 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:93, code: attn = self.attn_drop(attn)
    attn_39 = self.L__mod___blocks_9_attn_attn_drop(attn_37);  attn_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_9_attn_v = self.L__mod___blocks_9_attn_v(x_132);  x_132 = None
    reshape_28 = l__mod___blocks_9_attn_v.reshape(8, 196, 16, 48);  l__mod___blocks_9_attn_v = None
    v_9 = reshape_28.permute(0, 2, 1, 3);  reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_19 = attn_39 @ v_9;  attn_39 = v_9 = None
    transpose_20 = matmul_19.transpose(1, 2);  matmul_19 = None
    x_133 = transpose_20.reshape(8, 196, 768);  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    x_134 = self.L__mod___blocks_9_attn_proj(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:77, code: x = self.proj_drop(x)
    x_135 = self.L__mod___blocks_9_attn_proj_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_9_drop_path = self.L__mod___blocks_9_drop_path(x_135);  x_135 = None
    x_136 = x_131 + l__mod___blocks_9_drop_path;  x_131 = l__mod___blocks_9_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_9_norm2_weight = self.L__mod___blocks_9_norm2_weight
    l__mod___blocks_9_norm2_bias = self.L__mod___blocks_9_norm2_bias
    x_137 = torch.nn.functional.layer_norm(x_136, (768,), l__mod___blocks_9_norm2_weight, l__mod___blocks_9_norm2_bias, 1e-06);  l__mod___blocks_9_norm2_weight = l__mod___blocks_9_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_138 = self.L__mod___blocks_9_mlp_fc1(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_139 = self.L__mod___blocks_9_mlp_act(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_140 = self.L__mod___blocks_9_mlp_drop1(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_141 = self.L__mod___blocks_9_mlp_norm(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_142 = self.L__mod___blocks_9_mlp_fc2(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_143 = self.L__mod___blocks_9_mlp_drop2(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_9_drop_path_1 = self.L__mod___blocks_9_drop_path(x_143);  x_143 = None
    x_145 = x_136 + l__mod___blocks_9_drop_path_1;  x_136 = l__mod___blocks_9_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:367, code: x = torch.cat((cls_tokens, x), dim=1)
    x_146 = torch.cat((cls_tokens, x_145), dim = 1);  cls_tokens = x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_10_norm1_weight = self.L__mod___blocks_10_norm1_weight
    l__mod___blocks_10_norm1_bias = self.L__mod___blocks_10_norm1_bias
    x_147 = torch.nn.functional.layer_norm(x_146, (768,), l__mod___blocks_10_norm1_weight, l__mod___blocks_10_norm1_bias, 1e-06);  l__mod___blocks_10_norm1_weight = l__mod___blocks_10_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_10_attn_qkv = self.L__mod___blocks_10_attn_qkv(x_147);  x_147 = None
    reshape_30 = l__mod___blocks_10_attn_qkv.reshape(8, 197, 3, 16, 48);  l__mod___blocks_10_attn_qkv = None
    qkv = reshape_30.permute(2, 0, 3, 1, 4);  reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind = qkv.unbind(0);  qkv = None
    q_10 = unbind[0]
    k_10 = unbind[1]
    v_10 = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_21 = k_10.transpose(-2, -1);  k_10 = None
    matmul_20 = q_10 @ transpose_21;  q_10 = transpose_21 = None
    attn_40 = matmul_20 * 0.14433756729740643;  matmul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    attn_41 = attn_40.softmax(dim = -1);  attn_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:180, code: attn = self.attn_drop(attn)
    attn_42 = self.L__mod___blocks_10_attn_attn_drop(attn_41);  attn_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_21 = attn_42 @ v_10;  attn_42 = v_10 = None
    transpose_22 = matmul_21.transpose(1, 2);  matmul_21 = None
    x_148 = transpose_22.reshape(8, 197, 768);  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    x_149 = self.L__mod___blocks_10_attn_proj(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:184, code: x = self.proj_drop(x)
    x_150 = self.L__mod___blocks_10_attn_proj_drop(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_10_drop_path = self.L__mod___blocks_10_drop_path(x_150);  x_150 = None
    x_151 = x_146 + l__mod___blocks_10_drop_path;  x_146 = l__mod___blocks_10_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_10_norm2_weight = self.L__mod___blocks_10_norm2_weight
    l__mod___blocks_10_norm2_bias = self.L__mod___blocks_10_norm2_bias
    x_152 = torch.nn.functional.layer_norm(x_151, (768,), l__mod___blocks_10_norm2_weight, l__mod___blocks_10_norm2_bias, 1e-06);  l__mod___blocks_10_norm2_weight = l__mod___blocks_10_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_153 = self.L__mod___blocks_10_mlp_fc1(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_154 = self.L__mod___blocks_10_mlp_act(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_155 = self.L__mod___blocks_10_mlp_drop1(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_156 = self.L__mod___blocks_10_mlp_norm(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_157 = self.L__mod___blocks_10_mlp_fc2(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_158 = self.L__mod___blocks_10_mlp_drop2(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_10_drop_path_1 = self.L__mod___blocks_10_drop_path(x_158);  x_158 = None
    x_160 = x_151 + l__mod___blocks_10_drop_path_1;  x_151 = l__mod___blocks_10_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_11_norm1_weight = self.L__mod___blocks_11_norm1_weight
    l__mod___blocks_11_norm1_bias = self.L__mod___blocks_11_norm1_bias
    x_161 = torch.nn.functional.layer_norm(x_160, (768,), l__mod___blocks_11_norm1_weight, l__mod___blocks_11_norm1_bias, 1e-06);  l__mod___blocks_11_norm1_weight = l__mod___blocks_11_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_11_attn_qkv = self.L__mod___blocks_11_attn_qkv(x_161);  x_161 = None
    reshape_32 = l__mod___blocks_11_attn_qkv.reshape(8, 197, 3, 16, 48);  l__mod___blocks_11_attn_qkv = None
    qkv_1 = reshape_32.permute(2, 0, 3, 1, 4);  reshape_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_11 = unbind_1[0]
    k_11 = unbind_1[1]
    v_11 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_23 = k_11.transpose(-2, -1);  k_11 = None
    matmul_22 = q_11 @ transpose_23;  q_11 = transpose_23 = None
    attn_43 = matmul_22 * 0.14433756729740643;  matmul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    attn_44 = attn_43.softmax(dim = -1);  attn_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:180, code: attn = self.attn_drop(attn)
    attn_45 = self.L__mod___blocks_11_attn_attn_drop(attn_44);  attn_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_23 = attn_45 @ v_11;  attn_45 = v_11 = None
    transpose_24 = matmul_23.transpose(1, 2);  matmul_23 = None
    x_162 = transpose_24.reshape(8, 197, 768);  transpose_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    x_163 = self.L__mod___blocks_11_attn_proj(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:184, code: x = self.proj_drop(x)
    x_164 = self.L__mod___blocks_11_attn_proj_drop(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:235, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_11_drop_path = self.L__mod___blocks_11_drop_path(x_164);  x_164 = None
    x_165 = x_160 + l__mod___blocks_11_drop_path;  x_160 = l__mod___blocks_11_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_11_norm2_weight = self.L__mod___blocks_11_norm2_weight
    l__mod___blocks_11_norm2_bias = self.L__mod___blocks_11_norm2_bias
    x_166 = torch.nn.functional.layer_norm(x_165, (768,), l__mod___blocks_11_norm2_weight, l__mod___blocks_11_norm2_bias, 1e-06);  l__mod___blocks_11_norm2_weight = l__mod___blocks_11_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_167 = self.L__mod___blocks_11_mlp_fc1(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_168 = self.L__mod___blocks_11_mlp_act(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_169 = self.L__mod___blocks_11_mlp_drop1(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_170 = self.L__mod___blocks_11_mlp_norm(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_171 = self.L__mod___blocks_11_mlp_fc2(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_172 = self.L__mod___blocks_11_mlp_drop2(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:236, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    l__mod___blocks_11_drop_path_1 = self.L__mod___blocks_11_drop_path(x_172);  x_172 = None
    x_174 = x_165 + l__mod___blocks_11_drop_path_1;  x_165 = l__mod___blocks_11_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___norm_weight = self.L__mod___norm_weight
    l__mod___norm_bias = self.L__mod___norm_bias
    x_177 = torch.nn.functional.layer_norm(x_174, (768,), l__mod___norm_weight, l__mod___norm_bias, 1e-06);  x_174 = l__mod___norm_weight = l__mod___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:374, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    x_178 = x_177[(slice(None, None, None), 0)];  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:375, code: x = self.head_drop(x)
    x_179 = self.L__mod___head_drop(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:376, code: return x if pre_logits else self.head(x)
    x_180 = self.L__mod___head(x_179);  x_179 = None
    return (x_180, to, to_1, to_2, to_3, to_4, to_5, to_6, to_7, to_8, to_9)
    