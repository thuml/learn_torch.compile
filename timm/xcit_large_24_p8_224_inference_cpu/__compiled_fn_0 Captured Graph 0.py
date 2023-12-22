from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:107, code: x = self.proj(x)
    l__mod___patch_embed_proj_0_0 = self.L__mod___patch_embed_proj_0_0(l_inputs_0_);  l_inputs_0_ = None
    l__mod___patch_embed_proj_0_1 = self.L__mod___patch_embed_proj_0_1(l__mod___patch_embed_proj_0_0);  l__mod___patch_embed_proj_0_0 = None
    l__mod___patch_embed_proj_1 = self.L__mod___patch_embed_proj_1(l__mod___patch_embed_proj_0_1);  l__mod___patch_embed_proj_0_1 = None
    l__mod___patch_embed_proj_2_0 = self.L__mod___patch_embed_proj_2_0(l__mod___patch_embed_proj_1);  l__mod___patch_embed_proj_1 = None
    l__mod___patch_embed_proj_2_1 = self.L__mod___patch_embed_proj_2_1(l__mod___patch_embed_proj_2_0);  l__mod___patch_embed_proj_2_0 = None
    l__mod___patch_embed_proj_3 = self.L__mod___patch_embed_proj_3(l__mod___patch_embed_proj_2_1);  l__mod___patch_embed_proj_2_1 = None
    l__mod___patch_embed_proj_4_0 = self.L__mod___patch_embed_proj_4_0(l__mod___patch_embed_proj_3);  l__mod___patch_embed_proj_3 = None
    x = self.L__mod___patch_embed_proj_4_1(l__mod___patch_embed_proj_4_0);  l__mod___patch_embed_proj_4_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:109, code: x = x.flatten(2).transpose(1, 2)  # (B, N, C)
    flatten = x.flatten(2);  x = None
    x_2 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:51, code: y_embed = torch.arange(1, H+1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
    arange = torch.arange(1, 29, dtype = torch.float32, device = device(type='cpu'))
    unsqueeze = arange.unsqueeze(1);  arange = None
    y_embed = unsqueeze.repeat(1, 1, 28);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:52, code: x_embed = torch.arange(1, W+1, dtype=torch.float32, device=device).repeat(1, H, 1)
    arange_1 = torch.arange(1, 29, dtype = torch.float32, device = device(type='cpu'))
    x_embed = arange_1.repeat(1, 28, 1);  arange_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:53, code: y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
    getitem = y_embed[(slice(None, None, None), slice(-1, None, None), slice(None, None, None))]
    add = getitem + 1e-06;  getitem = None
    truediv = y_embed / add;  y_embed = add = None
    y_embed_1 = truediv * 6.283185307179586;  truediv = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:54, code: x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
    getitem_1 = x_embed[(slice(None, None, None), slice(None, None, None), slice(-1, None, None))]
    add_1 = getitem_1 + 1e-06;  getitem_1 = None
    truediv_1 = x_embed / add_1;  x_embed = add_1 = None
    x_embed_1 = truediv_1 * 6.283185307179586;  truediv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:55, code: dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
    dim_t = torch.arange(32, dtype = torch.float32, device = device(type='cpu'))
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:56, code: dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)
    div = torch.div(dim_t, 2, rounding_mode = 'floor');  dim_t = None
    mul_2 = 2 * div;  div = None
    truediv_2 = mul_2 / 32;  mul_2 = None
    dim_t_1 = 10000 ** truediv_2;  truediv_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:57, code: pos_x = x_embed[:, :, :, None] / dim_t
    getitem_2 = x_embed_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), None)];  x_embed_1 = None
    pos_x = getitem_2 / dim_t_1;  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:58, code: pos_y = y_embed[:, :, :, None] / dim_t
    getitem_3 = y_embed_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), None)];  y_embed_1 = None
    pos_y = getitem_3 / dim_t_1;  getitem_3 = dim_t_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:59, code: pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
    getitem_4 = pos_x[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(0, None, 2))]
    sin = getitem_4.sin();  getitem_4 = None
    getitem_5 = pos_x[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  pos_x = None
    cos = getitem_5.cos();  getitem_5 = None
    stack = torch.stack([sin, cos], dim = 4);  sin = cos = None
    pos_x_1 = stack.flatten(3);  stack = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:60, code: pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
    getitem_6 = pos_y[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(0, None, 2))]
    sin_1 = getitem_6.sin();  getitem_6 = None
    getitem_7 = pos_y[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, 2))];  pos_y = None
    cos_1 = getitem_7.cos();  getitem_7 = None
    stack_1 = torch.stack([sin_1, cos_1], dim = 4);  sin_1 = cos_1 = None
    pos_y_1 = stack_1.flatten(3);  stack_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:61, code: pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    cat = torch.cat((pos_y_1, pos_x_1), dim = 3);  pos_y_1 = pos_x_1 = None
    pos = cat.permute(0, 3, 1, 2);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:62, code: pos = self.token_projection(pos)
    pos_1 = self.L__mod___pos_embed_token_projection(pos);  pos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:63, code: return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)
    repeat_2 = pos_1.repeat(8, 1, 1, 1);  pos_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:437, code: pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
    reshape = repeat_2.reshape(8, -1, 784);  repeat_2 = None
    pos_encoding = reshape.permute(0, 2, 1);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:438, code: x = x + pos_encoding
    x_3 = x_2 + pos_encoding;  x_2 = pos_encoding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:439, code: x = self.pos_drop(x)
    x_4 = self.L__mod___pos_drop(x_3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_0_gamma1 = self.L__mod___blocks_0_gamma1
    l__mod___blocks_0_norm1 = self.L__mod___blocks_0_norm1(x_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_0_attn_qkv = self.L__mod___blocks_0_attn_qkv(l__mod___blocks_0_norm1);  l__mod___blocks_0_norm1 = None
    reshape_1 = l__mod___blocks_0_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_0_attn_qkv = None
    qkv = reshape_1.permute(2, 0, 3, 4, 1);  reshape_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_1 = torch.nn.functional.normalize(q, dim = -1);  q = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_1 = torch.nn.functional.normalize(k, dim = -1);  k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_1 = k_1.transpose(-2, -1);  k_1 = None
    matmul = q_1 @ transpose_1;  q_1 = transpose_1 = None
    l__mod___blocks_0_attn_temperature = self.L__mod___blocks_0_attn_temperature
    attn = matmul * l__mod___blocks_0_attn_temperature;  matmul = l__mod___blocks_0_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_1 = attn.softmax(dim = -1);  attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_2 = self.L__mod___blocks_0_attn_attn_drop(attn_1);  attn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_1 = attn_2 @ v;  attn_2 = v = None
    permute_3 = matmul_1.permute(0, 3, 1, 2);  matmul_1 = None
    x_5 = permute_3.reshape(8, 784, 768);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_6 = self.L__mod___blocks_0_attn_proj(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_7 = self.L__mod___blocks_0_attn_proj_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_4 = l__mod___blocks_0_gamma1 * x_7;  l__mod___blocks_0_gamma1 = x_7 = None
    l__mod___blocks_0_drop_path = self.L__mod___blocks_0_drop_path(mul_4);  mul_4 = None
    x_8 = x_4 + l__mod___blocks_0_drop_path;  x_4 = l__mod___blocks_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_0_gamma3 = self.L__mod___blocks_0_gamma3
    l__mod___blocks_0_norm3 = self.L__mod___blocks_0_norm3(x_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_4 = l__mod___blocks_0_norm3.permute(0, 2, 1);  l__mod___blocks_0_norm3 = None
    x_9 = permute_4.reshape(8, 768, 28, 28);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_10 = self.L__mod___blocks_0_local_mp_conv1(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_11 = self.L__mod___blocks_0_local_mp_act(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_12 = self.L__mod___blocks_0_local_mp_bn(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_13 = self.L__mod___blocks_0_local_mp_conv2(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_4 = x_13.reshape(8, 768, 784);  x_13 = None
    x_14 = reshape_4.permute(0, 2, 1);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_5 = l__mod___blocks_0_gamma3 * x_14;  l__mod___blocks_0_gamma3 = x_14 = None
    l__mod___blocks_0_drop_path_1 = self.L__mod___blocks_0_drop_path(mul_5);  mul_5 = None
    x_15 = x_8 + l__mod___blocks_0_drop_path_1;  x_8 = l__mod___blocks_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_0_gamma2 = self.L__mod___blocks_0_gamma2
    l__mod___blocks_0_norm2 = self.L__mod___blocks_0_norm2(x_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_16 = self.L__mod___blocks_0_mlp_fc1(l__mod___blocks_0_norm2);  l__mod___blocks_0_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_17 = self.L__mod___blocks_0_mlp_act(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_18 = self.L__mod___blocks_0_mlp_drop1(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_19 = self.L__mod___blocks_0_mlp_norm(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_20 = self.L__mod___blocks_0_mlp_fc2(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_21 = self.L__mod___blocks_0_mlp_drop2(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_6 = l__mod___blocks_0_gamma2 * x_21;  l__mod___blocks_0_gamma2 = x_21 = None
    l__mod___blocks_0_drop_path_2 = self.L__mod___blocks_0_drop_path(mul_6);  mul_6 = None
    x_23 = x_15 + l__mod___blocks_0_drop_path_2;  x_15 = l__mod___blocks_0_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_1_gamma1 = self.L__mod___blocks_1_gamma1
    l__mod___blocks_1_norm1 = self.L__mod___blocks_1_norm1(x_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_1_attn_qkv = self.L__mod___blocks_1_attn_qkv(l__mod___blocks_1_norm1);  l__mod___blocks_1_norm1 = None
    reshape_5 = l__mod___blocks_1_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_1_attn_qkv = None
    qkv_1 = reshape_5.permute(2, 0, 3, 4, 1);  reshape_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_2 = unbind_1[0]
    k_2 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_3 = torch.nn.functional.normalize(q_2, dim = -1);  q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_3 = torch.nn.functional.normalize(k_2, dim = -1);  k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_2 = k_3.transpose(-2, -1);  k_3 = None
    matmul_2 = q_3 @ transpose_2;  q_3 = transpose_2 = None
    l__mod___blocks_1_attn_temperature = self.L__mod___blocks_1_attn_temperature
    attn_3 = matmul_2 * l__mod___blocks_1_attn_temperature;  matmul_2 = l__mod___blocks_1_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_4 = attn_3.softmax(dim = -1);  attn_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_5 = self.L__mod___blocks_1_attn_attn_drop(attn_4);  attn_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_3 = attn_5 @ v_1;  attn_5 = v_1 = None
    permute_7 = matmul_3.permute(0, 3, 1, 2);  matmul_3 = None
    x_24 = permute_7.reshape(8, 784, 768);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_25 = self.L__mod___blocks_1_attn_proj(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_26 = self.L__mod___blocks_1_attn_proj_drop(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_8 = l__mod___blocks_1_gamma1 * x_26;  l__mod___blocks_1_gamma1 = x_26 = None
    l__mod___blocks_1_drop_path = self.L__mod___blocks_1_drop_path(mul_8);  mul_8 = None
    x_27 = x_23 + l__mod___blocks_1_drop_path;  x_23 = l__mod___blocks_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_1_gamma3 = self.L__mod___blocks_1_gamma3
    l__mod___blocks_1_norm3 = self.L__mod___blocks_1_norm3(x_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_8 = l__mod___blocks_1_norm3.permute(0, 2, 1);  l__mod___blocks_1_norm3 = None
    x_28 = permute_8.reshape(8, 768, 28, 28);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_29 = self.L__mod___blocks_1_local_mp_conv1(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_30 = self.L__mod___blocks_1_local_mp_act(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_31 = self.L__mod___blocks_1_local_mp_bn(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_32 = self.L__mod___blocks_1_local_mp_conv2(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_8 = x_32.reshape(8, 768, 784);  x_32 = None
    x_33 = reshape_8.permute(0, 2, 1);  reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_9 = l__mod___blocks_1_gamma3 * x_33;  l__mod___blocks_1_gamma3 = x_33 = None
    l__mod___blocks_1_drop_path_1 = self.L__mod___blocks_1_drop_path(mul_9);  mul_9 = None
    x_34 = x_27 + l__mod___blocks_1_drop_path_1;  x_27 = l__mod___blocks_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_1_gamma2 = self.L__mod___blocks_1_gamma2
    l__mod___blocks_1_norm2 = self.L__mod___blocks_1_norm2(x_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_35 = self.L__mod___blocks_1_mlp_fc1(l__mod___blocks_1_norm2);  l__mod___blocks_1_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_36 = self.L__mod___blocks_1_mlp_act(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_37 = self.L__mod___blocks_1_mlp_drop1(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_38 = self.L__mod___blocks_1_mlp_norm(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_39 = self.L__mod___blocks_1_mlp_fc2(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_40 = self.L__mod___blocks_1_mlp_drop2(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_10 = l__mod___blocks_1_gamma2 * x_40;  l__mod___blocks_1_gamma2 = x_40 = None
    l__mod___blocks_1_drop_path_2 = self.L__mod___blocks_1_drop_path(mul_10);  mul_10 = None
    x_42 = x_34 + l__mod___blocks_1_drop_path_2;  x_34 = l__mod___blocks_1_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_2_gamma1 = self.L__mod___blocks_2_gamma1
    l__mod___blocks_2_norm1 = self.L__mod___blocks_2_norm1(x_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_2_attn_qkv = self.L__mod___blocks_2_attn_qkv(l__mod___blocks_2_norm1);  l__mod___blocks_2_norm1 = None
    reshape_9 = l__mod___blocks_2_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_2_attn_qkv = None
    qkv_2 = reshape_9.permute(2, 0, 3, 4, 1);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_4 = unbind_2[0]
    k_4 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_5 = torch.nn.functional.normalize(q_4, dim = -1);  q_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_5 = torch.nn.functional.normalize(k_4, dim = -1);  k_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_3 = k_5.transpose(-2, -1);  k_5 = None
    matmul_4 = q_5 @ transpose_3;  q_5 = transpose_3 = None
    l__mod___blocks_2_attn_temperature = self.L__mod___blocks_2_attn_temperature
    attn_6 = matmul_4 * l__mod___blocks_2_attn_temperature;  matmul_4 = l__mod___blocks_2_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_7 = attn_6.softmax(dim = -1);  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_8 = self.L__mod___blocks_2_attn_attn_drop(attn_7);  attn_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_5 = attn_8 @ v_2;  attn_8 = v_2 = None
    permute_11 = matmul_5.permute(0, 3, 1, 2);  matmul_5 = None
    x_43 = permute_11.reshape(8, 784, 768);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_44 = self.L__mod___blocks_2_attn_proj(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_45 = self.L__mod___blocks_2_attn_proj_drop(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_12 = l__mod___blocks_2_gamma1 * x_45;  l__mod___blocks_2_gamma1 = x_45 = None
    l__mod___blocks_2_drop_path = self.L__mod___blocks_2_drop_path(mul_12);  mul_12 = None
    x_46 = x_42 + l__mod___blocks_2_drop_path;  x_42 = l__mod___blocks_2_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_2_gamma3 = self.L__mod___blocks_2_gamma3
    l__mod___blocks_2_norm3 = self.L__mod___blocks_2_norm3(x_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_12 = l__mod___blocks_2_norm3.permute(0, 2, 1);  l__mod___blocks_2_norm3 = None
    x_47 = permute_12.reshape(8, 768, 28, 28);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_48 = self.L__mod___blocks_2_local_mp_conv1(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_49 = self.L__mod___blocks_2_local_mp_act(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_50 = self.L__mod___blocks_2_local_mp_bn(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_51 = self.L__mod___blocks_2_local_mp_conv2(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_12 = x_51.reshape(8, 768, 784);  x_51 = None
    x_52 = reshape_12.permute(0, 2, 1);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_13 = l__mod___blocks_2_gamma3 * x_52;  l__mod___blocks_2_gamma3 = x_52 = None
    l__mod___blocks_2_drop_path_1 = self.L__mod___blocks_2_drop_path(mul_13);  mul_13 = None
    x_53 = x_46 + l__mod___blocks_2_drop_path_1;  x_46 = l__mod___blocks_2_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_2_gamma2 = self.L__mod___blocks_2_gamma2
    l__mod___blocks_2_norm2 = self.L__mod___blocks_2_norm2(x_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_54 = self.L__mod___blocks_2_mlp_fc1(l__mod___blocks_2_norm2);  l__mod___blocks_2_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_55 = self.L__mod___blocks_2_mlp_act(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_56 = self.L__mod___blocks_2_mlp_drop1(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_57 = self.L__mod___blocks_2_mlp_norm(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_58 = self.L__mod___blocks_2_mlp_fc2(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_59 = self.L__mod___blocks_2_mlp_drop2(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_14 = l__mod___blocks_2_gamma2 * x_59;  l__mod___blocks_2_gamma2 = x_59 = None
    l__mod___blocks_2_drop_path_2 = self.L__mod___blocks_2_drop_path(mul_14);  mul_14 = None
    x_61 = x_53 + l__mod___blocks_2_drop_path_2;  x_53 = l__mod___blocks_2_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_3_gamma1 = self.L__mod___blocks_3_gamma1
    l__mod___blocks_3_norm1 = self.L__mod___blocks_3_norm1(x_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_3_attn_qkv = self.L__mod___blocks_3_attn_qkv(l__mod___blocks_3_norm1);  l__mod___blocks_3_norm1 = None
    reshape_13 = l__mod___blocks_3_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_3_attn_qkv = None
    qkv_3 = reshape_13.permute(2, 0, 3, 4, 1);  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_6 = unbind_3[0]
    k_6 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_7 = torch.nn.functional.normalize(q_6, dim = -1);  q_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_7 = torch.nn.functional.normalize(k_6, dim = -1);  k_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_4 = k_7.transpose(-2, -1);  k_7 = None
    matmul_6 = q_7 @ transpose_4;  q_7 = transpose_4 = None
    l__mod___blocks_3_attn_temperature = self.L__mod___blocks_3_attn_temperature
    attn_9 = matmul_6 * l__mod___blocks_3_attn_temperature;  matmul_6 = l__mod___blocks_3_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_10 = attn_9.softmax(dim = -1);  attn_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_11 = self.L__mod___blocks_3_attn_attn_drop(attn_10);  attn_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_7 = attn_11 @ v_3;  attn_11 = v_3 = None
    permute_15 = matmul_7.permute(0, 3, 1, 2);  matmul_7 = None
    x_62 = permute_15.reshape(8, 784, 768);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_63 = self.L__mod___blocks_3_attn_proj(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_64 = self.L__mod___blocks_3_attn_proj_drop(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_16 = l__mod___blocks_3_gamma1 * x_64;  l__mod___blocks_3_gamma1 = x_64 = None
    l__mod___blocks_3_drop_path = self.L__mod___blocks_3_drop_path(mul_16);  mul_16 = None
    x_65 = x_61 + l__mod___blocks_3_drop_path;  x_61 = l__mod___blocks_3_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_3_gamma3 = self.L__mod___blocks_3_gamma3
    l__mod___blocks_3_norm3 = self.L__mod___blocks_3_norm3(x_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_16 = l__mod___blocks_3_norm3.permute(0, 2, 1);  l__mod___blocks_3_norm3 = None
    x_66 = permute_16.reshape(8, 768, 28, 28);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_67 = self.L__mod___blocks_3_local_mp_conv1(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_68 = self.L__mod___blocks_3_local_mp_act(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_69 = self.L__mod___blocks_3_local_mp_bn(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_70 = self.L__mod___blocks_3_local_mp_conv2(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_16 = x_70.reshape(8, 768, 784);  x_70 = None
    x_71 = reshape_16.permute(0, 2, 1);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_17 = l__mod___blocks_3_gamma3 * x_71;  l__mod___blocks_3_gamma3 = x_71 = None
    l__mod___blocks_3_drop_path_1 = self.L__mod___blocks_3_drop_path(mul_17);  mul_17 = None
    x_72 = x_65 + l__mod___blocks_3_drop_path_1;  x_65 = l__mod___blocks_3_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_3_gamma2 = self.L__mod___blocks_3_gamma2
    l__mod___blocks_3_norm2 = self.L__mod___blocks_3_norm2(x_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_73 = self.L__mod___blocks_3_mlp_fc1(l__mod___blocks_3_norm2);  l__mod___blocks_3_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_74 = self.L__mod___blocks_3_mlp_act(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_75 = self.L__mod___blocks_3_mlp_drop1(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_76 = self.L__mod___blocks_3_mlp_norm(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_77 = self.L__mod___blocks_3_mlp_fc2(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_78 = self.L__mod___blocks_3_mlp_drop2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_18 = l__mod___blocks_3_gamma2 * x_78;  l__mod___blocks_3_gamma2 = x_78 = None
    l__mod___blocks_3_drop_path_2 = self.L__mod___blocks_3_drop_path(mul_18);  mul_18 = None
    x_80 = x_72 + l__mod___blocks_3_drop_path_2;  x_72 = l__mod___blocks_3_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_4_gamma1 = self.L__mod___blocks_4_gamma1
    l__mod___blocks_4_norm1 = self.L__mod___blocks_4_norm1(x_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_4_attn_qkv = self.L__mod___blocks_4_attn_qkv(l__mod___blocks_4_norm1);  l__mod___blocks_4_norm1 = None
    reshape_17 = l__mod___blocks_4_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_4_attn_qkv = None
    qkv_4 = reshape_17.permute(2, 0, 3, 4, 1);  reshape_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_8 = unbind_4[0]
    k_8 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_9 = torch.nn.functional.normalize(q_8, dim = -1);  q_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_9 = torch.nn.functional.normalize(k_8, dim = -1);  k_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_5 = k_9.transpose(-2, -1);  k_9 = None
    matmul_8 = q_9 @ transpose_5;  q_9 = transpose_5 = None
    l__mod___blocks_4_attn_temperature = self.L__mod___blocks_4_attn_temperature
    attn_12 = matmul_8 * l__mod___blocks_4_attn_temperature;  matmul_8 = l__mod___blocks_4_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_13 = attn_12.softmax(dim = -1);  attn_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_14 = self.L__mod___blocks_4_attn_attn_drop(attn_13);  attn_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_9 = attn_14 @ v_4;  attn_14 = v_4 = None
    permute_19 = matmul_9.permute(0, 3, 1, 2);  matmul_9 = None
    x_81 = permute_19.reshape(8, 784, 768);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_82 = self.L__mod___blocks_4_attn_proj(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_83 = self.L__mod___blocks_4_attn_proj_drop(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_20 = l__mod___blocks_4_gamma1 * x_83;  l__mod___blocks_4_gamma1 = x_83 = None
    l__mod___blocks_4_drop_path = self.L__mod___blocks_4_drop_path(mul_20);  mul_20 = None
    x_84 = x_80 + l__mod___blocks_4_drop_path;  x_80 = l__mod___blocks_4_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_4_gamma3 = self.L__mod___blocks_4_gamma3
    l__mod___blocks_4_norm3 = self.L__mod___blocks_4_norm3(x_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_20 = l__mod___blocks_4_norm3.permute(0, 2, 1);  l__mod___blocks_4_norm3 = None
    x_85 = permute_20.reshape(8, 768, 28, 28);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_86 = self.L__mod___blocks_4_local_mp_conv1(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_87 = self.L__mod___blocks_4_local_mp_act(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_88 = self.L__mod___blocks_4_local_mp_bn(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_89 = self.L__mod___blocks_4_local_mp_conv2(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_20 = x_89.reshape(8, 768, 784);  x_89 = None
    x_90 = reshape_20.permute(0, 2, 1);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_21 = l__mod___blocks_4_gamma3 * x_90;  l__mod___blocks_4_gamma3 = x_90 = None
    l__mod___blocks_4_drop_path_1 = self.L__mod___blocks_4_drop_path(mul_21);  mul_21 = None
    x_91 = x_84 + l__mod___blocks_4_drop_path_1;  x_84 = l__mod___blocks_4_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_4_gamma2 = self.L__mod___blocks_4_gamma2
    l__mod___blocks_4_norm2 = self.L__mod___blocks_4_norm2(x_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_92 = self.L__mod___blocks_4_mlp_fc1(l__mod___blocks_4_norm2);  l__mod___blocks_4_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_93 = self.L__mod___blocks_4_mlp_act(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_94 = self.L__mod___blocks_4_mlp_drop1(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_95 = self.L__mod___blocks_4_mlp_norm(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_96 = self.L__mod___blocks_4_mlp_fc2(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_97 = self.L__mod___blocks_4_mlp_drop2(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_22 = l__mod___blocks_4_gamma2 * x_97;  l__mod___blocks_4_gamma2 = x_97 = None
    l__mod___blocks_4_drop_path_2 = self.L__mod___blocks_4_drop_path(mul_22);  mul_22 = None
    x_99 = x_91 + l__mod___blocks_4_drop_path_2;  x_91 = l__mod___blocks_4_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_5_gamma1 = self.L__mod___blocks_5_gamma1
    l__mod___blocks_5_norm1 = self.L__mod___blocks_5_norm1(x_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_5_attn_qkv = self.L__mod___blocks_5_attn_qkv(l__mod___blocks_5_norm1);  l__mod___blocks_5_norm1 = None
    reshape_21 = l__mod___blocks_5_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_5_attn_qkv = None
    qkv_5 = reshape_21.permute(2, 0, 3, 4, 1);  reshape_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_10 = unbind_5[0]
    k_10 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_11 = torch.nn.functional.normalize(q_10, dim = -1);  q_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_11 = torch.nn.functional.normalize(k_10, dim = -1);  k_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_6 = k_11.transpose(-2, -1);  k_11 = None
    matmul_10 = q_11 @ transpose_6;  q_11 = transpose_6 = None
    l__mod___blocks_5_attn_temperature = self.L__mod___blocks_5_attn_temperature
    attn_15 = matmul_10 * l__mod___blocks_5_attn_temperature;  matmul_10 = l__mod___blocks_5_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_16 = attn_15.softmax(dim = -1);  attn_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_17 = self.L__mod___blocks_5_attn_attn_drop(attn_16);  attn_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_11 = attn_17 @ v_5;  attn_17 = v_5 = None
    permute_23 = matmul_11.permute(0, 3, 1, 2);  matmul_11 = None
    x_100 = permute_23.reshape(8, 784, 768);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_101 = self.L__mod___blocks_5_attn_proj(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_102 = self.L__mod___blocks_5_attn_proj_drop(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_24 = l__mod___blocks_5_gamma1 * x_102;  l__mod___blocks_5_gamma1 = x_102 = None
    l__mod___blocks_5_drop_path = self.L__mod___blocks_5_drop_path(mul_24);  mul_24 = None
    x_103 = x_99 + l__mod___blocks_5_drop_path;  x_99 = l__mod___blocks_5_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_5_gamma3 = self.L__mod___blocks_5_gamma3
    l__mod___blocks_5_norm3 = self.L__mod___blocks_5_norm3(x_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_24 = l__mod___blocks_5_norm3.permute(0, 2, 1);  l__mod___blocks_5_norm3 = None
    x_104 = permute_24.reshape(8, 768, 28, 28);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_105 = self.L__mod___blocks_5_local_mp_conv1(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_106 = self.L__mod___blocks_5_local_mp_act(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_107 = self.L__mod___blocks_5_local_mp_bn(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_108 = self.L__mod___blocks_5_local_mp_conv2(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_24 = x_108.reshape(8, 768, 784);  x_108 = None
    x_109 = reshape_24.permute(0, 2, 1);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_25 = l__mod___blocks_5_gamma3 * x_109;  l__mod___blocks_5_gamma3 = x_109 = None
    l__mod___blocks_5_drop_path_1 = self.L__mod___blocks_5_drop_path(mul_25);  mul_25 = None
    x_110 = x_103 + l__mod___blocks_5_drop_path_1;  x_103 = l__mod___blocks_5_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_5_gamma2 = self.L__mod___blocks_5_gamma2
    l__mod___blocks_5_norm2 = self.L__mod___blocks_5_norm2(x_110)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_111 = self.L__mod___blocks_5_mlp_fc1(l__mod___blocks_5_norm2);  l__mod___blocks_5_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_112 = self.L__mod___blocks_5_mlp_act(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_113 = self.L__mod___blocks_5_mlp_drop1(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_114 = self.L__mod___blocks_5_mlp_norm(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_115 = self.L__mod___blocks_5_mlp_fc2(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_116 = self.L__mod___blocks_5_mlp_drop2(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_26 = l__mod___blocks_5_gamma2 * x_116;  l__mod___blocks_5_gamma2 = x_116 = None
    l__mod___blocks_5_drop_path_2 = self.L__mod___blocks_5_drop_path(mul_26);  mul_26 = None
    x_118 = x_110 + l__mod___blocks_5_drop_path_2;  x_110 = l__mod___blocks_5_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_6_gamma1 = self.L__mod___blocks_6_gamma1
    l__mod___blocks_6_norm1 = self.L__mod___blocks_6_norm1(x_118)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_6_attn_qkv = self.L__mod___blocks_6_attn_qkv(l__mod___blocks_6_norm1);  l__mod___blocks_6_norm1 = None
    reshape_25 = l__mod___blocks_6_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_6_attn_qkv = None
    qkv_6 = reshape_25.permute(2, 0, 3, 4, 1);  reshape_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_12 = unbind_6[0]
    k_12 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_13 = torch.nn.functional.normalize(q_12, dim = -1);  q_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_13 = torch.nn.functional.normalize(k_12, dim = -1);  k_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_7 = k_13.transpose(-2, -1);  k_13 = None
    matmul_12 = q_13 @ transpose_7;  q_13 = transpose_7 = None
    l__mod___blocks_6_attn_temperature = self.L__mod___blocks_6_attn_temperature
    attn_18 = matmul_12 * l__mod___blocks_6_attn_temperature;  matmul_12 = l__mod___blocks_6_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_19 = attn_18.softmax(dim = -1);  attn_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_20 = self.L__mod___blocks_6_attn_attn_drop(attn_19);  attn_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_13 = attn_20 @ v_6;  attn_20 = v_6 = None
    permute_27 = matmul_13.permute(0, 3, 1, 2);  matmul_13 = None
    x_119 = permute_27.reshape(8, 784, 768);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_120 = self.L__mod___blocks_6_attn_proj(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_121 = self.L__mod___blocks_6_attn_proj_drop(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_28 = l__mod___blocks_6_gamma1 * x_121;  l__mod___blocks_6_gamma1 = x_121 = None
    l__mod___blocks_6_drop_path = self.L__mod___blocks_6_drop_path(mul_28);  mul_28 = None
    x_122 = x_118 + l__mod___blocks_6_drop_path;  x_118 = l__mod___blocks_6_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_6_gamma3 = self.L__mod___blocks_6_gamma3
    l__mod___blocks_6_norm3 = self.L__mod___blocks_6_norm3(x_122)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_28 = l__mod___blocks_6_norm3.permute(0, 2, 1);  l__mod___blocks_6_norm3 = None
    x_123 = permute_28.reshape(8, 768, 28, 28);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_124 = self.L__mod___blocks_6_local_mp_conv1(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_125 = self.L__mod___blocks_6_local_mp_act(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_126 = self.L__mod___blocks_6_local_mp_bn(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_127 = self.L__mod___blocks_6_local_mp_conv2(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_28 = x_127.reshape(8, 768, 784);  x_127 = None
    x_128 = reshape_28.permute(0, 2, 1);  reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_29 = l__mod___blocks_6_gamma3 * x_128;  l__mod___blocks_6_gamma3 = x_128 = None
    l__mod___blocks_6_drop_path_1 = self.L__mod___blocks_6_drop_path(mul_29);  mul_29 = None
    x_129 = x_122 + l__mod___blocks_6_drop_path_1;  x_122 = l__mod___blocks_6_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_6_gamma2 = self.L__mod___blocks_6_gamma2
    l__mod___blocks_6_norm2 = self.L__mod___blocks_6_norm2(x_129)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_130 = self.L__mod___blocks_6_mlp_fc1(l__mod___blocks_6_norm2);  l__mod___blocks_6_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_131 = self.L__mod___blocks_6_mlp_act(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_132 = self.L__mod___blocks_6_mlp_drop1(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_133 = self.L__mod___blocks_6_mlp_norm(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_134 = self.L__mod___blocks_6_mlp_fc2(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_135 = self.L__mod___blocks_6_mlp_drop2(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_30 = l__mod___blocks_6_gamma2 * x_135;  l__mod___blocks_6_gamma2 = x_135 = None
    l__mod___blocks_6_drop_path_2 = self.L__mod___blocks_6_drop_path(mul_30);  mul_30 = None
    x_137 = x_129 + l__mod___blocks_6_drop_path_2;  x_129 = l__mod___blocks_6_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_7_gamma1 = self.L__mod___blocks_7_gamma1
    l__mod___blocks_7_norm1 = self.L__mod___blocks_7_norm1(x_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_7_attn_qkv = self.L__mod___blocks_7_attn_qkv(l__mod___blocks_7_norm1);  l__mod___blocks_7_norm1 = None
    reshape_29 = l__mod___blocks_7_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_7_attn_qkv = None
    qkv_7 = reshape_29.permute(2, 0, 3, 4, 1);  reshape_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_14 = unbind_7[0]
    k_14 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_15 = torch.nn.functional.normalize(q_14, dim = -1);  q_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_15 = torch.nn.functional.normalize(k_14, dim = -1);  k_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_8 = k_15.transpose(-2, -1);  k_15 = None
    matmul_14 = q_15 @ transpose_8;  q_15 = transpose_8 = None
    l__mod___blocks_7_attn_temperature = self.L__mod___blocks_7_attn_temperature
    attn_21 = matmul_14 * l__mod___blocks_7_attn_temperature;  matmul_14 = l__mod___blocks_7_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_22 = attn_21.softmax(dim = -1);  attn_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_23 = self.L__mod___blocks_7_attn_attn_drop(attn_22);  attn_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_15 = attn_23 @ v_7;  attn_23 = v_7 = None
    permute_31 = matmul_15.permute(0, 3, 1, 2);  matmul_15 = None
    x_138 = permute_31.reshape(8, 784, 768);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_139 = self.L__mod___blocks_7_attn_proj(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_140 = self.L__mod___blocks_7_attn_proj_drop(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_32 = l__mod___blocks_7_gamma1 * x_140;  l__mod___blocks_7_gamma1 = x_140 = None
    l__mod___blocks_7_drop_path = self.L__mod___blocks_7_drop_path(mul_32);  mul_32 = None
    x_141 = x_137 + l__mod___blocks_7_drop_path;  x_137 = l__mod___blocks_7_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_7_gamma3 = self.L__mod___blocks_7_gamma3
    l__mod___blocks_7_norm3 = self.L__mod___blocks_7_norm3(x_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_32 = l__mod___blocks_7_norm3.permute(0, 2, 1);  l__mod___blocks_7_norm3 = None
    x_142 = permute_32.reshape(8, 768, 28, 28);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_143 = self.L__mod___blocks_7_local_mp_conv1(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_144 = self.L__mod___blocks_7_local_mp_act(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_145 = self.L__mod___blocks_7_local_mp_bn(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_146 = self.L__mod___blocks_7_local_mp_conv2(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_32 = x_146.reshape(8, 768, 784);  x_146 = None
    x_147 = reshape_32.permute(0, 2, 1);  reshape_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_33 = l__mod___blocks_7_gamma3 * x_147;  l__mod___blocks_7_gamma3 = x_147 = None
    l__mod___blocks_7_drop_path_1 = self.L__mod___blocks_7_drop_path(mul_33);  mul_33 = None
    x_148 = x_141 + l__mod___blocks_7_drop_path_1;  x_141 = l__mod___blocks_7_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_7_gamma2 = self.L__mod___blocks_7_gamma2
    l__mod___blocks_7_norm2 = self.L__mod___blocks_7_norm2(x_148)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_149 = self.L__mod___blocks_7_mlp_fc1(l__mod___blocks_7_norm2);  l__mod___blocks_7_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_150 = self.L__mod___blocks_7_mlp_act(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_151 = self.L__mod___blocks_7_mlp_drop1(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_152 = self.L__mod___blocks_7_mlp_norm(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_153 = self.L__mod___blocks_7_mlp_fc2(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_154 = self.L__mod___blocks_7_mlp_drop2(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_34 = l__mod___blocks_7_gamma2 * x_154;  l__mod___blocks_7_gamma2 = x_154 = None
    l__mod___blocks_7_drop_path_2 = self.L__mod___blocks_7_drop_path(mul_34);  mul_34 = None
    x_156 = x_148 + l__mod___blocks_7_drop_path_2;  x_148 = l__mod___blocks_7_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_8_gamma1 = self.L__mod___blocks_8_gamma1
    l__mod___blocks_8_norm1 = self.L__mod___blocks_8_norm1(x_156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_8_attn_qkv = self.L__mod___blocks_8_attn_qkv(l__mod___blocks_8_norm1);  l__mod___blocks_8_norm1 = None
    reshape_33 = l__mod___blocks_8_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_8_attn_qkv = None
    qkv_8 = reshape_33.permute(2, 0, 3, 4, 1);  reshape_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = qkv_8.unbind(0);  qkv_8 = None
    q_16 = unbind_8[0]
    k_16 = unbind_8[1]
    v_8 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_17 = torch.nn.functional.normalize(q_16, dim = -1);  q_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_17 = torch.nn.functional.normalize(k_16, dim = -1);  k_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_9 = k_17.transpose(-2, -1);  k_17 = None
    matmul_16 = q_17 @ transpose_9;  q_17 = transpose_9 = None
    l__mod___blocks_8_attn_temperature = self.L__mod___blocks_8_attn_temperature
    attn_24 = matmul_16 * l__mod___blocks_8_attn_temperature;  matmul_16 = l__mod___blocks_8_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_25 = attn_24.softmax(dim = -1);  attn_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_26 = self.L__mod___blocks_8_attn_attn_drop(attn_25);  attn_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_17 = attn_26 @ v_8;  attn_26 = v_8 = None
    permute_35 = matmul_17.permute(0, 3, 1, 2);  matmul_17 = None
    x_157 = permute_35.reshape(8, 784, 768);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_158 = self.L__mod___blocks_8_attn_proj(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_159 = self.L__mod___blocks_8_attn_proj_drop(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_36 = l__mod___blocks_8_gamma1 * x_159;  l__mod___blocks_8_gamma1 = x_159 = None
    l__mod___blocks_8_drop_path = self.L__mod___blocks_8_drop_path(mul_36);  mul_36 = None
    x_160 = x_156 + l__mod___blocks_8_drop_path;  x_156 = l__mod___blocks_8_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_8_gamma3 = self.L__mod___blocks_8_gamma3
    l__mod___blocks_8_norm3 = self.L__mod___blocks_8_norm3(x_160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_36 = l__mod___blocks_8_norm3.permute(0, 2, 1);  l__mod___blocks_8_norm3 = None
    x_161 = permute_36.reshape(8, 768, 28, 28);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_162 = self.L__mod___blocks_8_local_mp_conv1(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_163 = self.L__mod___blocks_8_local_mp_act(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_164 = self.L__mod___blocks_8_local_mp_bn(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_165 = self.L__mod___blocks_8_local_mp_conv2(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_36 = x_165.reshape(8, 768, 784);  x_165 = None
    x_166 = reshape_36.permute(0, 2, 1);  reshape_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_37 = l__mod___blocks_8_gamma3 * x_166;  l__mod___blocks_8_gamma3 = x_166 = None
    l__mod___blocks_8_drop_path_1 = self.L__mod___blocks_8_drop_path(mul_37);  mul_37 = None
    x_167 = x_160 + l__mod___blocks_8_drop_path_1;  x_160 = l__mod___blocks_8_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_8_gamma2 = self.L__mod___blocks_8_gamma2
    l__mod___blocks_8_norm2 = self.L__mod___blocks_8_norm2(x_167)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_168 = self.L__mod___blocks_8_mlp_fc1(l__mod___blocks_8_norm2);  l__mod___blocks_8_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_169 = self.L__mod___blocks_8_mlp_act(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_170 = self.L__mod___blocks_8_mlp_drop1(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_171 = self.L__mod___blocks_8_mlp_norm(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_172 = self.L__mod___blocks_8_mlp_fc2(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_173 = self.L__mod___blocks_8_mlp_drop2(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_38 = l__mod___blocks_8_gamma2 * x_173;  l__mod___blocks_8_gamma2 = x_173 = None
    l__mod___blocks_8_drop_path_2 = self.L__mod___blocks_8_drop_path(mul_38);  mul_38 = None
    x_175 = x_167 + l__mod___blocks_8_drop_path_2;  x_167 = l__mod___blocks_8_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_9_gamma1 = self.L__mod___blocks_9_gamma1
    l__mod___blocks_9_norm1 = self.L__mod___blocks_9_norm1(x_175)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_9_attn_qkv = self.L__mod___blocks_9_attn_qkv(l__mod___blocks_9_norm1);  l__mod___blocks_9_norm1 = None
    reshape_37 = l__mod___blocks_9_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_9_attn_qkv = None
    qkv_9 = reshape_37.permute(2, 0, 3, 4, 1);  reshape_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = qkv_9.unbind(0);  qkv_9 = None
    q_18 = unbind_9[0]
    k_18 = unbind_9[1]
    v_9 = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_19 = torch.nn.functional.normalize(q_18, dim = -1);  q_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_19 = torch.nn.functional.normalize(k_18, dim = -1);  k_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_10 = k_19.transpose(-2, -1);  k_19 = None
    matmul_18 = q_19 @ transpose_10;  q_19 = transpose_10 = None
    l__mod___blocks_9_attn_temperature = self.L__mod___blocks_9_attn_temperature
    attn_27 = matmul_18 * l__mod___blocks_9_attn_temperature;  matmul_18 = l__mod___blocks_9_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_28 = attn_27.softmax(dim = -1);  attn_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_29 = self.L__mod___blocks_9_attn_attn_drop(attn_28);  attn_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_19 = attn_29 @ v_9;  attn_29 = v_9 = None
    permute_39 = matmul_19.permute(0, 3, 1, 2);  matmul_19 = None
    x_176 = permute_39.reshape(8, 784, 768);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_177 = self.L__mod___blocks_9_attn_proj(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_178 = self.L__mod___blocks_9_attn_proj_drop(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_40 = l__mod___blocks_9_gamma1 * x_178;  l__mod___blocks_9_gamma1 = x_178 = None
    l__mod___blocks_9_drop_path = self.L__mod___blocks_9_drop_path(mul_40);  mul_40 = None
    x_179 = x_175 + l__mod___blocks_9_drop_path;  x_175 = l__mod___blocks_9_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_9_gamma3 = self.L__mod___blocks_9_gamma3
    l__mod___blocks_9_norm3 = self.L__mod___blocks_9_norm3(x_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_40 = l__mod___blocks_9_norm3.permute(0, 2, 1);  l__mod___blocks_9_norm3 = None
    x_180 = permute_40.reshape(8, 768, 28, 28);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_181 = self.L__mod___blocks_9_local_mp_conv1(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_182 = self.L__mod___blocks_9_local_mp_act(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_183 = self.L__mod___blocks_9_local_mp_bn(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_184 = self.L__mod___blocks_9_local_mp_conv2(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_40 = x_184.reshape(8, 768, 784);  x_184 = None
    x_185 = reshape_40.permute(0, 2, 1);  reshape_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_41 = l__mod___blocks_9_gamma3 * x_185;  l__mod___blocks_9_gamma3 = x_185 = None
    l__mod___blocks_9_drop_path_1 = self.L__mod___blocks_9_drop_path(mul_41);  mul_41 = None
    x_186 = x_179 + l__mod___blocks_9_drop_path_1;  x_179 = l__mod___blocks_9_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_9_gamma2 = self.L__mod___blocks_9_gamma2
    l__mod___blocks_9_norm2 = self.L__mod___blocks_9_norm2(x_186)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_187 = self.L__mod___blocks_9_mlp_fc1(l__mod___blocks_9_norm2);  l__mod___blocks_9_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_188 = self.L__mod___blocks_9_mlp_act(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_189 = self.L__mod___blocks_9_mlp_drop1(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_190 = self.L__mod___blocks_9_mlp_norm(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_191 = self.L__mod___blocks_9_mlp_fc2(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_192 = self.L__mod___blocks_9_mlp_drop2(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_42 = l__mod___blocks_9_gamma2 * x_192;  l__mod___blocks_9_gamma2 = x_192 = None
    l__mod___blocks_9_drop_path_2 = self.L__mod___blocks_9_drop_path(mul_42);  mul_42 = None
    x_194 = x_186 + l__mod___blocks_9_drop_path_2;  x_186 = l__mod___blocks_9_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_10_gamma1 = self.L__mod___blocks_10_gamma1
    l__mod___blocks_10_norm1 = self.L__mod___blocks_10_norm1(x_194)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_10_attn_qkv = self.L__mod___blocks_10_attn_qkv(l__mod___blocks_10_norm1);  l__mod___blocks_10_norm1 = None
    reshape_41 = l__mod___blocks_10_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_10_attn_qkv = None
    qkv_10 = reshape_41.permute(2, 0, 3, 4, 1);  reshape_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = qkv_10.unbind(0);  qkv_10 = None
    q_20 = unbind_10[0]
    k_20 = unbind_10[1]
    v_10 = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_21 = torch.nn.functional.normalize(q_20, dim = -1);  q_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_21 = torch.nn.functional.normalize(k_20, dim = -1);  k_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_11 = k_21.transpose(-2, -1);  k_21 = None
    matmul_20 = q_21 @ transpose_11;  q_21 = transpose_11 = None
    l__mod___blocks_10_attn_temperature = self.L__mod___blocks_10_attn_temperature
    attn_30 = matmul_20 * l__mod___blocks_10_attn_temperature;  matmul_20 = l__mod___blocks_10_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_31 = attn_30.softmax(dim = -1);  attn_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_32 = self.L__mod___blocks_10_attn_attn_drop(attn_31);  attn_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_21 = attn_32 @ v_10;  attn_32 = v_10 = None
    permute_43 = matmul_21.permute(0, 3, 1, 2);  matmul_21 = None
    x_195 = permute_43.reshape(8, 784, 768);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_196 = self.L__mod___blocks_10_attn_proj(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_197 = self.L__mod___blocks_10_attn_proj_drop(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_44 = l__mod___blocks_10_gamma1 * x_197;  l__mod___blocks_10_gamma1 = x_197 = None
    l__mod___blocks_10_drop_path = self.L__mod___blocks_10_drop_path(mul_44);  mul_44 = None
    x_198 = x_194 + l__mod___blocks_10_drop_path;  x_194 = l__mod___blocks_10_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_10_gamma3 = self.L__mod___blocks_10_gamma3
    l__mod___blocks_10_norm3 = self.L__mod___blocks_10_norm3(x_198)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_44 = l__mod___blocks_10_norm3.permute(0, 2, 1);  l__mod___blocks_10_norm3 = None
    x_199 = permute_44.reshape(8, 768, 28, 28);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_200 = self.L__mod___blocks_10_local_mp_conv1(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_201 = self.L__mod___blocks_10_local_mp_act(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_202 = self.L__mod___blocks_10_local_mp_bn(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_203 = self.L__mod___blocks_10_local_mp_conv2(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_44 = x_203.reshape(8, 768, 784);  x_203 = None
    x_204 = reshape_44.permute(0, 2, 1);  reshape_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_45 = l__mod___blocks_10_gamma3 * x_204;  l__mod___blocks_10_gamma3 = x_204 = None
    l__mod___blocks_10_drop_path_1 = self.L__mod___blocks_10_drop_path(mul_45);  mul_45 = None
    x_205 = x_198 + l__mod___blocks_10_drop_path_1;  x_198 = l__mod___blocks_10_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_10_gamma2 = self.L__mod___blocks_10_gamma2
    l__mod___blocks_10_norm2 = self.L__mod___blocks_10_norm2(x_205)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_206 = self.L__mod___blocks_10_mlp_fc1(l__mod___blocks_10_norm2);  l__mod___blocks_10_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_207 = self.L__mod___blocks_10_mlp_act(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_208 = self.L__mod___blocks_10_mlp_drop1(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_209 = self.L__mod___blocks_10_mlp_norm(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_210 = self.L__mod___blocks_10_mlp_fc2(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_211 = self.L__mod___blocks_10_mlp_drop2(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_46 = l__mod___blocks_10_gamma2 * x_211;  l__mod___blocks_10_gamma2 = x_211 = None
    l__mod___blocks_10_drop_path_2 = self.L__mod___blocks_10_drop_path(mul_46);  mul_46 = None
    x_213 = x_205 + l__mod___blocks_10_drop_path_2;  x_205 = l__mod___blocks_10_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_11_gamma1 = self.L__mod___blocks_11_gamma1
    l__mod___blocks_11_norm1 = self.L__mod___blocks_11_norm1(x_213)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_11_attn_qkv = self.L__mod___blocks_11_attn_qkv(l__mod___blocks_11_norm1);  l__mod___blocks_11_norm1 = None
    reshape_45 = l__mod___blocks_11_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_11_attn_qkv = None
    qkv_11 = reshape_45.permute(2, 0, 3, 4, 1);  reshape_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = qkv_11.unbind(0);  qkv_11 = None
    q_22 = unbind_11[0]
    k_22 = unbind_11[1]
    v_11 = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_23 = torch.nn.functional.normalize(q_22, dim = -1);  q_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_23 = torch.nn.functional.normalize(k_22, dim = -1);  k_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_12 = k_23.transpose(-2, -1);  k_23 = None
    matmul_22 = q_23 @ transpose_12;  q_23 = transpose_12 = None
    l__mod___blocks_11_attn_temperature = self.L__mod___blocks_11_attn_temperature
    attn_33 = matmul_22 * l__mod___blocks_11_attn_temperature;  matmul_22 = l__mod___blocks_11_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_34 = attn_33.softmax(dim = -1);  attn_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_35 = self.L__mod___blocks_11_attn_attn_drop(attn_34);  attn_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_23 = attn_35 @ v_11;  attn_35 = v_11 = None
    permute_47 = matmul_23.permute(0, 3, 1, 2);  matmul_23 = None
    x_214 = permute_47.reshape(8, 784, 768);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_215 = self.L__mod___blocks_11_attn_proj(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_216 = self.L__mod___blocks_11_attn_proj_drop(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_48 = l__mod___blocks_11_gamma1 * x_216;  l__mod___blocks_11_gamma1 = x_216 = None
    l__mod___blocks_11_drop_path = self.L__mod___blocks_11_drop_path(mul_48);  mul_48 = None
    x_217 = x_213 + l__mod___blocks_11_drop_path;  x_213 = l__mod___blocks_11_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_11_gamma3 = self.L__mod___blocks_11_gamma3
    l__mod___blocks_11_norm3 = self.L__mod___blocks_11_norm3(x_217)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_48 = l__mod___blocks_11_norm3.permute(0, 2, 1);  l__mod___blocks_11_norm3 = None
    x_218 = permute_48.reshape(8, 768, 28, 28);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_219 = self.L__mod___blocks_11_local_mp_conv1(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_220 = self.L__mod___blocks_11_local_mp_act(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_221 = self.L__mod___blocks_11_local_mp_bn(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_222 = self.L__mod___blocks_11_local_mp_conv2(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_48 = x_222.reshape(8, 768, 784);  x_222 = None
    x_223 = reshape_48.permute(0, 2, 1);  reshape_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_49 = l__mod___blocks_11_gamma3 * x_223;  l__mod___blocks_11_gamma3 = x_223 = None
    l__mod___blocks_11_drop_path_1 = self.L__mod___blocks_11_drop_path(mul_49);  mul_49 = None
    x_224 = x_217 + l__mod___blocks_11_drop_path_1;  x_217 = l__mod___blocks_11_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_11_gamma2 = self.L__mod___blocks_11_gamma2
    l__mod___blocks_11_norm2 = self.L__mod___blocks_11_norm2(x_224)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_225 = self.L__mod___blocks_11_mlp_fc1(l__mod___blocks_11_norm2);  l__mod___blocks_11_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_226 = self.L__mod___blocks_11_mlp_act(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_227 = self.L__mod___blocks_11_mlp_drop1(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_228 = self.L__mod___blocks_11_mlp_norm(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_229 = self.L__mod___blocks_11_mlp_fc2(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_230 = self.L__mod___blocks_11_mlp_drop2(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_50 = l__mod___blocks_11_gamma2 * x_230;  l__mod___blocks_11_gamma2 = x_230 = None
    l__mod___blocks_11_drop_path_2 = self.L__mod___blocks_11_drop_path(mul_50);  mul_50 = None
    x_232 = x_224 + l__mod___blocks_11_drop_path_2;  x_224 = l__mod___blocks_11_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_12_gamma1 = self.L__mod___blocks_12_gamma1
    l__mod___blocks_12_norm1 = self.L__mod___blocks_12_norm1(x_232)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_12_attn_qkv = self.L__mod___blocks_12_attn_qkv(l__mod___blocks_12_norm1);  l__mod___blocks_12_norm1 = None
    reshape_49 = l__mod___blocks_12_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_12_attn_qkv = None
    qkv_12 = reshape_49.permute(2, 0, 3, 4, 1);  reshape_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = qkv_12.unbind(0);  qkv_12 = None
    q_24 = unbind_12[0]
    k_24 = unbind_12[1]
    v_12 = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_25 = torch.nn.functional.normalize(q_24, dim = -1);  q_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_25 = torch.nn.functional.normalize(k_24, dim = -1);  k_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_13 = k_25.transpose(-2, -1);  k_25 = None
    matmul_24 = q_25 @ transpose_13;  q_25 = transpose_13 = None
    l__mod___blocks_12_attn_temperature = self.L__mod___blocks_12_attn_temperature
    attn_36 = matmul_24 * l__mod___blocks_12_attn_temperature;  matmul_24 = l__mod___blocks_12_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_37 = attn_36.softmax(dim = -1);  attn_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_38 = self.L__mod___blocks_12_attn_attn_drop(attn_37);  attn_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_25 = attn_38 @ v_12;  attn_38 = v_12 = None
    permute_51 = matmul_25.permute(0, 3, 1, 2);  matmul_25 = None
    x_233 = permute_51.reshape(8, 784, 768);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_234 = self.L__mod___blocks_12_attn_proj(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_235 = self.L__mod___blocks_12_attn_proj_drop(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_52 = l__mod___blocks_12_gamma1 * x_235;  l__mod___blocks_12_gamma1 = x_235 = None
    l__mod___blocks_12_drop_path = self.L__mod___blocks_12_drop_path(mul_52);  mul_52 = None
    x_236 = x_232 + l__mod___blocks_12_drop_path;  x_232 = l__mod___blocks_12_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_12_gamma3 = self.L__mod___blocks_12_gamma3
    l__mod___blocks_12_norm3 = self.L__mod___blocks_12_norm3(x_236)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_52 = l__mod___blocks_12_norm3.permute(0, 2, 1);  l__mod___blocks_12_norm3 = None
    x_237 = permute_52.reshape(8, 768, 28, 28);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_238 = self.L__mod___blocks_12_local_mp_conv1(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_239 = self.L__mod___blocks_12_local_mp_act(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_240 = self.L__mod___blocks_12_local_mp_bn(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_241 = self.L__mod___blocks_12_local_mp_conv2(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_52 = x_241.reshape(8, 768, 784);  x_241 = None
    x_242 = reshape_52.permute(0, 2, 1);  reshape_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_53 = l__mod___blocks_12_gamma3 * x_242;  l__mod___blocks_12_gamma3 = x_242 = None
    l__mod___blocks_12_drop_path_1 = self.L__mod___blocks_12_drop_path(mul_53);  mul_53 = None
    x_243 = x_236 + l__mod___blocks_12_drop_path_1;  x_236 = l__mod___blocks_12_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_12_gamma2 = self.L__mod___blocks_12_gamma2
    l__mod___blocks_12_norm2 = self.L__mod___blocks_12_norm2(x_243)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_244 = self.L__mod___blocks_12_mlp_fc1(l__mod___blocks_12_norm2);  l__mod___blocks_12_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_245 = self.L__mod___blocks_12_mlp_act(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_246 = self.L__mod___blocks_12_mlp_drop1(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_247 = self.L__mod___blocks_12_mlp_norm(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_248 = self.L__mod___blocks_12_mlp_fc2(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_249 = self.L__mod___blocks_12_mlp_drop2(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_54 = l__mod___blocks_12_gamma2 * x_249;  l__mod___blocks_12_gamma2 = x_249 = None
    l__mod___blocks_12_drop_path_2 = self.L__mod___blocks_12_drop_path(mul_54);  mul_54 = None
    x_251 = x_243 + l__mod___blocks_12_drop_path_2;  x_243 = l__mod___blocks_12_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_13_gamma1 = self.L__mod___blocks_13_gamma1
    l__mod___blocks_13_norm1 = self.L__mod___blocks_13_norm1(x_251)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_13_attn_qkv = self.L__mod___blocks_13_attn_qkv(l__mod___blocks_13_norm1);  l__mod___blocks_13_norm1 = None
    reshape_53 = l__mod___blocks_13_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_13_attn_qkv = None
    qkv_13 = reshape_53.permute(2, 0, 3, 4, 1);  reshape_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = qkv_13.unbind(0);  qkv_13 = None
    q_26 = unbind_13[0]
    k_26 = unbind_13[1]
    v_13 = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_27 = torch.nn.functional.normalize(q_26, dim = -1);  q_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_27 = torch.nn.functional.normalize(k_26, dim = -1);  k_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_14 = k_27.transpose(-2, -1);  k_27 = None
    matmul_26 = q_27 @ transpose_14;  q_27 = transpose_14 = None
    l__mod___blocks_13_attn_temperature = self.L__mod___blocks_13_attn_temperature
    attn_39 = matmul_26 * l__mod___blocks_13_attn_temperature;  matmul_26 = l__mod___blocks_13_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_40 = attn_39.softmax(dim = -1);  attn_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_41 = self.L__mod___blocks_13_attn_attn_drop(attn_40);  attn_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_27 = attn_41 @ v_13;  attn_41 = v_13 = None
    permute_55 = matmul_27.permute(0, 3, 1, 2);  matmul_27 = None
    x_252 = permute_55.reshape(8, 784, 768);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_253 = self.L__mod___blocks_13_attn_proj(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_254 = self.L__mod___blocks_13_attn_proj_drop(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_56 = l__mod___blocks_13_gamma1 * x_254;  l__mod___blocks_13_gamma1 = x_254 = None
    l__mod___blocks_13_drop_path = self.L__mod___blocks_13_drop_path(mul_56);  mul_56 = None
    x_255 = x_251 + l__mod___blocks_13_drop_path;  x_251 = l__mod___blocks_13_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_13_gamma3 = self.L__mod___blocks_13_gamma3
    l__mod___blocks_13_norm3 = self.L__mod___blocks_13_norm3(x_255)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_56 = l__mod___blocks_13_norm3.permute(0, 2, 1);  l__mod___blocks_13_norm3 = None
    x_256 = permute_56.reshape(8, 768, 28, 28);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_257 = self.L__mod___blocks_13_local_mp_conv1(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_258 = self.L__mod___blocks_13_local_mp_act(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_259 = self.L__mod___blocks_13_local_mp_bn(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_260 = self.L__mod___blocks_13_local_mp_conv2(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_56 = x_260.reshape(8, 768, 784);  x_260 = None
    x_261 = reshape_56.permute(0, 2, 1);  reshape_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_57 = l__mod___blocks_13_gamma3 * x_261;  l__mod___blocks_13_gamma3 = x_261 = None
    l__mod___blocks_13_drop_path_1 = self.L__mod___blocks_13_drop_path(mul_57);  mul_57 = None
    x_262 = x_255 + l__mod___blocks_13_drop_path_1;  x_255 = l__mod___blocks_13_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_13_gamma2 = self.L__mod___blocks_13_gamma2
    l__mod___blocks_13_norm2 = self.L__mod___blocks_13_norm2(x_262)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_263 = self.L__mod___blocks_13_mlp_fc1(l__mod___blocks_13_norm2);  l__mod___blocks_13_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_264 = self.L__mod___blocks_13_mlp_act(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_265 = self.L__mod___blocks_13_mlp_drop1(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_266 = self.L__mod___blocks_13_mlp_norm(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_267 = self.L__mod___blocks_13_mlp_fc2(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_268 = self.L__mod___blocks_13_mlp_drop2(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_58 = l__mod___blocks_13_gamma2 * x_268;  l__mod___blocks_13_gamma2 = x_268 = None
    l__mod___blocks_13_drop_path_2 = self.L__mod___blocks_13_drop_path(mul_58);  mul_58 = None
    x_270 = x_262 + l__mod___blocks_13_drop_path_2;  x_262 = l__mod___blocks_13_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_14_gamma1 = self.L__mod___blocks_14_gamma1
    l__mod___blocks_14_norm1 = self.L__mod___blocks_14_norm1(x_270)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_14_attn_qkv = self.L__mod___blocks_14_attn_qkv(l__mod___blocks_14_norm1);  l__mod___blocks_14_norm1 = None
    reshape_57 = l__mod___blocks_14_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_14_attn_qkv = None
    qkv_14 = reshape_57.permute(2, 0, 3, 4, 1);  reshape_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = qkv_14.unbind(0);  qkv_14 = None
    q_28 = unbind_14[0]
    k_28 = unbind_14[1]
    v_14 = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_29 = torch.nn.functional.normalize(q_28, dim = -1);  q_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_29 = torch.nn.functional.normalize(k_28, dim = -1);  k_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_15 = k_29.transpose(-2, -1);  k_29 = None
    matmul_28 = q_29 @ transpose_15;  q_29 = transpose_15 = None
    l__mod___blocks_14_attn_temperature = self.L__mod___blocks_14_attn_temperature
    attn_42 = matmul_28 * l__mod___blocks_14_attn_temperature;  matmul_28 = l__mod___blocks_14_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_43 = attn_42.softmax(dim = -1);  attn_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_44 = self.L__mod___blocks_14_attn_attn_drop(attn_43);  attn_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_29 = attn_44 @ v_14;  attn_44 = v_14 = None
    permute_59 = matmul_29.permute(0, 3, 1, 2);  matmul_29 = None
    x_271 = permute_59.reshape(8, 784, 768);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_272 = self.L__mod___blocks_14_attn_proj(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_273 = self.L__mod___blocks_14_attn_proj_drop(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_60 = l__mod___blocks_14_gamma1 * x_273;  l__mod___blocks_14_gamma1 = x_273 = None
    l__mod___blocks_14_drop_path = self.L__mod___blocks_14_drop_path(mul_60);  mul_60 = None
    x_274 = x_270 + l__mod___blocks_14_drop_path;  x_270 = l__mod___blocks_14_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_14_gamma3 = self.L__mod___blocks_14_gamma3
    l__mod___blocks_14_norm3 = self.L__mod___blocks_14_norm3(x_274)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_60 = l__mod___blocks_14_norm3.permute(0, 2, 1);  l__mod___blocks_14_norm3 = None
    x_275 = permute_60.reshape(8, 768, 28, 28);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_276 = self.L__mod___blocks_14_local_mp_conv1(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_277 = self.L__mod___blocks_14_local_mp_act(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_278 = self.L__mod___blocks_14_local_mp_bn(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_279 = self.L__mod___blocks_14_local_mp_conv2(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_60 = x_279.reshape(8, 768, 784);  x_279 = None
    x_280 = reshape_60.permute(0, 2, 1);  reshape_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_61 = l__mod___blocks_14_gamma3 * x_280;  l__mod___blocks_14_gamma3 = x_280 = None
    l__mod___blocks_14_drop_path_1 = self.L__mod___blocks_14_drop_path(mul_61);  mul_61 = None
    x_281 = x_274 + l__mod___blocks_14_drop_path_1;  x_274 = l__mod___blocks_14_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_14_gamma2 = self.L__mod___blocks_14_gamma2
    l__mod___blocks_14_norm2 = self.L__mod___blocks_14_norm2(x_281)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_282 = self.L__mod___blocks_14_mlp_fc1(l__mod___blocks_14_norm2);  l__mod___blocks_14_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_283 = self.L__mod___blocks_14_mlp_act(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_284 = self.L__mod___blocks_14_mlp_drop1(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_285 = self.L__mod___blocks_14_mlp_norm(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_286 = self.L__mod___blocks_14_mlp_fc2(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_287 = self.L__mod___blocks_14_mlp_drop2(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_62 = l__mod___blocks_14_gamma2 * x_287;  l__mod___blocks_14_gamma2 = x_287 = None
    l__mod___blocks_14_drop_path_2 = self.L__mod___blocks_14_drop_path(mul_62);  mul_62 = None
    x_289 = x_281 + l__mod___blocks_14_drop_path_2;  x_281 = l__mod___blocks_14_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_15_gamma1 = self.L__mod___blocks_15_gamma1
    l__mod___blocks_15_norm1 = self.L__mod___blocks_15_norm1(x_289)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_15_attn_qkv = self.L__mod___blocks_15_attn_qkv(l__mod___blocks_15_norm1);  l__mod___blocks_15_norm1 = None
    reshape_61 = l__mod___blocks_15_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_15_attn_qkv = None
    qkv_15 = reshape_61.permute(2, 0, 3, 4, 1);  reshape_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = qkv_15.unbind(0);  qkv_15 = None
    q_30 = unbind_15[0]
    k_30 = unbind_15[1]
    v_15 = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_31 = torch.nn.functional.normalize(q_30, dim = -1);  q_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_31 = torch.nn.functional.normalize(k_30, dim = -1);  k_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_16 = k_31.transpose(-2, -1);  k_31 = None
    matmul_30 = q_31 @ transpose_16;  q_31 = transpose_16 = None
    l__mod___blocks_15_attn_temperature = self.L__mod___blocks_15_attn_temperature
    attn_45 = matmul_30 * l__mod___blocks_15_attn_temperature;  matmul_30 = l__mod___blocks_15_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_46 = attn_45.softmax(dim = -1);  attn_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_47 = self.L__mod___blocks_15_attn_attn_drop(attn_46);  attn_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_31 = attn_47 @ v_15;  attn_47 = v_15 = None
    permute_63 = matmul_31.permute(0, 3, 1, 2);  matmul_31 = None
    x_290 = permute_63.reshape(8, 784, 768);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_291 = self.L__mod___blocks_15_attn_proj(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_292 = self.L__mod___blocks_15_attn_proj_drop(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_64 = l__mod___blocks_15_gamma1 * x_292;  l__mod___blocks_15_gamma1 = x_292 = None
    l__mod___blocks_15_drop_path = self.L__mod___blocks_15_drop_path(mul_64);  mul_64 = None
    x_293 = x_289 + l__mod___blocks_15_drop_path;  x_289 = l__mod___blocks_15_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_15_gamma3 = self.L__mod___blocks_15_gamma3
    l__mod___blocks_15_norm3 = self.L__mod___blocks_15_norm3(x_293)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_64 = l__mod___blocks_15_norm3.permute(0, 2, 1);  l__mod___blocks_15_norm3 = None
    x_294 = permute_64.reshape(8, 768, 28, 28);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_295 = self.L__mod___blocks_15_local_mp_conv1(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_296 = self.L__mod___blocks_15_local_mp_act(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_297 = self.L__mod___blocks_15_local_mp_bn(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_298 = self.L__mod___blocks_15_local_mp_conv2(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_64 = x_298.reshape(8, 768, 784);  x_298 = None
    x_299 = reshape_64.permute(0, 2, 1);  reshape_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_65 = l__mod___blocks_15_gamma3 * x_299;  l__mod___blocks_15_gamma3 = x_299 = None
    l__mod___blocks_15_drop_path_1 = self.L__mod___blocks_15_drop_path(mul_65);  mul_65 = None
    x_300 = x_293 + l__mod___blocks_15_drop_path_1;  x_293 = l__mod___blocks_15_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_15_gamma2 = self.L__mod___blocks_15_gamma2
    l__mod___blocks_15_norm2 = self.L__mod___blocks_15_norm2(x_300)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_301 = self.L__mod___blocks_15_mlp_fc1(l__mod___blocks_15_norm2);  l__mod___blocks_15_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_302 = self.L__mod___blocks_15_mlp_act(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_303 = self.L__mod___blocks_15_mlp_drop1(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_304 = self.L__mod___blocks_15_mlp_norm(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_305 = self.L__mod___blocks_15_mlp_fc2(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_306 = self.L__mod___blocks_15_mlp_drop2(x_305);  x_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_66 = l__mod___blocks_15_gamma2 * x_306;  l__mod___blocks_15_gamma2 = x_306 = None
    l__mod___blocks_15_drop_path_2 = self.L__mod___blocks_15_drop_path(mul_66);  mul_66 = None
    x_308 = x_300 + l__mod___blocks_15_drop_path_2;  x_300 = l__mod___blocks_15_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_16_gamma1 = self.L__mod___blocks_16_gamma1
    l__mod___blocks_16_norm1 = self.L__mod___blocks_16_norm1(x_308)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_16_attn_qkv = self.L__mod___blocks_16_attn_qkv(l__mod___blocks_16_norm1);  l__mod___blocks_16_norm1 = None
    reshape_65 = l__mod___blocks_16_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_16_attn_qkv = None
    qkv_16 = reshape_65.permute(2, 0, 3, 4, 1);  reshape_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = qkv_16.unbind(0);  qkv_16 = None
    q_32 = unbind_16[0]
    k_32 = unbind_16[1]
    v_16 = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_33 = torch.nn.functional.normalize(q_32, dim = -1);  q_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_33 = torch.nn.functional.normalize(k_32, dim = -1);  k_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_17 = k_33.transpose(-2, -1);  k_33 = None
    matmul_32 = q_33 @ transpose_17;  q_33 = transpose_17 = None
    l__mod___blocks_16_attn_temperature = self.L__mod___blocks_16_attn_temperature
    attn_48 = matmul_32 * l__mod___blocks_16_attn_temperature;  matmul_32 = l__mod___blocks_16_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_49 = attn_48.softmax(dim = -1);  attn_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_50 = self.L__mod___blocks_16_attn_attn_drop(attn_49);  attn_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_33 = attn_50 @ v_16;  attn_50 = v_16 = None
    permute_67 = matmul_33.permute(0, 3, 1, 2);  matmul_33 = None
    x_309 = permute_67.reshape(8, 784, 768);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_310 = self.L__mod___blocks_16_attn_proj(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_311 = self.L__mod___blocks_16_attn_proj_drop(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_68 = l__mod___blocks_16_gamma1 * x_311;  l__mod___blocks_16_gamma1 = x_311 = None
    l__mod___blocks_16_drop_path = self.L__mod___blocks_16_drop_path(mul_68);  mul_68 = None
    x_312 = x_308 + l__mod___blocks_16_drop_path;  x_308 = l__mod___blocks_16_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_16_gamma3 = self.L__mod___blocks_16_gamma3
    l__mod___blocks_16_norm3 = self.L__mod___blocks_16_norm3(x_312)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_68 = l__mod___blocks_16_norm3.permute(0, 2, 1);  l__mod___blocks_16_norm3 = None
    x_313 = permute_68.reshape(8, 768, 28, 28);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_314 = self.L__mod___blocks_16_local_mp_conv1(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_315 = self.L__mod___blocks_16_local_mp_act(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_316 = self.L__mod___blocks_16_local_mp_bn(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_317 = self.L__mod___blocks_16_local_mp_conv2(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_68 = x_317.reshape(8, 768, 784);  x_317 = None
    x_318 = reshape_68.permute(0, 2, 1);  reshape_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_69 = l__mod___blocks_16_gamma3 * x_318;  l__mod___blocks_16_gamma3 = x_318 = None
    l__mod___blocks_16_drop_path_1 = self.L__mod___blocks_16_drop_path(mul_69);  mul_69 = None
    x_319 = x_312 + l__mod___blocks_16_drop_path_1;  x_312 = l__mod___blocks_16_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_16_gamma2 = self.L__mod___blocks_16_gamma2
    l__mod___blocks_16_norm2 = self.L__mod___blocks_16_norm2(x_319)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_320 = self.L__mod___blocks_16_mlp_fc1(l__mod___blocks_16_norm2);  l__mod___blocks_16_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_321 = self.L__mod___blocks_16_mlp_act(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_322 = self.L__mod___blocks_16_mlp_drop1(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_323 = self.L__mod___blocks_16_mlp_norm(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_324 = self.L__mod___blocks_16_mlp_fc2(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_325 = self.L__mod___blocks_16_mlp_drop2(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_70 = l__mod___blocks_16_gamma2 * x_325;  l__mod___blocks_16_gamma2 = x_325 = None
    l__mod___blocks_16_drop_path_2 = self.L__mod___blocks_16_drop_path(mul_70);  mul_70 = None
    x_327 = x_319 + l__mod___blocks_16_drop_path_2;  x_319 = l__mod___blocks_16_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_17_gamma1 = self.L__mod___blocks_17_gamma1
    l__mod___blocks_17_norm1 = self.L__mod___blocks_17_norm1(x_327)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_17_attn_qkv = self.L__mod___blocks_17_attn_qkv(l__mod___blocks_17_norm1);  l__mod___blocks_17_norm1 = None
    reshape_69 = l__mod___blocks_17_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_17_attn_qkv = None
    qkv_17 = reshape_69.permute(2, 0, 3, 4, 1);  reshape_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = qkv_17.unbind(0);  qkv_17 = None
    q_34 = unbind_17[0]
    k_34 = unbind_17[1]
    v_17 = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_35 = torch.nn.functional.normalize(q_34, dim = -1);  q_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_35 = torch.nn.functional.normalize(k_34, dim = -1);  k_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_18 = k_35.transpose(-2, -1);  k_35 = None
    matmul_34 = q_35 @ transpose_18;  q_35 = transpose_18 = None
    l__mod___blocks_17_attn_temperature = self.L__mod___blocks_17_attn_temperature
    attn_51 = matmul_34 * l__mod___blocks_17_attn_temperature;  matmul_34 = l__mod___blocks_17_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_52 = attn_51.softmax(dim = -1);  attn_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_53 = self.L__mod___blocks_17_attn_attn_drop(attn_52);  attn_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_35 = attn_53 @ v_17;  attn_53 = v_17 = None
    permute_71 = matmul_35.permute(0, 3, 1, 2);  matmul_35 = None
    x_328 = permute_71.reshape(8, 784, 768);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_329 = self.L__mod___blocks_17_attn_proj(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_330 = self.L__mod___blocks_17_attn_proj_drop(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_72 = l__mod___blocks_17_gamma1 * x_330;  l__mod___blocks_17_gamma1 = x_330 = None
    l__mod___blocks_17_drop_path = self.L__mod___blocks_17_drop_path(mul_72);  mul_72 = None
    x_331 = x_327 + l__mod___blocks_17_drop_path;  x_327 = l__mod___blocks_17_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_17_gamma3 = self.L__mod___blocks_17_gamma3
    l__mod___blocks_17_norm3 = self.L__mod___blocks_17_norm3(x_331)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_72 = l__mod___blocks_17_norm3.permute(0, 2, 1);  l__mod___blocks_17_norm3 = None
    x_332 = permute_72.reshape(8, 768, 28, 28);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_333 = self.L__mod___blocks_17_local_mp_conv1(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_334 = self.L__mod___blocks_17_local_mp_act(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_335 = self.L__mod___blocks_17_local_mp_bn(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_336 = self.L__mod___blocks_17_local_mp_conv2(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_72 = x_336.reshape(8, 768, 784);  x_336 = None
    x_337 = reshape_72.permute(0, 2, 1);  reshape_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_73 = l__mod___blocks_17_gamma3 * x_337;  l__mod___blocks_17_gamma3 = x_337 = None
    l__mod___blocks_17_drop_path_1 = self.L__mod___blocks_17_drop_path(mul_73);  mul_73 = None
    x_338 = x_331 + l__mod___blocks_17_drop_path_1;  x_331 = l__mod___blocks_17_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_17_gamma2 = self.L__mod___blocks_17_gamma2
    l__mod___blocks_17_norm2 = self.L__mod___blocks_17_norm2(x_338)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_339 = self.L__mod___blocks_17_mlp_fc1(l__mod___blocks_17_norm2);  l__mod___blocks_17_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_340 = self.L__mod___blocks_17_mlp_act(x_339);  x_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_341 = self.L__mod___blocks_17_mlp_drop1(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_342 = self.L__mod___blocks_17_mlp_norm(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_343 = self.L__mod___blocks_17_mlp_fc2(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_344 = self.L__mod___blocks_17_mlp_drop2(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_74 = l__mod___blocks_17_gamma2 * x_344;  l__mod___blocks_17_gamma2 = x_344 = None
    l__mod___blocks_17_drop_path_2 = self.L__mod___blocks_17_drop_path(mul_74);  mul_74 = None
    x_346 = x_338 + l__mod___blocks_17_drop_path_2;  x_338 = l__mod___blocks_17_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_18_gamma1 = self.L__mod___blocks_18_gamma1
    l__mod___blocks_18_norm1 = self.L__mod___blocks_18_norm1(x_346)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_18_attn_qkv = self.L__mod___blocks_18_attn_qkv(l__mod___blocks_18_norm1);  l__mod___blocks_18_norm1 = None
    reshape_73 = l__mod___blocks_18_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_18_attn_qkv = None
    qkv_18 = reshape_73.permute(2, 0, 3, 4, 1);  reshape_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = qkv_18.unbind(0);  qkv_18 = None
    q_36 = unbind_18[0]
    k_36 = unbind_18[1]
    v_18 = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_37 = torch.nn.functional.normalize(q_36, dim = -1);  q_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_37 = torch.nn.functional.normalize(k_36, dim = -1);  k_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_19 = k_37.transpose(-2, -1);  k_37 = None
    matmul_36 = q_37 @ transpose_19;  q_37 = transpose_19 = None
    l__mod___blocks_18_attn_temperature = self.L__mod___blocks_18_attn_temperature
    attn_54 = matmul_36 * l__mod___blocks_18_attn_temperature;  matmul_36 = l__mod___blocks_18_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_55 = attn_54.softmax(dim = -1);  attn_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_56 = self.L__mod___blocks_18_attn_attn_drop(attn_55);  attn_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_37 = attn_56 @ v_18;  attn_56 = v_18 = None
    permute_75 = matmul_37.permute(0, 3, 1, 2);  matmul_37 = None
    x_347 = permute_75.reshape(8, 784, 768);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_348 = self.L__mod___blocks_18_attn_proj(x_347);  x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_349 = self.L__mod___blocks_18_attn_proj_drop(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_76 = l__mod___blocks_18_gamma1 * x_349;  l__mod___blocks_18_gamma1 = x_349 = None
    l__mod___blocks_18_drop_path = self.L__mod___blocks_18_drop_path(mul_76);  mul_76 = None
    x_350 = x_346 + l__mod___blocks_18_drop_path;  x_346 = l__mod___blocks_18_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_18_gamma3 = self.L__mod___blocks_18_gamma3
    l__mod___blocks_18_norm3 = self.L__mod___blocks_18_norm3(x_350)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_76 = l__mod___blocks_18_norm3.permute(0, 2, 1);  l__mod___blocks_18_norm3 = None
    x_351 = permute_76.reshape(8, 768, 28, 28);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_352 = self.L__mod___blocks_18_local_mp_conv1(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_353 = self.L__mod___blocks_18_local_mp_act(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_354 = self.L__mod___blocks_18_local_mp_bn(x_353);  x_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_355 = self.L__mod___blocks_18_local_mp_conv2(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_76 = x_355.reshape(8, 768, 784);  x_355 = None
    x_356 = reshape_76.permute(0, 2, 1);  reshape_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_77 = l__mod___blocks_18_gamma3 * x_356;  l__mod___blocks_18_gamma3 = x_356 = None
    l__mod___blocks_18_drop_path_1 = self.L__mod___blocks_18_drop_path(mul_77);  mul_77 = None
    x_357 = x_350 + l__mod___blocks_18_drop_path_1;  x_350 = l__mod___blocks_18_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_18_gamma2 = self.L__mod___blocks_18_gamma2
    l__mod___blocks_18_norm2 = self.L__mod___blocks_18_norm2(x_357)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_358 = self.L__mod___blocks_18_mlp_fc1(l__mod___blocks_18_norm2);  l__mod___blocks_18_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_359 = self.L__mod___blocks_18_mlp_act(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_360 = self.L__mod___blocks_18_mlp_drop1(x_359);  x_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_361 = self.L__mod___blocks_18_mlp_norm(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_362 = self.L__mod___blocks_18_mlp_fc2(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_363 = self.L__mod___blocks_18_mlp_drop2(x_362);  x_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_78 = l__mod___blocks_18_gamma2 * x_363;  l__mod___blocks_18_gamma2 = x_363 = None
    l__mod___blocks_18_drop_path_2 = self.L__mod___blocks_18_drop_path(mul_78);  mul_78 = None
    x_365 = x_357 + l__mod___blocks_18_drop_path_2;  x_357 = l__mod___blocks_18_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_19_gamma1 = self.L__mod___blocks_19_gamma1
    l__mod___blocks_19_norm1 = self.L__mod___blocks_19_norm1(x_365)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_19_attn_qkv = self.L__mod___blocks_19_attn_qkv(l__mod___blocks_19_norm1);  l__mod___blocks_19_norm1 = None
    reshape_77 = l__mod___blocks_19_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_19_attn_qkv = None
    qkv_19 = reshape_77.permute(2, 0, 3, 4, 1);  reshape_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = qkv_19.unbind(0);  qkv_19 = None
    q_38 = unbind_19[0]
    k_38 = unbind_19[1]
    v_19 = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_39 = torch.nn.functional.normalize(q_38, dim = -1);  q_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_39 = torch.nn.functional.normalize(k_38, dim = -1);  k_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_20 = k_39.transpose(-2, -1);  k_39 = None
    matmul_38 = q_39 @ transpose_20;  q_39 = transpose_20 = None
    l__mod___blocks_19_attn_temperature = self.L__mod___blocks_19_attn_temperature
    attn_57 = matmul_38 * l__mod___blocks_19_attn_temperature;  matmul_38 = l__mod___blocks_19_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_58 = attn_57.softmax(dim = -1);  attn_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_59 = self.L__mod___blocks_19_attn_attn_drop(attn_58);  attn_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_39 = attn_59 @ v_19;  attn_59 = v_19 = None
    permute_79 = matmul_39.permute(0, 3, 1, 2);  matmul_39 = None
    x_366 = permute_79.reshape(8, 784, 768);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_367 = self.L__mod___blocks_19_attn_proj(x_366);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_368 = self.L__mod___blocks_19_attn_proj_drop(x_367);  x_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_80 = l__mod___blocks_19_gamma1 * x_368;  l__mod___blocks_19_gamma1 = x_368 = None
    l__mod___blocks_19_drop_path = self.L__mod___blocks_19_drop_path(mul_80);  mul_80 = None
    x_369 = x_365 + l__mod___blocks_19_drop_path;  x_365 = l__mod___blocks_19_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_19_gamma3 = self.L__mod___blocks_19_gamma3
    l__mod___blocks_19_norm3 = self.L__mod___blocks_19_norm3(x_369)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_80 = l__mod___blocks_19_norm3.permute(0, 2, 1);  l__mod___blocks_19_norm3 = None
    x_370 = permute_80.reshape(8, 768, 28, 28);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_371 = self.L__mod___blocks_19_local_mp_conv1(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_372 = self.L__mod___blocks_19_local_mp_act(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_373 = self.L__mod___blocks_19_local_mp_bn(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_374 = self.L__mod___blocks_19_local_mp_conv2(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_80 = x_374.reshape(8, 768, 784);  x_374 = None
    x_375 = reshape_80.permute(0, 2, 1);  reshape_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_81 = l__mod___blocks_19_gamma3 * x_375;  l__mod___blocks_19_gamma3 = x_375 = None
    l__mod___blocks_19_drop_path_1 = self.L__mod___blocks_19_drop_path(mul_81);  mul_81 = None
    x_376 = x_369 + l__mod___blocks_19_drop_path_1;  x_369 = l__mod___blocks_19_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_19_gamma2 = self.L__mod___blocks_19_gamma2
    l__mod___blocks_19_norm2 = self.L__mod___blocks_19_norm2(x_376)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_377 = self.L__mod___blocks_19_mlp_fc1(l__mod___blocks_19_norm2);  l__mod___blocks_19_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_378 = self.L__mod___blocks_19_mlp_act(x_377);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_379 = self.L__mod___blocks_19_mlp_drop1(x_378);  x_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_380 = self.L__mod___blocks_19_mlp_norm(x_379);  x_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_381 = self.L__mod___blocks_19_mlp_fc2(x_380);  x_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_382 = self.L__mod___blocks_19_mlp_drop2(x_381);  x_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_82 = l__mod___blocks_19_gamma2 * x_382;  l__mod___blocks_19_gamma2 = x_382 = None
    l__mod___blocks_19_drop_path_2 = self.L__mod___blocks_19_drop_path(mul_82);  mul_82 = None
    x_384 = x_376 + l__mod___blocks_19_drop_path_2;  x_376 = l__mod___blocks_19_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_20_gamma1 = self.L__mod___blocks_20_gamma1
    l__mod___blocks_20_norm1 = self.L__mod___blocks_20_norm1(x_384)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_20_attn_qkv = self.L__mod___blocks_20_attn_qkv(l__mod___blocks_20_norm1);  l__mod___blocks_20_norm1 = None
    reshape_81 = l__mod___blocks_20_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_20_attn_qkv = None
    qkv_20 = reshape_81.permute(2, 0, 3, 4, 1);  reshape_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = qkv_20.unbind(0);  qkv_20 = None
    q_40 = unbind_20[0]
    k_40 = unbind_20[1]
    v_20 = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_41 = torch.nn.functional.normalize(q_40, dim = -1);  q_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_41 = torch.nn.functional.normalize(k_40, dim = -1);  k_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_21 = k_41.transpose(-2, -1);  k_41 = None
    matmul_40 = q_41 @ transpose_21;  q_41 = transpose_21 = None
    l__mod___blocks_20_attn_temperature = self.L__mod___blocks_20_attn_temperature
    attn_60 = matmul_40 * l__mod___blocks_20_attn_temperature;  matmul_40 = l__mod___blocks_20_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_61 = attn_60.softmax(dim = -1);  attn_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_62 = self.L__mod___blocks_20_attn_attn_drop(attn_61);  attn_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_41 = attn_62 @ v_20;  attn_62 = v_20 = None
    permute_83 = matmul_41.permute(0, 3, 1, 2);  matmul_41 = None
    x_385 = permute_83.reshape(8, 784, 768);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_386 = self.L__mod___blocks_20_attn_proj(x_385);  x_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_387 = self.L__mod___blocks_20_attn_proj_drop(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_84 = l__mod___blocks_20_gamma1 * x_387;  l__mod___blocks_20_gamma1 = x_387 = None
    l__mod___blocks_20_drop_path = self.L__mod___blocks_20_drop_path(mul_84);  mul_84 = None
    x_388 = x_384 + l__mod___blocks_20_drop_path;  x_384 = l__mod___blocks_20_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_20_gamma3 = self.L__mod___blocks_20_gamma3
    l__mod___blocks_20_norm3 = self.L__mod___blocks_20_norm3(x_388)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_84 = l__mod___blocks_20_norm3.permute(0, 2, 1);  l__mod___blocks_20_norm3 = None
    x_389 = permute_84.reshape(8, 768, 28, 28);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_390 = self.L__mod___blocks_20_local_mp_conv1(x_389);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_391 = self.L__mod___blocks_20_local_mp_act(x_390);  x_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_392 = self.L__mod___blocks_20_local_mp_bn(x_391);  x_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_393 = self.L__mod___blocks_20_local_mp_conv2(x_392);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_84 = x_393.reshape(8, 768, 784);  x_393 = None
    x_394 = reshape_84.permute(0, 2, 1);  reshape_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_85 = l__mod___blocks_20_gamma3 * x_394;  l__mod___blocks_20_gamma3 = x_394 = None
    l__mod___blocks_20_drop_path_1 = self.L__mod___blocks_20_drop_path(mul_85);  mul_85 = None
    x_395 = x_388 + l__mod___blocks_20_drop_path_1;  x_388 = l__mod___blocks_20_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_20_gamma2 = self.L__mod___blocks_20_gamma2
    l__mod___blocks_20_norm2 = self.L__mod___blocks_20_norm2(x_395)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_396 = self.L__mod___blocks_20_mlp_fc1(l__mod___blocks_20_norm2);  l__mod___blocks_20_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_397 = self.L__mod___blocks_20_mlp_act(x_396);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_398 = self.L__mod___blocks_20_mlp_drop1(x_397);  x_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_399 = self.L__mod___blocks_20_mlp_norm(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_400 = self.L__mod___blocks_20_mlp_fc2(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_401 = self.L__mod___blocks_20_mlp_drop2(x_400);  x_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_86 = l__mod___blocks_20_gamma2 * x_401;  l__mod___blocks_20_gamma2 = x_401 = None
    l__mod___blocks_20_drop_path_2 = self.L__mod___blocks_20_drop_path(mul_86);  mul_86 = None
    x_403 = x_395 + l__mod___blocks_20_drop_path_2;  x_395 = l__mod___blocks_20_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_21_gamma1 = self.L__mod___blocks_21_gamma1
    l__mod___blocks_21_norm1 = self.L__mod___blocks_21_norm1(x_403)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_21_attn_qkv = self.L__mod___blocks_21_attn_qkv(l__mod___blocks_21_norm1);  l__mod___blocks_21_norm1 = None
    reshape_85 = l__mod___blocks_21_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_21_attn_qkv = None
    qkv_21 = reshape_85.permute(2, 0, 3, 4, 1);  reshape_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = qkv_21.unbind(0);  qkv_21 = None
    q_42 = unbind_21[0]
    k_42 = unbind_21[1]
    v_21 = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_43 = torch.nn.functional.normalize(q_42, dim = -1);  q_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_43 = torch.nn.functional.normalize(k_42, dim = -1);  k_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_22 = k_43.transpose(-2, -1);  k_43 = None
    matmul_42 = q_43 @ transpose_22;  q_43 = transpose_22 = None
    l__mod___blocks_21_attn_temperature = self.L__mod___blocks_21_attn_temperature
    attn_63 = matmul_42 * l__mod___blocks_21_attn_temperature;  matmul_42 = l__mod___blocks_21_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_64 = attn_63.softmax(dim = -1);  attn_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_65 = self.L__mod___blocks_21_attn_attn_drop(attn_64);  attn_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_43 = attn_65 @ v_21;  attn_65 = v_21 = None
    permute_87 = matmul_43.permute(0, 3, 1, 2);  matmul_43 = None
    x_404 = permute_87.reshape(8, 784, 768);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_405 = self.L__mod___blocks_21_attn_proj(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_406 = self.L__mod___blocks_21_attn_proj_drop(x_405);  x_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_88 = l__mod___blocks_21_gamma1 * x_406;  l__mod___blocks_21_gamma1 = x_406 = None
    l__mod___blocks_21_drop_path = self.L__mod___blocks_21_drop_path(mul_88);  mul_88 = None
    x_407 = x_403 + l__mod___blocks_21_drop_path;  x_403 = l__mod___blocks_21_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_21_gamma3 = self.L__mod___blocks_21_gamma3
    l__mod___blocks_21_norm3 = self.L__mod___blocks_21_norm3(x_407)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_88 = l__mod___blocks_21_norm3.permute(0, 2, 1);  l__mod___blocks_21_norm3 = None
    x_408 = permute_88.reshape(8, 768, 28, 28);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_409 = self.L__mod___blocks_21_local_mp_conv1(x_408);  x_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_410 = self.L__mod___blocks_21_local_mp_act(x_409);  x_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_411 = self.L__mod___blocks_21_local_mp_bn(x_410);  x_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_412 = self.L__mod___blocks_21_local_mp_conv2(x_411);  x_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_88 = x_412.reshape(8, 768, 784);  x_412 = None
    x_413 = reshape_88.permute(0, 2, 1);  reshape_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_89 = l__mod___blocks_21_gamma3 * x_413;  l__mod___blocks_21_gamma3 = x_413 = None
    l__mod___blocks_21_drop_path_1 = self.L__mod___blocks_21_drop_path(mul_89);  mul_89 = None
    x_414 = x_407 + l__mod___blocks_21_drop_path_1;  x_407 = l__mod___blocks_21_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_21_gamma2 = self.L__mod___blocks_21_gamma2
    l__mod___blocks_21_norm2 = self.L__mod___blocks_21_norm2(x_414)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_415 = self.L__mod___blocks_21_mlp_fc1(l__mod___blocks_21_norm2);  l__mod___blocks_21_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_416 = self.L__mod___blocks_21_mlp_act(x_415);  x_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_417 = self.L__mod___blocks_21_mlp_drop1(x_416);  x_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_418 = self.L__mod___blocks_21_mlp_norm(x_417);  x_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_419 = self.L__mod___blocks_21_mlp_fc2(x_418);  x_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_420 = self.L__mod___blocks_21_mlp_drop2(x_419);  x_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_90 = l__mod___blocks_21_gamma2 * x_420;  l__mod___blocks_21_gamma2 = x_420 = None
    l__mod___blocks_21_drop_path_2 = self.L__mod___blocks_21_drop_path(mul_90);  mul_90 = None
    x_422 = x_414 + l__mod___blocks_21_drop_path_2;  x_414 = l__mod___blocks_21_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_22_gamma1 = self.L__mod___blocks_22_gamma1
    l__mod___blocks_22_norm1 = self.L__mod___blocks_22_norm1(x_422)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_22_attn_qkv = self.L__mod___blocks_22_attn_qkv(l__mod___blocks_22_norm1);  l__mod___blocks_22_norm1 = None
    reshape_89 = l__mod___blocks_22_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_22_attn_qkv = None
    qkv_22 = reshape_89.permute(2, 0, 3, 4, 1);  reshape_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = qkv_22.unbind(0);  qkv_22 = None
    q_44 = unbind_22[0]
    k_44 = unbind_22[1]
    v_22 = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_45 = torch.nn.functional.normalize(q_44, dim = -1);  q_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_45 = torch.nn.functional.normalize(k_44, dim = -1);  k_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_23 = k_45.transpose(-2, -1);  k_45 = None
    matmul_44 = q_45 @ transpose_23;  q_45 = transpose_23 = None
    l__mod___blocks_22_attn_temperature = self.L__mod___blocks_22_attn_temperature
    attn_66 = matmul_44 * l__mod___blocks_22_attn_temperature;  matmul_44 = l__mod___blocks_22_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_67 = attn_66.softmax(dim = -1);  attn_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_68 = self.L__mod___blocks_22_attn_attn_drop(attn_67);  attn_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_45 = attn_68 @ v_22;  attn_68 = v_22 = None
    permute_91 = matmul_45.permute(0, 3, 1, 2);  matmul_45 = None
    x_423 = permute_91.reshape(8, 784, 768);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_424 = self.L__mod___blocks_22_attn_proj(x_423);  x_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_425 = self.L__mod___blocks_22_attn_proj_drop(x_424);  x_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_92 = l__mod___blocks_22_gamma1 * x_425;  l__mod___blocks_22_gamma1 = x_425 = None
    l__mod___blocks_22_drop_path = self.L__mod___blocks_22_drop_path(mul_92);  mul_92 = None
    x_426 = x_422 + l__mod___blocks_22_drop_path;  x_422 = l__mod___blocks_22_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_22_gamma3 = self.L__mod___blocks_22_gamma3
    l__mod___blocks_22_norm3 = self.L__mod___blocks_22_norm3(x_426)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_92 = l__mod___blocks_22_norm3.permute(0, 2, 1);  l__mod___blocks_22_norm3 = None
    x_427 = permute_92.reshape(8, 768, 28, 28);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_428 = self.L__mod___blocks_22_local_mp_conv1(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_429 = self.L__mod___blocks_22_local_mp_act(x_428);  x_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_430 = self.L__mod___blocks_22_local_mp_bn(x_429);  x_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_431 = self.L__mod___blocks_22_local_mp_conv2(x_430);  x_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_92 = x_431.reshape(8, 768, 784);  x_431 = None
    x_432 = reshape_92.permute(0, 2, 1);  reshape_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_93 = l__mod___blocks_22_gamma3 * x_432;  l__mod___blocks_22_gamma3 = x_432 = None
    l__mod___blocks_22_drop_path_1 = self.L__mod___blocks_22_drop_path(mul_93);  mul_93 = None
    x_433 = x_426 + l__mod___blocks_22_drop_path_1;  x_426 = l__mod___blocks_22_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_22_gamma2 = self.L__mod___blocks_22_gamma2
    l__mod___blocks_22_norm2 = self.L__mod___blocks_22_norm2(x_433)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_434 = self.L__mod___blocks_22_mlp_fc1(l__mod___blocks_22_norm2);  l__mod___blocks_22_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_435 = self.L__mod___blocks_22_mlp_act(x_434);  x_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_436 = self.L__mod___blocks_22_mlp_drop1(x_435);  x_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_437 = self.L__mod___blocks_22_mlp_norm(x_436);  x_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_438 = self.L__mod___blocks_22_mlp_fc2(x_437);  x_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_439 = self.L__mod___blocks_22_mlp_drop2(x_438);  x_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_94 = l__mod___blocks_22_gamma2 * x_439;  l__mod___blocks_22_gamma2 = x_439 = None
    l__mod___blocks_22_drop_path_2 = self.L__mod___blocks_22_drop_path(mul_94);  mul_94 = None
    x_441 = x_433 + l__mod___blocks_22_drop_path_2;  x_433 = l__mod___blocks_22_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    l__mod___blocks_23_gamma1 = self.L__mod___blocks_23_gamma1
    l__mod___blocks_23_norm1 = self.L__mod___blocks_23_norm1(x_441)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    l__mod___blocks_23_attn_qkv = self.L__mod___blocks_23_attn_qkv(l__mod___blocks_23_norm1);  l__mod___blocks_23_norm1 = None
    reshape_93 = l__mod___blocks_23_attn_qkv.reshape(8, 784, 3, 16, 48);  l__mod___blocks_23_attn_qkv = None
    qkv_23 = reshape_93.permute(2, 0, 3, 4, 1);  reshape_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = qkv_23.unbind(0);  qkv_23 = None
    q_46 = unbind_23[0]
    k_46 = unbind_23[1]
    v_23 = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    q_47 = torch.nn.functional.normalize(q_46, dim = -1);  q_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    k_47 = torch.nn.functional.normalize(k_46, dim = -1);  k_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_24 = k_47.transpose(-2, -1);  k_47 = None
    matmul_46 = q_47 @ transpose_24;  q_47 = transpose_24 = None
    l__mod___blocks_23_attn_temperature = self.L__mod___blocks_23_attn_temperature
    attn_69 = matmul_46 * l__mod___blocks_23_attn_temperature;  matmul_46 = l__mod___blocks_23_attn_temperature = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    attn_70 = attn_69.softmax(dim = -1);  attn_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    attn_71 = self.L__mod___blocks_23_attn_attn_drop(attn_70);  attn_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    matmul_47 = attn_71 @ v_23;  attn_71 = v_23 = None
    permute_95 = matmul_47.permute(0, 3, 1, 2);  matmul_47 = None
    x_442 = permute_95.reshape(8, 784, 768);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    x_443 = self.L__mod___blocks_23_attn_proj(x_442);  x_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    x_444 = self.L__mod___blocks_23_attn_proj_drop(x_443);  x_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_96 = l__mod___blocks_23_gamma1 * x_444;  l__mod___blocks_23_gamma1 = x_444 = None
    l__mod___blocks_23_drop_path = self.L__mod___blocks_23_drop_path(mul_96);  mul_96 = None
    x_445 = x_441 + l__mod___blocks_23_drop_path;  x_441 = l__mod___blocks_23_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    l__mod___blocks_23_gamma3 = self.L__mod___blocks_23_gamma3
    l__mod___blocks_23_norm3 = self.L__mod___blocks_23_norm3(x_445)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_96 = l__mod___blocks_23_norm3.permute(0, 2, 1);  l__mod___blocks_23_norm3 = None
    x_446 = permute_96.reshape(8, 768, 28, 28);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    x_447 = self.L__mod___blocks_23_local_mp_conv1(x_446);  x_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    x_448 = self.L__mod___blocks_23_local_mp_act(x_447);  x_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    x_449 = self.L__mod___blocks_23_local_mp_bn(x_448);  x_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    x_450 = self.L__mod___blocks_23_local_mp_conv2(x_449);  x_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    reshape_96 = x_450.reshape(8, 768, 784);  x_450 = None
    x_451 = reshape_96.permute(0, 2, 1);  reshape_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_97 = l__mod___blocks_23_gamma3 * x_451;  l__mod___blocks_23_gamma3 = x_451 = None
    l__mod___blocks_23_drop_path_1 = self.L__mod___blocks_23_drop_path(mul_97);  mul_97 = None
    x_452 = x_445 + l__mod___blocks_23_drop_path_1;  x_445 = l__mod___blocks_23_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    l__mod___blocks_23_gamma2 = self.L__mod___blocks_23_gamma2
    l__mod___blocks_23_norm2 = self.L__mod___blocks_23_norm2(x_452)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_453 = self.L__mod___blocks_23_mlp_fc1(l__mod___blocks_23_norm2);  l__mod___blocks_23_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_454 = self.L__mod___blocks_23_mlp_act(x_453);  x_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_455 = self.L__mod___blocks_23_mlp_drop1(x_454);  x_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_456 = self.L__mod___blocks_23_mlp_norm(x_455);  x_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_457 = self.L__mod___blocks_23_mlp_fc2(x_456);  x_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_458 = self.L__mod___blocks_23_mlp_drop2(x_457);  x_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_98 = l__mod___blocks_23_gamma2 * x_458;  l__mod___blocks_23_gamma2 = x_458 = None
    l__mod___blocks_23_drop_path_2 = self.L__mod___blocks_23_drop_path(mul_98);  mul_98 = None
    x_460 = x_452 + l__mod___blocks_23_drop_path_2;  x_452 = l__mod___blocks_23_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:447, code: x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
    l__mod___cls_token = self.L__mod___cls_token
    expand = l__mod___cls_token.expand(8, -1, -1);  l__mod___cls_token = None
    x_461 = torch.cat((expand, x_460), dim = 1);  expand = x_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    x_norm1 = self.L__mod___cls_attn_blocks_0_norm1(x_461)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_80 = x_norm1[(slice(None, None, None), 0)]
    l__mod___cls_attn_blocks_0_attn_q = self.L__mod___cls_attn_blocks_0_attn_q(getitem_80);  getitem_80 = None
    unsqueeze_1 = l__mod___cls_attn_blocks_0_attn_q.unsqueeze(1);  l__mod___cls_attn_blocks_0_attn_q = None
    reshape_97 = unsqueeze_1.reshape(8, 1, 16, 48);  unsqueeze_1 = None
    q_48 = reshape_97.permute(0, 2, 1, 3);  reshape_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___cls_attn_blocks_0_attn_k = self.L__mod___cls_attn_blocks_0_attn_k(x_norm1)
    reshape_98 = l__mod___cls_attn_blocks_0_attn_k.reshape(8, 785, 16, 48);  l__mod___cls_attn_blocks_0_attn_k = None
    k_48 = reshape_98.permute(0, 2, 1, 3);  reshape_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___cls_attn_blocks_0_attn_v = self.L__mod___cls_attn_blocks_0_attn_v(x_norm1)
    reshape_99 = l__mod___cls_attn_blocks_0_attn_v.reshape(8, 785, 16, 48);  l__mod___cls_attn_blocks_0_attn_v = None
    v_24 = reshape_99.permute(0, 2, 1, 3);  reshape_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    x_cls = torch._C._nn.scaled_dot_product_attention(q_48, k_48, v_24, dropout_p = 0.0);  q_48 = k_48 = v_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    transpose_25 = x_cls.transpose(1, 2);  x_cls = None
    x_cls_1 = transpose_25.reshape(8, 1, 768);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    x_cls_2 = self.L__mod___cls_attn_blocks_0_attn_proj(x_cls_1);  x_cls_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    x_cls_3 = self.L__mod___cls_attn_blocks_0_attn_proj_drop(x_cls_2);  x_cls_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    getitem_81 = x_norm1[(slice(None, None, None), slice(1, None, None))];  x_norm1 = None
    x_attn = torch.cat([x_cls_3, getitem_81], dim = 1);  x_cls_3 = getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    l__mod___cls_attn_blocks_0_gamma1 = self.L__mod___cls_attn_blocks_0_gamma1
    mul_99 = l__mod___cls_attn_blocks_0_gamma1 * x_attn;  l__mod___cls_attn_blocks_0_gamma1 = x_attn = None
    l__mod___cls_attn_blocks_0_drop_path = self.L__mod___cls_attn_blocks_0_drop_path(mul_99);  mul_99 = None
    x_462 = x_461 + l__mod___cls_attn_blocks_0_drop_path;  x_461 = l__mod___cls_attn_blocks_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    x_res = self.L__mod___cls_attn_blocks_0_norm2(x_462);  x_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    cls_token = x_res[(slice(None, None, None), slice(0, 1, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    l__mod___cls_attn_blocks_0_gamma2 = self.L__mod___cls_attn_blocks_0_gamma2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_464 = self.L__mod___cls_attn_blocks_0_mlp_fc1(cls_token);  cls_token = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_465 = self.L__mod___cls_attn_blocks_0_mlp_act(x_464);  x_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_466 = self.L__mod___cls_attn_blocks_0_mlp_drop1(x_465);  x_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_467 = self.L__mod___cls_attn_blocks_0_mlp_norm(x_466);  x_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_468 = self.L__mod___cls_attn_blocks_0_mlp_fc2(x_467);  x_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_469 = self.L__mod___cls_attn_blocks_0_mlp_drop2(x_468);  x_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    cls_token_1 = l__mod___cls_attn_blocks_0_gamma2 * x_469;  l__mod___cls_attn_blocks_0_gamma2 = x_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    getitem_83 = x_res[(slice(None, None, None), slice(1, None, None))]
    x_470 = torch.cat([cls_token_1, getitem_83], dim = 1);  cls_token_1 = getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:192, code: x = x_res + self.drop_path(x)
    l__mod___cls_attn_blocks_0_drop_path_1 = self.L__mod___cls_attn_blocks_0_drop_path(x_470);  x_470 = None
    x_472 = x_res + l__mod___cls_attn_blocks_0_drop_path_1;  x_res = l__mod___cls_attn_blocks_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    x_norm1_1 = self.L__mod___cls_attn_blocks_1_norm1(x_472)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_84 = x_norm1_1[(slice(None, None, None), 0)]
    l__mod___cls_attn_blocks_1_attn_q = self.L__mod___cls_attn_blocks_1_attn_q(getitem_84);  getitem_84 = None
    unsqueeze_2 = l__mod___cls_attn_blocks_1_attn_q.unsqueeze(1);  l__mod___cls_attn_blocks_1_attn_q = None
    reshape_101 = unsqueeze_2.reshape(8, 1, 16, 48);  unsqueeze_2 = None
    q_49 = reshape_101.permute(0, 2, 1, 3);  reshape_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___cls_attn_blocks_1_attn_k = self.L__mod___cls_attn_blocks_1_attn_k(x_norm1_1)
    reshape_102 = l__mod___cls_attn_blocks_1_attn_k.reshape(8, 785, 16, 48);  l__mod___cls_attn_blocks_1_attn_k = None
    k_49 = reshape_102.permute(0, 2, 1, 3);  reshape_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___cls_attn_blocks_1_attn_v = self.L__mod___cls_attn_blocks_1_attn_v(x_norm1_1)
    reshape_103 = l__mod___cls_attn_blocks_1_attn_v.reshape(8, 785, 16, 48);  l__mod___cls_attn_blocks_1_attn_v = None
    v_25 = reshape_103.permute(0, 2, 1, 3);  reshape_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    x_cls_4 = torch._C._nn.scaled_dot_product_attention(q_49, k_49, v_25, dropout_p = 0.0);  q_49 = k_49 = v_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    transpose_26 = x_cls_4.transpose(1, 2);  x_cls_4 = None
    x_cls_5 = transpose_26.reshape(8, 1, 768);  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    x_cls_6 = self.L__mod___cls_attn_blocks_1_attn_proj(x_cls_5);  x_cls_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    x_cls_7 = self.L__mod___cls_attn_blocks_1_attn_proj_drop(x_cls_6);  x_cls_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    getitem_85 = x_norm1_1[(slice(None, None, None), slice(1, None, None))];  x_norm1_1 = None
    x_attn_1 = torch.cat([x_cls_7, getitem_85], dim = 1);  x_cls_7 = getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    l__mod___cls_attn_blocks_1_gamma1 = self.L__mod___cls_attn_blocks_1_gamma1
    mul_101 = l__mod___cls_attn_blocks_1_gamma1 * x_attn_1;  l__mod___cls_attn_blocks_1_gamma1 = x_attn_1 = None
    l__mod___cls_attn_blocks_1_drop_path = self.L__mod___cls_attn_blocks_1_drop_path(mul_101);  mul_101 = None
    x_473 = x_472 + l__mod___cls_attn_blocks_1_drop_path;  x_472 = l__mod___cls_attn_blocks_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    x_res_1 = self.L__mod___cls_attn_blocks_1_norm2(x_473);  x_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    cls_token_2 = x_res_1[(slice(None, None, None), slice(0, 1, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    l__mod___cls_attn_blocks_1_gamma2 = self.L__mod___cls_attn_blocks_1_gamma2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_475 = self.L__mod___cls_attn_blocks_1_mlp_fc1(cls_token_2);  cls_token_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_476 = self.L__mod___cls_attn_blocks_1_mlp_act(x_475);  x_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_477 = self.L__mod___cls_attn_blocks_1_mlp_drop1(x_476);  x_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_478 = self.L__mod___cls_attn_blocks_1_mlp_norm(x_477);  x_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_479 = self.L__mod___cls_attn_blocks_1_mlp_fc2(x_478);  x_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_480 = self.L__mod___cls_attn_blocks_1_mlp_drop2(x_479);  x_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    cls_token_3 = l__mod___cls_attn_blocks_1_gamma2 * x_480;  l__mod___cls_attn_blocks_1_gamma2 = x_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    getitem_87 = x_res_1[(slice(None, None, None), slice(1, None, None))]
    x_481 = torch.cat([cls_token_3, getitem_87], dim = 1);  cls_token_3 = getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:192, code: x = x_res + self.drop_path(x)
    l__mod___cls_attn_blocks_1_drop_path_1 = self.L__mod___cls_attn_blocks_1_drop_path(x_481);  x_481 = None
    x_483 = x_res_1 + l__mod___cls_attn_blocks_1_drop_path_1;  x_res_1 = l__mod___cls_attn_blocks_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:455, code: x = self.norm(x)
    x_485 = self.L__mod___norm(x_483);  x_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:460, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    x_486 = x_485[(slice(None, None, None), 0)];  x_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:461, code: x = self.head_drop(x)
    x_487 = self.L__mod___head_drop(x_486);  x_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:462, code: return x if pre_logits else self.head(x)
    x_488 = self.L__mod___head(x_487);  x_487 = None
    return (x_488,)
    