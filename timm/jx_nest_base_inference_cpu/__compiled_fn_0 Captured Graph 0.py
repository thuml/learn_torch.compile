from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x = self.L__mod___patch_embed_proj(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_2 = self.L__mod___patch_embed_norm(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:232, code: x = self.pool(x)
    x_3 = self.getattr_L__mod___levels___0___pool(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    x_4 = x_3.permute(0, 2, 3, 1);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    x_5 = x_4.reshape(8, 4, 14, 4, 14, 128);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    transpose = x_5.transpose(2, 3);  x_5 = None
    x_7 = transpose.reshape(8, 16, -1, 128);  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    getattr_l__mod___levels___0___pos_embed = self.getattr_L__mod___levels___0___pos_embed
    x_8 = x_7 + getattr_l__mod___levels___0___pos_embed;  x_7 = getattr_l__mod___levels___0___pos_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm1_weight = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___norm1_weight
    getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm1_bias = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___norm1_bias
    y = torch.nn.functional.layer_norm(x_8, (128,), getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm1_weight, getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm1_weight = getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___0___transformer_encoder___0___attn_qkv = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___attn_qkv(y);  y = None
    reshape_2 = getattr_getattr_l__mod___levels___0___transformer_encoder___0___attn_qkv.reshape(8, 16, 196, 3, 4, 32);  getattr_getattr_l__mod___levels___0___transformer_encoder___0___attn_qkv = None
    qkv = reshape_2.permute(3, 0, 4, 1, 2, 5);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_10 = torch._C._nn.scaled_dot_product_attention(q, k, v, dropout_p = 0.0);  q = k = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_2 = x_10.permute(0, 2, 3, 4, 1);  x_10 = None
    x_11 = permute_2.reshape(8, 16, 196, 128);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_12 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___attn_proj(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_13 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___attn_proj_drop(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    getattr_getattr_l__mod___levels___0___transformer_encoder___0___drop_path = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___drop_path(x_13);  x_13 = None
    x_14 = x_8 + getattr_getattr_l__mod___levels___0___transformer_encoder___0___drop_path;  x_8 = getattr_getattr_l__mod___levels___0___transformer_encoder___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm2_weight = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___norm2_weight
    getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm2_bias = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___norm2_bias
    x_15 = torch.nn.functional.layer_norm(x_14, (128,), getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm2_weight, getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm2_weight = getattr_getattr_l__mod___levels___0___transformer_encoder___0___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_16 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___mlp_fc1(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_17 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___mlp_act(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_18 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___mlp_drop1(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_19 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___mlp_norm(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_20 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___mlp_fc2(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_21 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___mlp_drop2(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___levels___0___transformer_encoder___0___drop_path_1 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___0___drop_path(x_21);  x_21 = None
    x_22 = x_14 + getattr_getattr_l__mod___levels___0___transformer_encoder___0___drop_path_1;  x_14 = getattr_getattr_l__mod___levels___0___transformer_encoder___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm1_weight = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___norm1_weight
    getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm1_bias = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___norm1_bias
    y_1 = torch.nn.functional.layer_norm(x_22, (128,), getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm1_weight, getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm1_weight = getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___0___transformer_encoder___1___attn_qkv = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___attn_qkv(y_1);  y_1 = None
    reshape_4 = getattr_getattr_l__mod___levels___0___transformer_encoder___1___attn_qkv.reshape(8, 16, 196, 3, 4, 32);  getattr_getattr_l__mod___levels___0___transformer_encoder___1___attn_qkv = None
    qkv_1 = reshape_4.permute(3, 0, 4, 1, 2, 5);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_1 = unbind_1[0]
    k_1 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_24 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v_1, dropout_p = 0.0);  q_1 = k_1 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_4 = x_24.permute(0, 2, 3, 4, 1);  x_24 = None
    x_25 = permute_4.reshape(8, 16, 196, 128);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_26 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___attn_proj(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_27 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___attn_proj_drop(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_28 = x_22 + x_27;  x_22 = x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm2_weight = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___norm2_weight
    getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm2_bias = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___norm2_bias
    x_29 = torch.nn.functional.layer_norm(x_28, (128,), getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm2_weight, getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm2_weight = getattr_getattr_l__mod___levels___0___transformer_encoder___1___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_30 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___mlp_fc1(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_31 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___mlp_act(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_32 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___mlp_drop1(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_33 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___mlp_norm(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_34 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___mlp_fc2(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_35 = self.getattr_getattr_L__mod___levels___0___transformer_encoder___1___mlp_drop2(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_37 = x_28 + x_35;  x_28 = x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x_38 = x_37.reshape(8, 4, 4, 14, 14, 128);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    transpose_1 = x_38.transpose(2, 3);  x_38 = None
    x_40 = transpose_1.reshape(8, 56, 56, 128);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_5 = x_40.permute(0, 3, 1, 2);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    x_41 = self.getattr_L__mod___levels___1___pool_conv(permute_5);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_6 = x_41.permute(0, 2, 3, 1);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_l__mod___levels___1___pool_norm_weight = self.getattr_L__mod___levels___1___pool_norm_weight
    getattr_l__mod___levels___1___pool_norm_bias = self.getattr_L__mod___levels___1___pool_norm_bias
    x_42 = torch.nn.functional.layer_norm(permute_6, (256,), getattr_l__mod___levels___1___pool_norm_weight, getattr_l__mod___levels___1___pool_norm_bias, 1e-06);  permute_6 = getattr_l__mod___levels___1___pool_norm_weight = getattr_l__mod___levels___1___pool_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    x_43 = x_42.permute(0, 3, 1, 2);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_45 = torch.nn.functional.pad(x_43, (0, 1, 0, 1), value = -inf);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_47 = torch.nn.functional.max_pool2d(x_45, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    x_48 = x_47.permute(0, 2, 3, 1);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    x_49 = x_48.reshape(8, 2, 14, 2, 14, 256);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    transpose_2 = x_49.transpose(2, 3);  x_49 = None
    x_51 = transpose_2.reshape(8, 4, -1, 256);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    getattr_l__mod___levels___1___pos_embed = self.getattr_L__mod___levels___1___pos_embed
    x_52 = x_51 + getattr_l__mod___levels___1___pos_embed;  x_51 = getattr_l__mod___levels___1___pos_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm1_weight = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___norm1_weight
    getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm1_bias = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___norm1_bias
    y_2 = torch.nn.functional.layer_norm(x_52, (256,), getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm1_weight, getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm1_weight = getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___1___transformer_encoder___0___attn_qkv = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___attn_qkv(y_2);  y_2 = None
    reshape_10 = getattr_getattr_l__mod___levels___1___transformer_encoder___0___attn_qkv.reshape(8, 4, 196, 3, 8, 32);  getattr_getattr_l__mod___levels___1___transformer_encoder___0___attn_qkv = None
    qkv_2 = reshape_10.permute(3, 0, 4, 1, 2, 5);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_2 = unbind_2[0]
    k_2 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_54 = torch._C._nn.scaled_dot_product_attention(q_2, k_2, v_2, dropout_p = 0.0);  q_2 = k_2 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_10 = x_54.permute(0, 2, 3, 4, 1);  x_54 = None
    x_55 = permute_10.reshape(8, 4, 196, 256);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_56 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___attn_proj(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_57 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___attn_proj_drop(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_58 = x_52 + x_57;  x_52 = x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm2_weight = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___norm2_weight
    getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm2_bias = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___norm2_bias
    x_59 = torch.nn.functional.layer_norm(x_58, (256,), getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm2_weight, getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm2_weight = getattr_getattr_l__mod___levels___1___transformer_encoder___0___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_60 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___mlp_fc1(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_61 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___mlp_act(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_62 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___mlp_drop1(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_63 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___mlp_norm(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_64 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___mlp_fc2(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_65 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___0___mlp_drop2(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_66 = x_58 + x_65;  x_58 = x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm1_weight = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___norm1_weight
    getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm1_bias = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___norm1_bias
    y_3 = torch.nn.functional.layer_norm(x_66, (256,), getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm1_weight, getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm1_weight = getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___1___transformer_encoder___1___attn_qkv = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___attn_qkv(y_3);  y_3 = None
    reshape_12 = getattr_getattr_l__mod___levels___1___transformer_encoder___1___attn_qkv.reshape(8, 4, 196, 3, 8, 32);  getattr_getattr_l__mod___levels___1___transformer_encoder___1___attn_qkv = None
    qkv_3 = reshape_12.permute(3, 0, 4, 1, 2, 5);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_3 = unbind_3[0]
    k_3 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_68 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_3, dropout_p = 0.0);  q_3 = k_3 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_12 = x_68.permute(0, 2, 3, 4, 1);  x_68 = None
    x_69 = permute_12.reshape(8, 4, 196, 256);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_70 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___attn_proj(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_71 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___attn_proj_drop(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_72 = x_66 + x_71;  x_66 = x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm2_weight = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___norm2_weight
    getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm2_bias = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___norm2_bias
    x_73 = torch.nn.functional.layer_norm(x_72, (256,), getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm2_weight, getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm2_weight = getattr_getattr_l__mod___levels___1___transformer_encoder___1___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_74 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___mlp_fc1(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_75 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___mlp_act(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_76 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___mlp_drop1(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_77 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___mlp_norm(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_78 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___mlp_fc2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_79 = self.getattr_getattr_L__mod___levels___1___transformer_encoder___1___mlp_drop2(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_81 = x_72 + x_79;  x_72 = x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x_82 = x_81.reshape(8, 2, 2, 14, 14, 256);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    transpose_3 = x_82.transpose(2, 3);  x_82 = None
    x_84 = transpose_3.reshape(8, 28, 28, 256);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_13 = x_84.permute(0, 3, 1, 2);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    x_85 = self.getattr_L__mod___levels___2___pool_conv(permute_13);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_14 = x_85.permute(0, 2, 3, 1);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_l__mod___levels___2___pool_norm_weight = self.getattr_L__mod___levels___2___pool_norm_weight
    getattr_l__mod___levels___2___pool_norm_bias = self.getattr_L__mod___levels___2___pool_norm_bias
    x_86 = torch.nn.functional.layer_norm(permute_14, (512,), getattr_l__mod___levels___2___pool_norm_weight, getattr_l__mod___levels___2___pool_norm_bias, 1e-06);  permute_14 = getattr_l__mod___levels___2___pool_norm_weight = getattr_l__mod___levels___2___pool_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    x_87 = x_86.permute(0, 3, 1, 2);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_89 = torch.nn.functional.pad(x_87, (0, 1, 0, 1), value = -inf);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_91 = torch.nn.functional.max_pool2d(x_89, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    x_92 = x_91.permute(0, 2, 3, 1);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    x_93 = x_92.reshape(8, 1, 14, 1, 14, 512);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    transpose_4 = x_93.transpose(2, 3);  x_93 = None
    x_95 = transpose_4.reshape(8, 1, -1, 512);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    getattr_l__mod___levels___2___pos_embed = self.getattr_L__mod___levels___2___pos_embed
    x_96 = x_95 + getattr_l__mod___levels___2___pos_embed;  x_95 = getattr_l__mod___levels___2___pos_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___norm1_bias
    y_4 = torch.nn.functional.layer_norm(x_96, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___attn_qkv(y_4);  y_4 = None
    reshape_18 = getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_qkv = None
    qkv_4 = reshape_18.permute(3, 0, 4, 1, 2, 5);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_4 = unbind_4[0]
    k_4 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_98 = torch._C._nn.scaled_dot_product_attention(q_4, k_4, v_4, dropout_p = 0.0);  q_4 = k_4 = v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_18 = x_98.permute(0, 2, 3, 4, 1);  x_98 = None
    x_99 = permute_18.reshape(8, 1, 196, 512);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_100 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___attn_proj(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_101 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___attn_proj_drop(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_102 = x_96 + x_101;  x_96 = x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___norm2_bias
    x_103 = torch.nn.functional.layer_norm(x_102, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___0___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_104 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___mlp_fc1(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_105 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___mlp_act(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_106 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___mlp_drop1(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_107 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___mlp_norm(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_108 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___mlp_fc2(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_109 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___0___mlp_drop2(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_110 = x_102 + x_109;  x_102 = x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___norm1_bias
    y_5 = torch.nn.functional.layer_norm(x_110, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___attn_qkv(y_5);  y_5 = None
    reshape_20 = getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_qkv = None
    qkv_5 = reshape_20.permute(3, 0, 4, 1, 2, 5);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_5 = unbind_5[0]
    k_5 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_112 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_5, dropout_p = 0.0);  q_5 = k_5 = v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_20 = x_112.permute(0, 2, 3, 4, 1);  x_112 = None
    x_113 = permute_20.reshape(8, 1, 196, 512);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_114 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___attn_proj(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_115 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___attn_proj_drop(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_116 = x_110 + x_115;  x_110 = x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___norm2_bias
    x_117 = torch.nn.functional.layer_norm(x_116, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___1___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_118 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___mlp_fc1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_119 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___mlp_act(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_120 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___mlp_drop1(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_121 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___mlp_norm(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_122 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___mlp_fc2(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_123 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___1___mlp_drop2(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_124 = x_116 + x_123;  x_116 = x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___norm1_bias
    y_6 = torch.nn.functional.layer_norm(x_124, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___2___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___attn_qkv(y_6);  y_6 = None
    reshape_22 = getattr_getattr_l__mod___levels___2___transformer_encoder___2___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___2___attn_qkv = None
    qkv_6 = reshape_22.permute(3, 0, 4, 1, 2, 5);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_6 = unbind_6[0]
    k_6 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_126 = torch._C._nn.scaled_dot_product_attention(q_6, k_6, v_6, dropout_p = 0.0);  q_6 = k_6 = v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_22 = x_126.permute(0, 2, 3, 4, 1);  x_126 = None
    x_127 = permute_22.reshape(8, 1, 196, 512);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_128 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___attn_proj(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_129 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___attn_proj_drop(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_130 = x_124 + x_129;  x_124 = x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___norm2_bias
    x_131 = torch.nn.functional.layer_norm(x_130, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___2___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_132 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___mlp_fc1(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_133 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___mlp_act(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_134 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___mlp_drop1(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_135 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___mlp_norm(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_136 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___mlp_fc2(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_137 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___2___mlp_drop2(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_138 = x_130 + x_137;  x_130 = x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___norm1_bias
    y_7 = torch.nn.functional.layer_norm(x_138, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___3___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___attn_qkv(y_7);  y_7 = None
    reshape_24 = getattr_getattr_l__mod___levels___2___transformer_encoder___3___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___3___attn_qkv = None
    qkv_7 = reshape_24.permute(3, 0, 4, 1, 2, 5);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_7 = unbind_7[0]
    k_7 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_140 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_7, dropout_p = 0.0);  q_7 = k_7 = v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_24 = x_140.permute(0, 2, 3, 4, 1);  x_140 = None
    x_141 = permute_24.reshape(8, 1, 196, 512);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_142 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___attn_proj(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_143 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___attn_proj_drop(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_144 = x_138 + x_143;  x_138 = x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___norm2_bias
    x_145 = torch.nn.functional.layer_norm(x_144, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___3___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_146 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___mlp_fc1(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_147 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___mlp_act(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_148 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___mlp_drop1(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_149 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___mlp_norm(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_150 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___mlp_fc2(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_151 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___3___mlp_drop2(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_152 = x_144 + x_151;  x_144 = x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___norm1_bias
    y_8 = torch.nn.functional.layer_norm(x_152, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___4___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___attn_qkv(y_8);  y_8 = None
    reshape_26 = getattr_getattr_l__mod___levels___2___transformer_encoder___4___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___4___attn_qkv = None
    qkv_8 = reshape_26.permute(3, 0, 4, 1, 2, 5);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = qkv_8.unbind(0);  qkv_8 = None
    q_8 = unbind_8[0]
    k_8 = unbind_8[1]
    v_8 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_154 = torch._C._nn.scaled_dot_product_attention(q_8, k_8, v_8, dropout_p = 0.0);  q_8 = k_8 = v_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_26 = x_154.permute(0, 2, 3, 4, 1);  x_154 = None
    x_155 = permute_26.reshape(8, 1, 196, 512);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_156 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___attn_proj(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_157 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___attn_proj_drop(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_158 = x_152 + x_157;  x_152 = x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___norm2_bias
    x_159 = torch.nn.functional.layer_norm(x_158, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___4___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_160 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___mlp_fc1(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_161 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___mlp_act(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_162 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___mlp_drop1(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_163 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___mlp_norm(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_164 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___mlp_fc2(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_165 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___4___mlp_drop2(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_166 = x_158 + x_165;  x_158 = x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___norm1_bias
    y_9 = torch.nn.functional.layer_norm(x_166, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___5___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___attn_qkv(y_9);  y_9 = None
    reshape_28 = getattr_getattr_l__mod___levels___2___transformer_encoder___5___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___5___attn_qkv = None
    qkv_9 = reshape_28.permute(3, 0, 4, 1, 2, 5);  reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = qkv_9.unbind(0);  qkv_9 = None
    q_9 = unbind_9[0]
    k_9 = unbind_9[1]
    v_9 = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_168 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_9, dropout_p = 0.0);  q_9 = k_9 = v_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_28 = x_168.permute(0, 2, 3, 4, 1);  x_168 = None
    x_169 = permute_28.reshape(8, 1, 196, 512);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_170 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___attn_proj(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_171 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___attn_proj_drop(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_172 = x_166 + x_171;  x_166 = x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___norm2_bias
    x_173 = torch.nn.functional.layer_norm(x_172, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___5___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_174 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___mlp_fc1(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_175 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___mlp_act(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_176 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___mlp_drop1(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_177 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___mlp_norm(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_178 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___mlp_fc2(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_179 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___5___mlp_drop2(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_180 = x_172 + x_179;  x_172 = x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___norm1_bias
    y_10 = torch.nn.functional.layer_norm(x_180, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___6___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___attn_qkv(y_10);  y_10 = None
    reshape_30 = getattr_getattr_l__mod___levels___2___transformer_encoder___6___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___6___attn_qkv = None
    qkv_10 = reshape_30.permute(3, 0, 4, 1, 2, 5);  reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = qkv_10.unbind(0);  qkv_10 = None
    q_10 = unbind_10[0]
    k_10 = unbind_10[1]
    v_10 = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_182 = torch._C._nn.scaled_dot_product_attention(q_10, k_10, v_10, dropout_p = 0.0);  q_10 = k_10 = v_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_30 = x_182.permute(0, 2, 3, 4, 1);  x_182 = None
    x_183 = permute_30.reshape(8, 1, 196, 512);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_184 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___attn_proj(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_185 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___attn_proj_drop(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_186 = x_180 + x_185;  x_180 = x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___norm2_bias
    x_187 = torch.nn.functional.layer_norm(x_186, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___6___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_188 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___mlp_fc1(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_189 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___mlp_act(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_190 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___mlp_drop1(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_191 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___mlp_norm(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_192 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___mlp_fc2(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_193 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___6___mlp_drop2(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_194 = x_186 + x_193;  x_186 = x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___norm1_bias
    y_11 = torch.nn.functional.layer_norm(x_194, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___7___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___attn_qkv(y_11);  y_11 = None
    reshape_32 = getattr_getattr_l__mod___levels___2___transformer_encoder___7___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___7___attn_qkv = None
    qkv_11 = reshape_32.permute(3, 0, 4, 1, 2, 5);  reshape_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = qkv_11.unbind(0);  qkv_11 = None
    q_11 = unbind_11[0]
    k_11 = unbind_11[1]
    v_11 = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_196 = torch._C._nn.scaled_dot_product_attention(q_11, k_11, v_11, dropout_p = 0.0);  q_11 = k_11 = v_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_32 = x_196.permute(0, 2, 3, 4, 1);  x_196 = None
    x_197 = permute_32.reshape(8, 1, 196, 512);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_198 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___attn_proj(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_199 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___attn_proj_drop(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_200 = x_194 + x_199;  x_194 = x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___norm2_bias
    x_201 = torch.nn.functional.layer_norm(x_200, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___7___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_202 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___mlp_fc1(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_203 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___mlp_act(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_204 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___mlp_drop1(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_205 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___mlp_norm(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_206 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___mlp_fc2(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_207 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___7___mlp_drop2(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_208 = x_200 + x_207;  x_200 = x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___norm1_bias
    y_12 = torch.nn.functional.layer_norm(x_208, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___8___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___attn_qkv(y_12);  y_12 = None
    reshape_34 = getattr_getattr_l__mod___levels___2___transformer_encoder___8___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___8___attn_qkv = None
    qkv_12 = reshape_34.permute(3, 0, 4, 1, 2, 5);  reshape_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = qkv_12.unbind(0);  qkv_12 = None
    q_12 = unbind_12[0]
    k_12 = unbind_12[1]
    v_12 = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_210 = torch._C._nn.scaled_dot_product_attention(q_12, k_12, v_12, dropout_p = 0.0);  q_12 = k_12 = v_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_34 = x_210.permute(0, 2, 3, 4, 1);  x_210 = None
    x_211 = permute_34.reshape(8, 1, 196, 512);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_212 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___attn_proj(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_213 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___attn_proj_drop(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_214 = x_208 + x_213;  x_208 = x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___norm2_bias
    x_215 = torch.nn.functional.layer_norm(x_214, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___8___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_216 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___mlp_fc1(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_217 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___mlp_act(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_218 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___mlp_drop1(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_219 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___mlp_norm(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_220 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___mlp_fc2(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_221 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___8___mlp_drop2(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_222 = x_214 + x_221;  x_214 = x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___norm1_bias
    y_13 = torch.nn.functional.layer_norm(x_222, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___9___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___attn_qkv(y_13);  y_13 = None
    reshape_36 = getattr_getattr_l__mod___levels___2___transformer_encoder___9___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___9___attn_qkv = None
    qkv_13 = reshape_36.permute(3, 0, 4, 1, 2, 5);  reshape_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = qkv_13.unbind(0);  qkv_13 = None
    q_13 = unbind_13[0]
    k_13 = unbind_13[1]
    v_13 = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_224 = torch._C._nn.scaled_dot_product_attention(q_13, k_13, v_13, dropout_p = 0.0);  q_13 = k_13 = v_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_36 = x_224.permute(0, 2, 3, 4, 1);  x_224 = None
    x_225 = permute_36.reshape(8, 1, 196, 512);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_226 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___attn_proj(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_227 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___attn_proj_drop(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_228 = x_222 + x_227;  x_222 = x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___norm2_bias
    x_229 = torch.nn.functional.layer_norm(x_228, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___9___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_230 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___mlp_fc1(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_231 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___mlp_act(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_232 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___mlp_drop1(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_233 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___mlp_norm(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_234 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___mlp_fc2(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_235 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___9___mlp_drop2(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_236 = x_228 + x_235;  x_228 = x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___norm1_bias
    y_14 = torch.nn.functional.layer_norm(x_236, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___10___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___attn_qkv(y_14);  y_14 = None
    reshape_38 = getattr_getattr_l__mod___levels___2___transformer_encoder___10___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___10___attn_qkv = None
    qkv_14 = reshape_38.permute(3, 0, 4, 1, 2, 5);  reshape_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = qkv_14.unbind(0);  qkv_14 = None
    q_14 = unbind_14[0]
    k_14 = unbind_14[1]
    v_14 = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_238 = torch._C._nn.scaled_dot_product_attention(q_14, k_14, v_14, dropout_p = 0.0);  q_14 = k_14 = v_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_38 = x_238.permute(0, 2, 3, 4, 1);  x_238 = None
    x_239 = permute_38.reshape(8, 1, 196, 512);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_240 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___attn_proj(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_241 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___attn_proj_drop(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_242 = x_236 + x_241;  x_236 = x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___norm2_bias
    x_243 = torch.nn.functional.layer_norm(x_242, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___10___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_244 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___mlp_fc1(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_245 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___mlp_act(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_246 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___mlp_drop1(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_247 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___mlp_norm(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_248 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___mlp_fc2(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_249 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___10___mlp_drop2(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_250 = x_242 + x_249;  x_242 = x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___norm1_bias
    y_15 = torch.nn.functional.layer_norm(x_250, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___11___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___attn_qkv(y_15);  y_15 = None
    reshape_40 = getattr_getattr_l__mod___levels___2___transformer_encoder___11___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___11___attn_qkv = None
    qkv_15 = reshape_40.permute(3, 0, 4, 1, 2, 5);  reshape_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = qkv_15.unbind(0);  qkv_15 = None
    q_15 = unbind_15[0]
    k_15 = unbind_15[1]
    v_15 = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_252 = torch._C._nn.scaled_dot_product_attention(q_15, k_15, v_15, dropout_p = 0.0);  q_15 = k_15 = v_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_40 = x_252.permute(0, 2, 3, 4, 1);  x_252 = None
    x_253 = permute_40.reshape(8, 1, 196, 512);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_254 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___attn_proj(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_255 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___attn_proj_drop(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_256 = x_250 + x_255;  x_250 = x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___norm2_bias
    x_257 = torch.nn.functional.layer_norm(x_256, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___11___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_258 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___mlp_fc1(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_259 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___mlp_act(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_260 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___mlp_drop1(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_261 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___mlp_norm(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_262 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___mlp_fc2(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_263 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___11___mlp_drop2(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_264 = x_256 + x_263;  x_256 = x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___norm1_bias
    y_16 = torch.nn.functional.layer_norm(x_264, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___12___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___attn_qkv(y_16);  y_16 = None
    reshape_42 = getattr_getattr_l__mod___levels___2___transformer_encoder___12___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___12___attn_qkv = None
    qkv_16 = reshape_42.permute(3, 0, 4, 1, 2, 5);  reshape_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = qkv_16.unbind(0);  qkv_16 = None
    q_16 = unbind_16[0]
    k_16 = unbind_16[1]
    v_16 = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_266 = torch._C._nn.scaled_dot_product_attention(q_16, k_16, v_16, dropout_p = 0.0);  q_16 = k_16 = v_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_42 = x_266.permute(0, 2, 3, 4, 1);  x_266 = None
    x_267 = permute_42.reshape(8, 1, 196, 512);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_268 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___attn_proj(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_269 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___attn_proj_drop(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_270 = x_264 + x_269;  x_264 = x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___norm2_bias
    x_271 = torch.nn.functional.layer_norm(x_270, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___12___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_272 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___mlp_fc1(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_273 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___mlp_act(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_274 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___mlp_drop1(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_275 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___mlp_norm(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_276 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___mlp_fc2(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_277 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___12___mlp_drop2(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_278 = x_270 + x_277;  x_270 = x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___norm1_bias
    y_17 = torch.nn.functional.layer_norm(x_278, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___13___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___attn_qkv(y_17);  y_17 = None
    reshape_44 = getattr_getattr_l__mod___levels___2___transformer_encoder___13___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___13___attn_qkv = None
    qkv_17 = reshape_44.permute(3, 0, 4, 1, 2, 5);  reshape_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = qkv_17.unbind(0);  qkv_17 = None
    q_17 = unbind_17[0]
    k_17 = unbind_17[1]
    v_17 = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_280 = torch._C._nn.scaled_dot_product_attention(q_17, k_17, v_17, dropout_p = 0.0);  q_17 = k_17 = v_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_44 = x_280.permute(0, 2, 3, 4, 1);  x_280 = None
    x_281 = permute_44.reshape(8, 1, 196, 512);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_282 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___attn_proj(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_283 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___attn_proj_drop(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_284 = x_278 + x_283;  x_278 = x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___norm2_bias
    x_285 = torch.nn.functional.layer_norm(x_284, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___13___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_286 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___mlp_fc1(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_287 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___mlp_act(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_288 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___mlp_drop1(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_289 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___mlp_norm(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_290 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___mlp_fc2(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_291 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___13___mlp_drop2(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_292 = x_284 + x_291;  x_284 = x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___norm1_bias
    y_18 = torch.nn.functional.layer_norm(x_292, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___14___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___attn_qkv(y_18);  y_18 = None
    reshape_46 = getattr_getattr_l__mod___levels___2___transformer_encoder___14___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___14___attn_qkv = None
    qkv_18 = reshape_46.permute(3, 0, 4, 1, 2, 5);  reshape_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = qkv_18.unbind(0);  qkv_18 = None
    q_18 = unbind_18[0]
    k_18 = unbind_18[1]
    v_18 = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_294 = torch._C._nn.scaled_dot_product_attention(q_18, k_18, v_18, dropout_p = 0.0);  q_18 = k_18 = v_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_46 = x_294.permute(0, 2, 3, 4, 1);  x_294 = None
    x_295 = permute_46.reshape(8, 1, 196, 512);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_296 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___attn_proj(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_297 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___attn_proj_drop(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_298 = x_292 + x_297;  x_292 = x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___norm2_bias
    x_299 = torch.nn.functional.layer_norm(x_298, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___14___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_300 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___mlp_fc1(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_301 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___mlp_act(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_302 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___mlp_drop1(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_303 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___mlp_norm(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_304 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___mlp_fc2(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_305 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___14___mlp_drop2(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_306 = x_298 + x_305;  x_298 = x_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___norm1_bias
    y_19 = torch.nn.functional.layer_norm(x_306, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___15___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___attn_qkv(y_19);  y_19 = None
    reshape_48 = getattr_getattr_l__mod___levels___2___transformer_encoder___15___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___15___attn_qkv = None
    qkv_19 = reshape_48.permute(3, 0, 4, 1, 2, 5);  reshape_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = qkv_19.unbind(0);  qkv_19 = None
    q_19 = unbind_19[0]
    k_19 = unbind_19[1]
    v_19 = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_308 = torch._C._nn.scaled_dot_product_attention(q_19, k_19, v_19, dropout_p = 0.0);  q_19 = k_19 = v_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_48 = x_308.permute(0, 2, 3, 4, 1);  x_308 = None
    x_309 = permute_48.reshape(8, 1, 196, 512);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_310 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___attn_proj(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_311 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___attn_proj_drop(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_312 = x_306 + x_311;  x_306 = x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___norm2_bias
    x_313 = torch.nn.functional.layer_norm(x_312, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___15___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_314 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___mlp_fc1(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_315 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___mlp_act(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_316 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___mlp_drop1(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_317 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___mlp_norm(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_318 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___mlp_fc2(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_319 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___15___mlp_drop2(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_320 = x_312 + x_319;  x_312 = x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___norm1_bias
    y_20 = torch.nn.functional.layer_norm(x_320, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___16___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___attn_qkv(y_20);  y_20 = None
    reshape_50 = getattr_getattr_l__mod___levels___2___transformer_encoder___16___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___16___attn_qkv = None
    qkv_20 = reshape_50.permute(3, 0, 4, 1, 2, 5);  reshape_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = qkv_20.unbind(0);  qkv_20 = None
    q_20 = unbind_20[0]
    k_20 = unbind_20[1]
    v_20 = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_322 = torch._C._nn.scaled_dot_product_attention(q_20, k_20, v_20, dropout_p = 0.0);  q_20 = k_20 = v_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_50 = x_322.permute(0, 2, 3, 4, 1);  x_322 = None
    x_323 = permute_50.reshape(8, 1, 196, 512);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_324 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___attn_proj(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_325 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___attn_proj_drop(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_326 = x_320 + x_325;  x_320 = x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___norm2_bias
    x_327 = torch.nn.functional.layer_norm(x_326, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___16___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_328 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___mlp_fc1(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_329 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___mlp_act(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_330 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___mlp_drop1(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_331 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___mlp_norm(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_332 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___mlp_fc2(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_333 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___16___mlp_drop2(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_334 = x_326 + x_333;  x_326 = x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___norm1_bias
    y_21 = torch.nn.functional.layer_norm(x_334, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___17___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___attn_qkv(y_21);  y_21 = None
    reshape_52 = getattr_getattr_l__mod___levels___2___transformer_encoder___17___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___17___attn_qkv = None
    qkv_21 = reshape_52.permute(3, 0, 4, 1, 2, 5);  reshape_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = qkv_21.unbind(0);  qkv_21 = None
    q_21 = unbind_21[0]
    k_21 = unbind_21[1]
    v_21 = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_336 = torch._C._nn.scaled_dot_product_attention(q_21, k_21, v_21, dropout_p = 0.0);  q_21 = k_21 = v_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_52 = x_336.permute(0, 2, 3, 4, 1);  x_336 = None
    x_337 = permute_52.reshape(8, 1, 196, 512);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_338 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___attn_proj(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_339 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___attn_proj_drop(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_340 = x_334 + x_339;  x_334 = x_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___norm2_bias
    x_341 = torch.nn.functional.layer_norm(x_340, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___17___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_342 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___mlp_fc1(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_343 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___mlp_act(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_344 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___mlp_drop1(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_345 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___mlp_norm(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_346 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___mlp_fc2(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_347 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___17___mlp_drop2(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_348 = x_340 + x_347;  x_340 = x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___norm1_bias
    y_22 = torch.nn.functional.layer_norm(x_348, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___18___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___attn_qkv(y_22);  y_22 = None
    reshape_54 = getattr_getattr_l__mod___levels___2___transformer_encoder___18___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___18___attn_qkv = None
    qkv_22 = reshape_54.permute(3, 0, 4, 1, 2, 5);  reshape_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = qkv_22.unbind(0);  qkv_22 = None
    q_22 = unbind_22[0]
    k_22 = unbind_22[1]
    v_22 = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_350 = torch._C._nn.scaled_dot_product_attention(q_22, k_22, v_22, dropout_p = 0.0);  q_22 = k_22 = v_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_54 = x_350.permute(0, 2, 3, 4, 1);  x_350 = None
    x_351 = permute_54.reshape(8, 1, 196, 512);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_352 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___attn_proj(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_353 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___attn_proj_drop(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_354 = x_348 + x_353;  x_348 = x_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___norm2_bias
    x_355 = torch.nn.functional.layer_norm(x_354, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___18___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_356 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___mlp_fc1(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_357 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___mlp_act(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_358 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___mlp_drop1(x_357);  x_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_359 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___mlp_norm(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_360 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___mlp_fc2(x_359);  x_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_361 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___18___mlp_drop2(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_362 = x_354 + x_361;  x_354 = x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm1_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___norm1_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm1_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___norm1_bias
    y_23 = torch.nn.functional.layer_norm(x_362, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm1_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm1_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm1_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    getattr_getattr_l__mod___levels___2___transformer_encoder___19___attn_qkv = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___attn_qkv(y_23);  y_23 = None
    reshape_56 = getattr_getattr_l__mod___levels___2___transformer_encoder___19___attn_qkv.reshape(8, 1, 196, 3, 16, 32);  getattr_getattr_l__mod___levels___2___transformer_encoder___19___attn_qkv = None
    qkv_23 = reshape_56.permute(3, 0, 4, 1, 2, 5);  reshape_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = qkv_23.unbind(0);  qkv_23 = None
    q_23 = unbind_23[0]
    k_23 = unbind_23[1]
    v_23 = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    x_364 = torch._C._nn.scaled_dot_product_attention(q_23, k_23, v_23, dropout_p = 0.0);  q_23 = k_23 = v_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    permute_56 = x_364.permute(0, 2, 3, 4, 1);  x_364 = None
    x_365 = permute_56.reshape(8, 1, 196, 512);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    x_366 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___attn_proj(x_365);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:80, code: x = self.proj_drop(x)
    x_367 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___attn_proj_drop(x_366);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:123, code: x = x + self.drop_path(self.attn(y))
    x_368 = x_362 + x_367;  x_362 = x_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm2_weight = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___norm2_weight
    getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm2_bias = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___norm2_bias
    x_369 = torch.nn.functional.layer_norm(x_368, (512,), getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm2_weight, getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm2_bias, 1e-06);  getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm2_weight = getattr_getattr_l__mod___levels___2___transformer_encoder___19___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_370 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___mlp_fc1(x_369);  x_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_371 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___mlp_act(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_372 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___mlp_drop1(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_373 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___mlp_norm(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_374 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___mlp_fc2(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_375 = self.getattr_getattr_L__mod___levels___2___transformer_encoder___19___mlp_drop2(x_374);  x_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:124, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    x_377 = x_368 + x_375;  x_368 = x_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x_378 = x_377.reshape(8, 1, 1, 14, 14, 512);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    transpose_5 = x_378.transpose(2, 3);  x_378 = None
    x_380 = transpose_5.reshape(8, 14, 14, 512);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    x_381 = x_380.permute(0, 3, 1, 2);  x_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_58 = x_381.permute(0, 2, 3, 1);  x_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___norm_weight = self.L__mod___norm_weight
    l__mod___norm_bias = self.L__mod___norm_bias
    x_382 = torch.nn.functional.layer_norm(permute_58, (512,), l__mod___norm_weight, l__mod___norm_bias, 1e-06);  permute_58 = l__mod___norm_weight = l__mod___norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    x_384 = x_382.permute(0, 3, 1, 2);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_385 = self.L__mod___global_pool_pool(x_384);  x_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_387 = self.L__mod___global_pool_flatten(x_385);  x_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:432, code: x = self.head_drop(x)
    x_388 = self.L__mod___head_drop(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:433, code: return x if pre_logits else self.head(x)
    x_389 = self.L__mod___head(x_388);  x_388 = None
    return (x_389,)
    