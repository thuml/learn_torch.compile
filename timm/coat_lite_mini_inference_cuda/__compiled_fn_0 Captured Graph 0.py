from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x = self.L__mod___patch_embed1_proj(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten = x.flatten(2);  x = None
    x_1 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x1 = self.L__mod___patch_embed1_norm(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:576, code: x1 = insert_cls(x1, self.cls_token1)
    l__mod___cls_token1 = self.L__mod___cls_token1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    cls_tokens = l__mod___cls_token1.expand(8, -1, -1);  l__mod___cls_token1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    x1_1 = torch.cat((cls_tokens, x1), dim = 1);  cls_tokens = x1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    cls_token = x1_1[(slice(None, None, None), slice(None, 1, None))]
    img_tokens = x1_1[(slice(None, None, None), slice(1, None, None))];  x1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    transpose_1 = img_tokens.transpose(1, 2);  img_tokens = None
    feat = transpose_1.view(8, 64, 56, 56);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    l__mod___serial_blocks1_0_cpe_proj = self.L__mod___serial_blocks1_0_cpe_proj(feat)
    x_4 = l__mod___serial_blocks1_0_cpe_proj + feat;  l__mod___serial_blocks1_0_cpe_proj = feat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    flatten_1 = x_4.flatten(2);  x_4 = None
    x_5 = flatten_1.transpose(1, 2);  flatten_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    x_7 = torch.cat((cls_token, x_5), dim = 1);  cls_token = x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks1_0_norm1_weight = self.L__mod___serial_blocks1_0_norm1_weight
    l__mod___serial_blocks1_0_norm1_bias = self.L__mod___serial_blocks1_0_norm1_bias
    cur = torch.nn.functional.layer_norm(x_7, (64,), l__mod___serial_blocks1_0_norm1_weight, l__mod___serial_blocks1_0_norm1_bias, 1e-06);  l__mod___serial_blocks1_0_norm1_weight = l__mod___serial_blocks1_0_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___serial_blocks1_0_factoratt_crpe_qkv = self.L__mod___serial_blocks1_0_factoratt_crpe_qkv(cur);  cur = None
    reshape = l__mod___serial_blocks1_0_factoratt_crpe_qkv.reshape(8, 3137, 3, 8, 8);  l__mod___serial_blocks1_0_factoratt_crpe_qkv = None
    qkv = reshape.permute(2, 0, 3, 1, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    k_softmax = k.softmax(dim = 2);  k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    transpose_3 = k_softmax.transpose(-1, -2);  k_softmax = None
    factor_att = transpose_3 @ v;  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    factor_att_1 = q @ factor_att;  factor_att = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    q_img = q[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  q = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    v_img = v[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    transpose_4 = v_img.transpose(-1, -2);  v_img = None
    v_img_1 = transpose_4.reshape(8, 64, 56, 56);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split = torch.functional.split(v_img_1, [16, 24, 24], dim = 1);  v_img_1 = None
    getitem_7 = split[0]
    getitem_8 = split[1]
    getitem_9 = split[2];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0 = self.L__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0(getitem_7);  getitem_7 = None
    l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1 = self.L__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1(getitem_8);  getitem_8 = None
    l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2 = self.L__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2(getitem_9);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    conv_v_img = torch.cat([l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0, l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1, l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2], dim = 1);  l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0 = l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1 = l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    reshape_2 = conv_v_img.reshape(8, 8, 8, 3136);  conv_v_img = None
    conv_v_img_1 = reshape_2.transpose(-1, -2);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    EV_hat = q_img * conv_v_img_1;  q_img = conv_v_img_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    crpe = torch.nn.functional.pad(EV_hat, (0, 0, 1, 0, 0, 0));  EV_hat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_1 = 0.3535533905932738 * factor_att_1;  factor_att_1 = None
    x_9 = mul_1 + crpe;  mul_1 = crpe = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    transpose_6 = x_9.transpose(1, 2);  x_9 = None
    x_10 = transpose_6.reshape(8, 3137, 64);  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    x_11 = self.L__mod___serial_blocks1_0_factoratt_crpe_proj(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    cur_1 = self.L__mod___serial_blocks1_0_factoratt_crpe_proj_drop(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks1_0_drop_path = self.L__mod___serial_blocks1_0_drop_path(cur_1);  cur_1 = None
    x_13 = x_7 + l__mod___serial_blocks1_0_drop_path;  x_7 = l__mod___serial_blocks1_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks1_0_norm2_weight = self.L__mod___serial_blocks1_0_norm2_weight
    l__mod___serial_blocks1_0_norm2_bias = self.L__mod___serial_blocks1_0_norm2_bias
    cur_2 = torch.nn.functional.layer_norm(x_13, (64,), l__mod___serial_blocks1_0_norm2_weight, l__mod___serial_blocks1_0_norm2_bias, 1e-06);  l__mod___serial_blocks1_0_norm2_weight = l__mod___serial_blocks1_0_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_15 = self.L__mod___serial_blocks1_0_mlp_fc1(cur_2);  cur_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_16 = self.L__mod___serial_blocks1_0_mlp_act(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_17 = self.L__mod___serial_blocks1_0_mlp_drop1(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_18 = self.L__mod___serial_blocks1_0_mlp_norm(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_19 = self.L__mod___serial_blocks1_0_mlp_fc2(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    cur_3 = self.L__mod___serial_blocks1_0_mlp_drop2(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks1_0_drop_path_1 = self.L__mod___serial_blocks1_0_drop_path(cur_3);  cur_3 = None
    x1_2 = x_13 + l__mod___serial_blocks1_0_drop_path_1;  x_13 = l__mod___serial_blocks1_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    cls_token_1 = x1_2[(slice(None, None, None), slice(None, 1, None))]
    img_tokens_1 = x1_2[(slice(None, None, None), slice(1, None, None))];  x1_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    transpose_7 = img_tokens_1.transpose(1, 2);  img_tokens_1 = None
    feat_1 = transpose_7.view(8, 64, 56, 56);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    l__mod___serial_blocks1_0_cpe_proj_1 = self.L__mod___serial_blocks1_0_cpe_proj(feat_1)
    x_22 = l__mod___serial_blocks1_0_cpe_proj_1 + feat_1;  l__mod___serial_blocks1_0_cpe_proj_1 = feat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    flatten_2 = x_22.flatten(2);  x_22 = None
    x_23 = flatten_2.transpose(1, 2);  flatten_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    x_25 = torch.cat((cls_token_1, x_23), dim = 1);  cls_token_1 = x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks1_1_norm1_weight = self.L__mod___serial_blocks1_1_norm1_weight
    l__mod___serial_blocks1_1_norm1_bias = self.L__mod___serial_blocks1_1_norm1_bias
    cur_4 = torch.nn.functional.layer_norm(x_25, (64,), l__mod___serial_blocks1_1_norm1_weight, l__mod___serial_blocks1_1_norm1_bias, 1e-06);  l__mod___serial_blocks1_1_norm1_weight = l__mod___serial_blocks1_1_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___serial_blocks1_1_factoratt_crpe_qkv = self.L__mod___serial_blocks1_1_factoratt_crpe_qkv(cur_4);  cur_4 = None
    reshape_4 = l__mod___serial_blocks1_1_factoratt_crpe_qkv.reshape(8, 3137, 3, 8, 8);  l__mod___serial_blocks1_1_factoratt_crpe_qkv = None
    qkv_1 = reshape_4.permute(2, 0, 3, 1, 4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_1 = unbind_1[0]
    k_1 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    k_softmax_1 = k_1.softmax(dim = 2);  k_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    transpose_9 = k_softmax_1.transpose(-1, -2);  k_softmax_1 = None
    factor_att_2 = transpose_9 @ v_1;  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    factor_att_3 = q_1 @ factor_att_2;  factor_att_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    q_img_1 = q_1[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  q_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    v_img_2 = v_1[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    transpose_10 = v_img_2.transpose(-1, -2);  v_img_2 = None
    v_img_3 = transpose_10.reshape(8, 64, 56, 56);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_1 = torch.functional.split(v_img_3, [16, 24, 24], dim = 1);  v_img_3 = None
    getitem_17 = split_1[0]
    getitem_18 = split_1[1]
    getitem_19 = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_3 = self.L__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0(getitem_17);  getitem_17 = None
    l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_4 = self.L__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1(getitem_18);  getitem_18 = None
    l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_5 = self.L__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2(getitem_19);  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    conv_v_img_2 = torch.cat([l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_3, l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_4, l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_5], dim = 1);  l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_3 = l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_4 = l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    reshape_6 = conv_v_img_2.reshape(8, 8, 8, 3136);  conv_v_img_2 = None
    conv_v_img_3 = reshape_6.transpose(-1, -2);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    EV_hat_2 = q_img_1 * conv_v_img_3;  q_img_1 = conv_v_img_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    crpe_1 = torch.nn.functional.pad(EV_hat_2, (0, 0, 1, 0, 0, 0));  EV_hat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_3 = 0.3535533905932738 * factor_att_3;  factor_att_3 = None
    x_27 = mul_3 + crpe_1;  mul_3 = crpe_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    transpose_12 = x_27.transpose(1, 2);  x_27 = None
    x_28 = transpose_12.reshape(8, 3137, 64);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    x_29 = self.L__mod___serial_blocks1_1_factoratt_crpe_proj(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    cur_5 = self.L__mod___serial_blocks1_1_factoratt_crpe_proj_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks1_1_drop_path = self.L__mod___serial_blocks1_1_drop_path(cur_5);  cur_5 = None
    x_31 = x_25 + l__mod___serial_blocks1_1_drop_path;  x_25 = l__mod___serial_blocks1_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks1_1_norm2_weight = self.L__mod___serial_blocks1_1_norm2_weight
    l__mod___serial_blocks1_1_norm2_bias = self.L__mod___serial_blocks1_1_norm2_bias
    cur_6 = torch.nn.functional.layer_norm(x_31, (64,), l__mod___serial_blocks1_1_norm2_weight, l__mod___serial_blocks1_1_norm2_bias, 1e-06);  l__mod___serial_blocks1_1_norm2_weight = l__mod___serial_blocks1_1_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_33 = self.L__mod___serial_blocks1_1_mlp_fc1(cur_6);  cur_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_34 = self.L__mod___serial_blocks1_1_mlp_act(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_35 = self.L__mod___serial_blocks1_1_mlp_drop1(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_36 = self.L__mod___serial_blocks1_1_mlp_norm(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_37 = self.L__mod___serial_blocks1_1_mlp_fc2(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    cur_7 = self.L__mod___serial_blocks1_1_mlp_drop2(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks1_1_drop_path_1 = self.L__mod___serial_blocks1_1_drop_path(cur_7);  cur_7 = None
    x1_3 = x_31 + l__mod___serial_blocks1_1_drop_path_1;  x_31 = l__mod___serial_blocks1_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    getitem_20 = x1_3[(slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x1_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:579, code: x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
    reshape_8 = getitem_20.reshape(8, 56, 56, -1);  getitem_20 = None
    permute_2 = reshape_8.permute(0, 3, 1, 2);  reshape_8 = None
    x1_nocls = permute_2.contiguous();  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x_40 = self.L__mod___patch_embed2_proj(x1_nocls);  x1_nocls = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten_3 = x_40.flatten(2);  x_40 = None
    x_41 = flatten_3.transpose(1, 2);  flatten_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x2 = self.L__mod___patch_embed2_norm(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:584, code: x2 = insert_cls(x2, self.cls_token2)
    l__mod___cls_token2 = self.L__mod___cls_token2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    cls_tokens_1 = l__mod___cls_token2.expand(8, -1, -1);  l__mod___cls_token2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    x2_1 = torch.cat((cls_tokens_1, x2), dim = 1);  cls_tokens_1 = x2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    cls_token_2 = x2_1[(slice(None, None, None), slice(None, 1, None))]
    img_tokens_2 = x2_1[(slice(None, None, None), slice(1, None, None))];  x2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    transpose_14 = img_tokens_2.transpose(1, 2);  img_tokens_2 = None
    feat_2 = transpose_14.view(8, 128, 28, 28);  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    l__mod___serial_blocks2_0_cpe_proj = self.L__mod___serial_blocks2_0_cpe_proj(feat_2)
    x_44 = l__mod___serial_blocks2_0_cpe_proj + feat_2;  l__mod___serial_blocks2_0_cpe_proj = feat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    flatten_4 = x_44.flatten(2);  x_44 = None
    x_45 = flatten_4.transpose(1, 2);  flatten_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    x_47 = torch.cat((cls_token_2, x_45), dim = 1);  cls_token_2 = x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks2_0_norm1_weight = self.L__mod___serial_blocks2_0_norm1_weight
    l__mod___serial_blocks2_0_norm1_bias = self.L__mod___serial_blocks2_0_norm1_bias
    cur_8 = torch.nn.functional.layer_norm(x_47, (128,), l__mod___serial_blocks2_0_norm1_weight, l__mod___serial_blocks2_0_norm1_bias, 1e-06);  l__mod___serial_blocks2_0_norm1_weight = l__mod___serial_blocks2_0_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___serial_blocks2_0_factoratt_crpe_qkv = self.L__mod___serial_blocks2_0_factoratt_crpe_qkv(cur_8);  cur_8 = None
    reshape_9 = l__mod___serial_blocks2_0_factoratt_crpe_qkv.reshape(8, 785, 3, 8, 16);  l__mod___serial_blocks2_0_factoratt_crpe_qkv = None
    qkv_2 = reshape_9.permute(2, 0, 3, 1, 4);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_2 = unbind_2[0]
    k_2 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    k_softmax_2 = k_2.softmax(dim = 2);  k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    transpose_16 = k_softmax_2.transpose(-1, -2);  k_softmax_2 = None
    factor_att_4 = transpose_16 @ v_2;  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    factor_att_5 = q_2 @ factor_att_4;  factor_att_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    q_img_2 = q_2[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    v_img_4 = v_2[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    transpose_17 = v_img_4.transpose(-1, -2);  v_img_4 = None
    v_img_5 = transpose_17.reshape(8, 128, 28, 28);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_2 = torch.functional.split(v_img_5, [32, 48, 48], dim = 1);  v_img_5 = None
    getitem_28 = split_2[0]
    getitem_29 = split_2[1]
    getitem_30 = split_2[2];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0 = self.L__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0(getitem_28);  getitem_28 = None
    l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1 = self.L__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1(getitem_29);  getitem_29 = None
    l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2 = self.L__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2(getitem_30);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    conv_v_img_4 = torch.cat([l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0, l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1, l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2], dim = 1);  l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0 = l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1 = l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    reshape_11 = conv_v_img_4.reshape(8, 8, 16, 784);  conv_v_img_4 = None
    conv_v_img_5 = reshape_11.transpose(-1, -2);  reshape_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    EV_hat_4 = q_img_2 * conv_v_img_5;  q_img_2 = conv_v_img_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    crpe_2 = torch.nn.functional.pad(EV_hat_4, (0, 0, 1, 0, 0, 0));  EV_hat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_5 = 0.25 * factor_att_5;  factor_att_5 = None
    x_49 = mul_5 + crpe_2;  mul_5 = crpe_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    transpose_19 = x_49.transpose(1, 2);  x_49 = None
    x_50 = transpose_19.reshape(8, 785, 128);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    x_51 = self.L__mod___serial_blocks2_0_factoratt_crpe_proj(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    cur_9 = self.L__mod___serial_blocks2_0_factoratt_crpe_proj_drop(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks2_0_drop_path = self.L__mod___serial_blocks2_0_drop_path(cur_9);  cur_9 = None
    x_53 = x_47 + l__mod___serial_blocks2_0_drop_path;  x_47 = l__mod___serial_blocks2_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks2_0_norm2_weight = self.L__mod___serial_blocks2_0_norm2_weight
    l__mod___serial_blocks2_0_norm2_bias = self.L__mod___serial_blocks2_0_norm2_bias
    cur_10 = torch.nn.functional.layer_norm(x_53, (128,), l__mod___serial_blocks2_0_norm2_weight, l__mod___serial_blocks2_0_norm2_bias, 1e-06);  l__mod___serial_blocks2_0_norm2_weight = l__mod___serial_blocks2_0_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_55 = self.L__mod___serial_blocks2_0_mlp_fc1(cur_10);  cur_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_56 = self.L__mod___serial_blocks2_0_mlp_act(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_57 = self.L__mod___serial_blocks2_0_mlp_drop1(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_58 = self.L__mod___serial_blocks2_0_mlp_norm(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_59 = self.L__mod___serial_blocks2_0_mlp_fc2(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    cur_11 = self.L__mod___serial_blocks2_0_mlp_drop2(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks2_0_drop_path_1 = self.L__mod___serial_blocks2_0_drop_path(cur_11);  cur_11 = None
    x2_2 = x_53 + l__mod___serial_blocks2_0_drop_path_1;  x_53 = l__mod___serial_blocks2_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    cls_token_3 = x2_2[(slice(None, None, None), slice(None, 1, None))]
    img_tokens_3 = x2_2[(slice(None, None, None), slice(1, None, None))];  x2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    transpose_20 = img_tokens_3.transpose(1, 2);  img_tokens_3 = None
    feat_3 = transpose_20.view(8, 128, 28, 28);  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    l__mod___serial_blocks2_0_cpe_proj_1 = self.L__mod___serial_blocks2_0_cpe_proj(feat_3)
    x_62 = l__mod___serial_blocks2_0_cpe_proj_1 + feat_3;  l__mod___serial_blocks2_0_cpe_proj_1 = feat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    flatten_5 = x_62.flatten(2);  x_62 = None
    x_63 = flatten_5.transpose(1, 2);  flatten_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    x_65 = torch.cat((cls_token_3, x_63), dim = 1);  cls_token_3 = x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks2_1_norm1_weight = self.L__mod___serial_blocks2_1_norm1_weight
    l__mod___serial_blocks2_1_norm1_bias = self.L__mod___serial_blocks2_1_norm1_bias
    cur_12 = torch.nn.functional.layer_norm(x_65, (128,), l__mod___serial_blocks2_1_norm1_weight, l__mod___serial_blocks2_1_norm1_bias, 1e-06);  l__mod___serial_blocks2_1_norm1_weight = l__mod___serial_blocks2_1_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___serial_blocks2_1_factoratt_crpe_qkv = self.L__mod___serial_blocks2_1_factoratt_crpe_qkv(cur_12);  cur_12 = None
    reshape_13 = l__mod___serial_blocks2_1_factoratt_crpe_qkv.reshape(8, 785, 3, 8, 16);  l__mod___serial_blocks2_1_factoratt_crpe_qkv = None
    qkv_3 = reshape_13.permute(2, 0, 3, 1, 4);  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_3 = unbind_3[0]
    k_3 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    k_softmax_3 = k_3.softmax(dim = 2);  k_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    transpose_22 = k_softmax_3.transpose(-1, -2);  k_softmax_3 = None
    factor_att_6 = transpose_22 @ v_3;  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    factor_att_7 = q_3 @ factor_att_6;  factor_att_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    q_img_3 = q_3[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  q_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    v_img_6 = v_3[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    transpose_23 = v_img_6.transpose(-1, -2);  v_img_6 = None
    v_img_7 = transpose_23.reshape(8, 128, 28, 28);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_3 = torch.functional.split(v_img_7, [32, 48, 48], dim = 1);  v_img_7 = None
    getitem_38 = split_3[0]
    getitem_39 = split_3[1]
    getitem_40 = split_3[2];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_3 = self.L__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0(getitem_38);  getitem_38 = None
    l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_4 = self.L__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1(getitem_39);  getitem_39 = None
    l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_5 = self.L__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2(getitem_40);  getitem_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    conv_v_img_6 = torch.cat([l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_3, l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_4, l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_5], dim = 1);  l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_3 = l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_4 = l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    reshape_15 = conv_v_img_6.reshape(8, 8, 16, 784);  conv_v_img_6 = None
    conv_v_img_7 = reshape_15.transpose(-1, -2);  reshape_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    EV_hat_6 = q_img_3 * conv_v_img_7;  q_img_3 = conv_v_img_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    crpe_3 = torch.nn.functional.pad(EV_hat_6, (0, 0, 1, 0, 0, 0));  EV_hat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_7 = 0.25 * factor_att_7;  factor_att_7 = None
    x_67 = mul_7 + crpe_3;  mul_7 = crpe_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    transpose_25 = x_67.transpose(1, 2);  x_67 = None
    x_68 = transpose_25.reshape(8, 785, 128);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    x_69 = self.L__mod___serial_blocks2_1_factoratt_crpe_proj(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    cur_13 = self.L__mod___serial_blocks2_1_factoratt_crpe_proj_drop(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks2_1_drop_path = self.L__mod___serial_blocks2_1_drop_path(cur_13);  cur_13 = None
    x_71 = x_65 + l__mod___serial_blocks2_1_drop_path;  x_65 = l__mod___serial_blocks2_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks2_1_norm2_weight = self.L__mod___serial_blocks2_1_norm2_weight
    l__mod___serial_blocks2_1_norm2_bias = self.L__mod___serial_blocks2_1_norm2_bias
    cur_14 = torch.nn.functional.layer_norm(x_71, (128,), l__mod___serial_blocks2_1_norm2_weight, l__mod___serial_blocks2_1_norm2_bias, 1e-06);  l__mod___serial_blocks2_1_norm2_weight = l__mod___serial_blocks2_1_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_73 = self.L__mod___serial_blocks2_1_mlp_fc1(cur_14);  cur_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_74 = self.L__mod___serial_blocks2_1_mlp_act(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_75 = self.L__mod___serial_blocks2_1_mlp_drop1(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_76 = self.L__mod___serial_blocks2_1_mlp_norm(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_77 = self.L__mod___serial_blocks2_1_mlp_fc2(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    cur_15 = self.L__mod___serial_blocks2_1_mlp_drop2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks2_1_drop_path_1 = self.L__mod___serial_blocks2_1_drop_path(cur_15);  cur_15 = None
    x2_3 = x_71 + l__mod___serial_blocks2_1_drop_path_1;  x_71 = l__mod___serial_blocks2_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    getitem_41 = x2_3[(slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x2_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:587, code: x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
    reshape_17 = getitem_41.reshape(8, 28, 28, -1);  getitem_41 = None
    permute_5 = reshape_17.permute(0, 3, 1, 2);  reshape_17 = None
    x2_nocls = permute_5.contiguous();  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x_80 = self.L__mod___patch_embed3_proj(x2_nocls);  x2_nocls = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten_6 = x_80.flatten(2);  x_80 = None
    x_81 = flatten_6.transpose(1, 2);  flatten_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x3 = self.L__mod___patch_embed3_norm(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:592, code: x3 = insert_cls(x3, self.cls_token3)
    l__mod___cls_token3 = self.L__mod___cls_token3
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    cls_tokens_2 = l__mod___cls_token3.expand(8, -1, -1);  l__mod___cls_token3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    x3_1 = torch.cat((cls_tokens_2, x3), dim = 1);  cls_tokens_2 = x3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    cls_token_4 = x3_1[(slice(None, None, None), slice(None, 1, None))]
    img_tokens_4 = x3_1[(slice(None, None, None), slice(1, None, None))];  x3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    transpose_27 = img_tokens_4.transpose(1, 2);  img_tokens_4 = None
    feat_4 = transpose_27.view(8, 320, 14, 14);  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    l__mod___serial_blocks3_0_cpe_proj = self.L__mod___serial_blocks3_0_cpe_proj(feat_4)
    x_84 = l__mod___serial_blocks3_0_cpe_proj + feat_4;  l__mod___serial_blocks3_0_cpe_proj = feat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    flatten_7 = x_84.flatten(2);  x_84 = None
    x_85 = flatten_7.transpose(1, 2);  flatten_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    x_87 = torch.cat((cls_token_4, x_85), dim = 1);  cls_token_4 = x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks3_0_norm1_weight = self.L__mod___serial_blocks3_0_norm1_weight
    l__mod___serial_blocks3_0_norm1_bias = self.L__mod___serial_blocks3_0_norm1_bias
    cur_16 = torch.nn.functional.layer_norm(x_87, (320,), l__mod___serial_blocks3_0_norm1_weight, l__mod___serial_blocks3_0_norm1_bias, 1e-06);  l__mod___serial_blocks3_0_norm1_weight = l__mod___serial_blocks3_0_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___serial_blocks3_0_factoratt_crpe_qkv = self.L__mod___serial_blocks3_0_factoratt_crpe_qkv(cur_16);  cur_16 = None
    reshape_18 = l__mod___serial_blocks3_0_factoratt_crpe_qkv.reshape(8, 197, 3, 8, 40);  l__mod___serial_blocks3_0_factoratt_crpe_qkv = None
    qkv_4 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_4 = unbind_4[0]
    k_4 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    k_softmax_4 = k_4.softmax(dim = 2);  k_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    transpose_29 = k_softmax_4.transpose(-1, -2);  k_softmax_4 = None
    factor_att_8 = transpose_29 @ v_4;  transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    factor_att_9 = q_4 @ factor_att_8;  factor_att_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    q_img_4 = q_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  q_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    v_img_8 = v_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    transpose_30 = v_img_8.transpose(-1, -2);  v_img_8 = None
    v_img_9 = transpose_30.reshape(8, 320, 14, 14);  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_4 = torch.functional.split(v_img_9, [80, 120, 120], dim = 1);  v_img_9 = None
    getitem_49 = split_4[0]
    getitem_50 = split_4[1]
    getitem_51 = split_4[2];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0 = self.L__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0(getitem_49);  getitem_49 = None
    l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1 = self.L__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1(getitem_50);  getitem_50 = None
    l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2 = self.L__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2(getitem_51);  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    conv_v_img_8 = torch.cat([l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0, l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1, l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2], dim = 1);  l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0 = l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1 = l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    reshape_20 = conv_v_img_8.reshape(8, 8, 40, 196);  conv_v_img_8 = None
    conv_v_img_9 = reshape_20.transpose(-1, -2);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    EV_hat_8 = q_img_4 * conv_v_img_9;  q_img_4 = conv_v_img_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    crpe_4 = torch.nn.functional.pad(EV_hat_8, (0, 0, 1, 0, 0, 0));  EV_hat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_9 = 0.15811388300841897 * factor_att_9;  factor_att_9 = None
    x_89 = mul_9 + crpe_4;  mul_9 = crpe_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    transpose_32 = x_89.transpose(1, 2);  x_89 = None
    x_90 = transpose_32.reshape(8, 197, 320);  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    x_91 = self.L__mod___serial_blocks3_0_factoratt_crpe_proj(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    cur_17 = self.L__mod___serial_blocks3_0_factoratt_crpe_proj_drop(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks3_0_drop_path = self.L__mod___serial_blocks3_0_drop_path(cur_17);  cur_17 = None
    x_93 = x_87 + l__mod___serial_blocks3_0_drop_path;  x_87 = l__mod___serial_blocks3_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks3_0_norm2_weight = self.L__mod___serial_blocks3_0_norm2_weight
    l__mod___serial_blocks3_0_norm2_bias = self.L__mod___serial_blocks3_0_norm2_bias
    cur_18 = torch.nn.functional.layer_norm(x_93, (320,), l__mod___serial_blocks3_0_norm2_weight, l__mod___serial_blocks3_0_norm2_bias, 1e-06);  l__mod___serial_blocks3_0_norm2_weight = l__mod___serial_blocks3_0_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_95 = self.L__mod___serial_blocks3_0_mlp_fc1(cur_18);  cur_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_96 = self.L__mod___serial_blocks3_0_mlp_act(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_97 = self.L__mod___serial_blocks3_0_mlp_drop1(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_98 = self.L__mod___serial_blocks3_0_mlp_norm(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_99 = self.L__mod___serial_blocks3_0_mlp_fc2(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    cur_19 = self.L__mod___serial_blocks3_0_mlp_drop2(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks3_0_drop_path_1 = self.L__mod___serial_blocks3_0_drop_path(cur_19);  cur_19 = None
    x3_2 = x_93 + l__mod___serial_blocks3_0_drop_path_1;  x_93 = l__mod___serial_blocks3_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    cls_token_5 = x3_2[(slice(None, None, None), slice(None, 1, None))]
    img_tokens_5 = x3_2[(slice(None, None, None), slice(1, None, None))];  x3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    transpose_33 = img_tokens_5.transpose(1, 2);  img_tokens_5 = None
    feat_5 = transpose_33.view(8, 320, 14, 14);  transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    l__mod___serial_blocks3_0_cpe_proj_1 = self.L__mod___serial_blocks3_0_cpe_proj(feat_5)
    x_102 = l__mod___serial_blocks3_0_cpe_proj_1 + feat_5;  l__mod___serial_blocks3_0_cpe_proj_1 = feat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    flatten_8 = x_102.flatten(2);  x_102 = None
    x_103 = flatten_8.transpose(1, 2);  flatten_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    x_105 = torch.cat((cls_token_5, x_103), dim = 1);  cls_token_5 = x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks3_1_norm1_weight = self.L__mod___serial_blocks3_1_norm1_weight
    l__mod___serial_blocks3_1_norm1_bias = self.L__mod___serial_blocks3_1_norm1_bias
    cur_20 = torch.nn.functional.layer_norm(x_105, (320,), l__mod___serial_blocks3_1_norm1_weight, l__mod___serial_blocks3_1_norm1_bias, 1e-06);  l__mod___serial_blocks3_1_norm1_weight = l__mod___serial_blocks3_1_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___serial_blocks3_1_factoratt_crpe_qkv = self.L__mod___serial_blocks3_1_factoratt_crpe_qkv(cur_20);  cur_20 = None
    reshape_22 = l__mod___serial_blocks3_1_factoratt_crpe_qkv.reshape(8, 197, 3, 8, 40);  l__mod___serial_blocks3_1_factoratt_crpe_qkv = None
    qkv_5 = reshape_22.permute(2, 0, 3, 1, 4);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_5 = unbind_5[0]
    k_5 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    k_softmax_5 = k_5.softmax(dim = 2);  k_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    transpose_35 = k_softmax_5.transpose(-1, -2);  k_softmax_5 = None
    factor_att_10 = transpose_35 @ v_5;  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    factor_att_11 = q_5 @ factor_att_10;  factor_att_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    q_img_5 = q_5[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  q_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    v_img_10 = v_5[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    transpose_36 = v_img_10.transpose(-1, -2);  v_img_10 = None
    v_img_11 = transpose_36.reshape(8, 320, 14, 14);  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_5 = torch.functional.split(v_img_11, [80, 120, 120], dim = 1);  v_img_11 = None
    getitem_59 = split_5[0]
    getitem_60 = split_5[1]
    getitem_61 = split_5[2];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_3 = self.L__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0(getitem_59);  getitem_59 = None
    l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_4 = self.L__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1(getitem_60);  getitem_60 = None
    l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_5 = self.L__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2(getitem_61);  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    conv_v_img_10 = torch.cat([l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_3, l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_4, l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_5], dim = 1);  l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_3 = l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_4 = l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    reshape_24 = conv_v_img_10.reshape(8, 8, 40, 196);  conv_v_img_10 = None
    conv_v_img_11 = reshape_24.transpose(-1, -2);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    EV_hat_10 = q_img_5 * conv_v_img_11;  q_img_5 = conv_v_img_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    crpe_5 = torch.nn.functional.pad(EV_hat_10, (0, 0, 1, 0, 0, 0));  EV_hat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_11 = 0.15811388300841897 * factor_att_11;  factor_att_11 = None
    x_107 = mul_11 + crpe_5;  mul_11 = crpe_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    transpose_38 = x_107.transpose(1, 2);  x_107 = None
    x_108 = transpose_38.reshape(8, 197, 320);  transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    x_109 = self.L__mod___serial_blocks3_1_factoratt_crpe_proj(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    cur_21 = self.L__mod___serial_blocks3_1_factoratt_crpe_proj_drop(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks3_1_drop_path = self.L__mod___serial_blocks3_1_drop_path(cur_21);  cur_21 = None
    x_111 = x_105 + l__mod___serial_blocks3_1_drop_path;  x_105 = l__mod___serial_blocks3_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks3_1_norm2_weight = self.L__mod___serial_blocks3_1_norm2_weight
    l__mod___serial_blocks3_1_norm2_bias = self.L__mod___serial_blocks3_1_norm2_bias
    cur_22 = torch.nn.functional.layer_norm(x_111, (320,), l__mod___serial_blocks3_1_norm2_weight, l__mod___serial_blocks3_1_norm2_bias, 1e-06);  l__mod___serial_blocks3_1_norm2_weight = l__mod___serial_blocks3_1_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_113 = self.L__mod___serial_blocks3_1_mlp_fc1(cur_22);  cur_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_114 = self.L__mod___serial_blocks3_1_mlp_act(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_115 = self.L__mod___serial_blocks3_1_mlp_drop1(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_116 = self.L__mod___serial_blocks3_1_mlp_norm(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_117 = self.L__mod___serial_blocks3_1_mlp_fc2(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    cur_23 = self.L__mod___serial_blocks3_1_mlp_drop2(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks3_1_drop_path_1 = self.L__mod___serial_blocks3_1_drop_path(cur_23);  cur_23 = None
    x3_3 = x_111 + l__mod___serial_blocks3_1_drop_path_1;  x_111 = l__mod___serial_blocks3_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    getitem_62 = x3_3[(slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x3_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:595, code: x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
    reshape_26 = getitem_62.reshape(8, 14, 14, -1);  getitem_62 = None
    permute_8 = reshape_26.permute(0, 3, 1, 2);  reshape_26 = None
    x3_nocls = permute_8.contiguous();  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x_120 = self.L__mod___patch_embed4_proj(x3_nocls);  x3_nocls = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten_9 = x_120.flatten(2);  x_120 = None
    x_121 = flatten_9.transpose(1, 2);  flatten_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x4 = self.L__mod___patch_embed4_norm(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:600, code: x4 = insert_cls(x4, self.cls_token4)
    l__mod___cls_token4 = self.L__mod___cls_token4
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    cls_tokens_3 = l__mod___cls_token4.expand(8, -1, -1);  l__mod___cls_token4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    x4_1 = torch.cat((cls_tokens_3, x4), dim = 1);  cls_tokens_3 = x4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    cls_token_6 = x4_1[(slice(None, None, None), slice(None, 1, None))]
    img_tokens_6 = x4_1[(slice(None, None, None), slice(1, None, None))];  x4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    transpose_40 = img_tokens_6.transpose(1, 2);  img_tokens_6 = None
    feat_6 = transpose_40.view(8, 512, 7, 7);  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    l__mod___serial_blocks4_0_cpe_proj = self.L__mod___serial_blocks4_0_cpe_proj(feat_6)
    x_124 = l__mod___serial_blocks4_0_cpe_proj + feat_6;  l__mod___serial_blocks4_0_cpe_proj = feat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    flatten_10 = x_124.flatten(2);  x_124 = None
    x_125 = flatten_10.transpose(1, 2);  flatten_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    x_127 = torch.cat((cls_token_6, x_125), dim = 1);  cls_token_6 = x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks4_0_norm1_weight = self.L__mod___serial_blocks4_0_norm1_weight
    l__mod___serial_blocks4_0_norm1_bias = self.L__mod___serial_blocks4_0_norm1_bias
    cur_24 = torch.nn.functional.layer_norm(x_127, (512,), l__mod___serial_blocks4_0_norm1_weight, l__mod___serial_blocks4_0_norm1_bias, 1e-06);  l__mod___serial_blocks4_0_norm1_weight = l__mod___serial_blocks4_0_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___serial_blocks4_0_factoratt_crpe_qkv = self.L__mod___serial_blocks4_0_factoratt_crpe_qkv(cur_24);  cur_24 = None
    reshape_27 = l__mod___serial_blocks4_0_factoratt_crpe_qkv.reshape(8, 50, 3, 8, 64);  l__mod___serial_blocks4_0_factoratt_crpe_qkv = None
    qkv_6 = reshape_27.permute(2, 0, 3, 1, 4);  reshape_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_6 = unbind_6[0]
    k_6 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    k_softmax_6 = k_6.softmax(dim = 2);  k_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    transpose_42 = k_softmax_6.transpose(-1, -2);  k_softmax_6 = None
    factor_att_12 = transpose_42 @ v_6;  transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    factor_att_13 = q_6 @ factor_att_12;  factor_att_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    q_img_6 = q_6[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  q_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    v_img_12 = v_6[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    transpose_43 = v_img_12.transpose(-1, -2);  v_img_12 = None
    v_img_13 = transpose_43.reshape(8, 512, 7, 7);  transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_6 = torch.functional.split(v_img_13, [128, 192, 192], dim = 1);  v_img_13 = None
    getitem_70 = split_6[0]
    getitem_71 = split_6[1]
    getitem_72 = split_6[2];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0 = self.L__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0(getitem_70);  getitem_70 = None
    l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1 = self.L__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1(getitem_71);  getitem_71 = None
    l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2 = self.L__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2(getitem_72);  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    conv_v_img_12 = torch.cat([l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0, l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1, l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2], dim = 1);  l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0 = l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1 = l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    reshape_29 = conv_v_img_12.reshape(8, 8, 64, 49);  conv_v_img_12 = None
    conv_v_img_13 = reshape_29.transpose(-1, -2);  reshape_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    EV_hat_12 = q_img_6 * conv_v_img_13;  q_img_6 = conv_v_img_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    crpe_6 = torch.nn.functional.pad(EV_hat_12, (0, 0, 1, 0, 0, 0));  EV_hat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_13 = 0.125 * factor_att_13;  factor_att_13 = None
    x_129 = mul_13 + crpe_6;  mul_13 = crpe_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    transpose_45 = x_129.transpose(1, 2);  x_129 = None
    x_130 = transpose_45.reshape(8, 50, 512);  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    x_131 = self.L__mod___serial_blocks4_0_factoratt_crpe_proj(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    cur_25 = self.L__mod___serial_blocks4_0_factoratt_crpe_proj_drop(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks4_0_drop_path = self.L__mod___serial_blocks4_0_drop_path(cur_25);  cur_25 = None
    x_133 = x_127 + l__mod___serial_blocks4_0_drop_path;  x_127 = l__mod___serial_blocks4_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks4_0_norm2_weight = self.L__mod___serial_blocks4_0_norm2_weight
    l__mod___serial_blocks4_0_norm2_bias = self.L__mod___serial_blocks4_0_norm2_bias
    cur_26 = torch.nn.functional.layer_norm(x_133, (512,), l__mod___serial_blocks4_0_norm2_weight, l__mod___serial_blocks4_0_norm2_bias, 1e-06);  l__mod___serial_blocks4_0_norm2_weight = l__mod___serial_blocks4_0_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_135 = self.L__mod___serial_blocks4_0_mlp_fc1(cur_26);  cur_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_136 = self.L__mod___serial_blocks4_0_mlp_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_137 = self.L__mod___serial_blocks4_0_mlp_drop1(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_138 = self.L__mod___serial_blocks4_0_mlp_norm(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_139 = self.L__mod___serial_blocks4_0_mlp_fc2(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    cur_27 = self.L__mod___serial_blocks4_0_mlp_drop2(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks4_0_drop_path_1 = self.L__mod___serial_blocks4_0_drop_path(cur_27);  cur_27 = None
    x4_2 = x_133 + l__mod___serial_blocks4_0_drop_path_1;  x_133 = l__mod___serial_blocks4_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    cls_token_7 = x4_2[(slice(None, None, None), slice(None, 1, None))]
    img_tokens_7 = x4_2[(slice(None, None, None), slice(1, None, None))];  x4_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    transpose_46 = img_tokens_7.transpose(1, 2);  img_tokens_7 = None
    feat_7 = transpose_46.view(8, 512, 7, 7);  transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    l__mod___serial_blocks4_0_cpe_proj_1 = self.L__mod___serial_blocks4_0_cpe_proj(feat_7)
    x_142 = l__mod___serial_blocks4_0_cpe_proj_1 + feat_7;  l__mod___serial_blocks4_0_cpe_proj_1 = feat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    flatten_11 = x_142.flatten(2);  x_142 = None
    x_143 = flatten_11.transpose(1, 2);  flatten_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    x_145 = torch.cat((cls_token_7, x_143), dim = 1);  cls_token_7 = x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks4_1_norm1_weight = self.L__mod___serial_blocks4_1_norm1_weight
    l__mod___serial_blocks4_1_norm1_bias = self.L__mod___serial_blocks4_1_norm1_bias
    cur_28 = torch.nn.functional.layer_norm(x_145, (512,), l__mod___serial_blocks4_1_norm1_weight, l__mod___serial_blocks4_1_norm1_bias, 1e-06);  l__mod___serial_blocks4_1_norm1_weight = l__mod___serial_blocks4_1_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___serial_blocks4_1_factoratt_crpe_qkv = self.L__mod___serial_blocks4_1_factoratt_crpe_qkv(cur_28);  cur_28 = None
    reshape_31 = l__mod___serial_blocks4_1_factoratt_crpe_qkv.reshape(8, 50, 3, 8, 64);  l__mod___serial_blocks4_1_factoratt_crpe_qkv = None
    qkv_7 = reshape_31.permute(2, 0, 3, 1, 4);  reshape_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_7 = unbind_7[0]
    k_7 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    k_softmax_7 = k_7.softmax(dim = 2);  k_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    transpose_48 = k_softmax_7.transpose(-1, -2);  k_softmax_7 = None
    factor_att_14 = transpose_48 @ v_7;  transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    factor_att_15 = q_7 @ factor_att_14;  factor_att_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    q_img_7 = q_7[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  q_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    v_img_14 = v_7[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    transpose_49 = v_img_14.transpose(-1, -2);  v_img_14 = None
    v_img_15 = transpose_49.reshape(8, 512, 7, 7);  transpose_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    split_7 = torch.functional.split(v_img_15, [128, 192, 192], dim = 1);  v_img_15 = None
    getitem_80 = split_7[0]
    getitem_81 = split_7[1]
    getitem_82 = split_7[2];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_3 = self.L__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0(getitem_80);  getitem_80 = None
    l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_4 = self.L__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1(getitem_81);  getitem_81 = None
    l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_5 = self.L__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2(getitem_82);  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    conv_v_img_14 = torch.cat([l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_3, l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_4, l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_5], dim = 1);  l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_3 = l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_4 = l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    reshape_33 = conv_v_img_14.reshape(8, 8, 64, 49);  conv_v_img_14 = None
    conv_v_img_15 = reshape_33.transpose(-1, -2);  reshape_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    EV_hat_14 = q_img_7 * conv_v_img_15;  q_img_7 = conv_v_img_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    crpe_7 = torch.nn.functional.pad(EV_hat_14, (0, 0, 1, 0, 0, 0));  EV_hat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_15 = 0.125 * factor_att_15;  factor_att_15 = None
    x_147 = mul_15 + crpe_7;  mul_15 = crpe_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    transpose_51 = x_147.transpose(1, 2);  x_147 = None
    x_148 = transpose_51.reshape(8, 50, 512);  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    x_149 = self.L__mod___serial_blocks4_1_factoratt_crpe_proj(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:136, code: x = self.proj_drop(x)
    cur_29 = self.L__mod___serial_blocks4_1_factoratt_crpe_proj_drop(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:216, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks4_1_drop_path = self.L__mod___serial_blocks4_1_drop_path(cur_29);  cur_29 = None
    x_151 = x_145 + l__mod___serial_blocks4_1_drop_path;  x_145 = l__mod___serial_blocks4_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___serial_blocks4_1_norm2_weight = self.L__mod___serial_blocks4_1_norm2_weight
    l__mod___serial_blocks4_1_norm2_bias = self.L__mod___serial_blocks4_1_norm2_bias
    cur_30 = torch.nn.functional.layer_norm(x_151, (512,), l__mod___serial_blocks4_1_norm2_weight, l__mod___serial_blocks4_1_norm2_bias, 1e-06);  l__mod___serial_blocks4_1_norm2_weight = l__mod___serial_blocks4_1_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_153 = self.L__mod___serial_blocks4_1_mlp_fc1(cur_30);  cur_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_154 = self.L__mod___serial_blocks4_1_mlp_act(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_155 = self.L__mod___serial_blocks4_1_mlp_drop1(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_156 = self.L__mod___serial_blocks4_1_mlp_norm(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_157 = self.L__mod___serial_blocks4_1_mlp_fc2(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    cur_31 = self.L__mod___serial_blocks4_1_mlp_drop2(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:221, code: x = x + self.drop_path(cur)
    l__mod___serial_blocks4_1_drop_path_1 = self.L__mod___serial_blocks4_1_drop_path(cur_31);  cur_31 = None
    x4_3 = x_151 + l__mod___serial_blocks4_1_drop_path_1;  x_151 = l__mod___serial_blocks4_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    getitem_83 = x4_3[(slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:603, code: x4_nocls = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
    reshape_35 = getitem_83.reshape(8, 7, 7, -1);  getitem_83 = None
    permute_11 = reshape_35.permute(0, 3, 1, 2);  reshape_35 = None
    x4_nocls = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___norm4_weight = self.L__mod___norm4_weight
    l__mod___norm4_bias = self.L__mod___norm4_bias
    x_feat = torch.nn.functional.layer_norm(x4_3, (512,), l__mod___norm4_weight, l__mod___norm4_bias, 1e-06);  x4_3 = l__mod___norm4_weight = l__mod___norm4_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:660, code: x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
    x_161 = x_feat[(slice(None, None, None), 0)];  x_feat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:661, code: x = self.head_drop(x)
    x_162 = self.L__mod___head_drop(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:662, code: return x if pre_logits else self.head(x)
    x_163 = self.L__mod___head(x_162);  x_162 = None
    return (x_163,)
    