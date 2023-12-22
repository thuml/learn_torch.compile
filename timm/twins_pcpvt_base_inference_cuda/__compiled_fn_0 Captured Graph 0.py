from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    l__mod___patch_embeds_0_proj = self.L__mod___patch_embeds_0_proj(l_inputs_0_);  l_inputs_0_ = None
    flatten = l__mod___patch_embeds_0_proj.flatten(2);  l__mod___patch_embeds_0_proj = None
    x = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    x_2 = self.L__mod___patch_embeds_0_norm(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    x_3 = self.L__mod___pos_drops_0(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_0_0_norm1 = self.L__mod___blocks_0_0_norm1(x_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_0_0_attn_q = self.L__mod___blocks_0_0_attn_q(l__mod___blocks_0_0_norm1)
    reshape = l__mod___blocks_0_0_attn_q.reshape(8, 3136, 1, 64);  l__mod___blocks_0_0_attn_q = None
    q = reshape.permute(0, 2, 1, 3);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_1 = l__mod___blocks_0_0_norm1.permute(0, 2, 1);  l__mod___blocks_0_0_norm1 = None
    x_4 = permute_1.reshape(8, 64, 56, 56);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_0_0_attn_sr = self.L__mod___blocks_0_0_attn_sr(x_4);  x_4 = None
    reshape_2 = l__mod___blocks_0_0_attn_sr.reshape(8, 64, -1);  l__mod___blocks_0_0_attn_sr = None
    x_5 = reshape_2.permute(0, 2, 1);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_6 = self.L__mod___blocks_0_0_attn_norm(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_0_0_attn_kv = self.L__mod___blocks_0_0_attn_kv(x_6);  x_6 = None
    reshape_3 = l__mod___blocks_0_0_attn_kv.reshape(8, -1, 2, 1, 64);  l__mod___blocks_0_0_attn_kv = None
    kv = reshape_3.permute(2, 0, 3, 1, 4);  reshape_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind = kv.unbind(0);  kv = None
    k = unbind[0]
    v = unbind[1];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_7 = torch._C._nn.scaled_dot_product_attention(q, k, v, dropout_p = 0.0);  q = k = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_1 = x_7.transpose(1, 2);  x_7 = None
    x_8 = transpose_1.reshape(8, 3136, 64);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_9 = self.L__mod___blocks_0_0_attn_proj(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_10 = self.L__mod___blocks_0_0_attn_proj_drop(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_0_0_drop_path1 = self.L__mod___blocks_0_0_drop_path1(x_10);  x_10 = None
    x_11 = x_3 + l__mod___blocks_0_0_drop_path1;  x_3 = l__mod___blocks_0_0_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_0_0_norm2 = self.L__mod___blocks_0_0_norm2(x_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_12 = self.L__mod___blocks_0_0_mlp_fc1(l__mod___blocks_0_0_norm2);  l__mod___blocks_0_0_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_13 = self.L__mod___blocks_0_0_mlp_act(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_14 = self.L__mod___blocks_0_0_mlp_drop1(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_15 = self.L__mod___blocks_0_0_mlp_norm(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_16 = self.L__mod___blocks_0_0_mlp_fc2(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_17 = self.L__mod___blocks_0_0_mlp_drop2(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_0_0_drop_path2 = self.L__mod___blocks_0_0_drop_path2(x_17);  x_17 = None
    x_19 = x_11 + l__mod___blocks_0_0_drop_path2;  x_11 = l__mod___blocks_0_0_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    transpose_2 = x_19.transpose(1, 2);  x_19 = None
    cnn_feat_token = transpose_2.view(8, 64, 56, 56);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    x_20 = self.L__mod___pos_block_0_proj_0(cnn_feat_token)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    x_20 += cnn_feat_token;  x_21 = x_20;  x_20 = cnn_feat_token = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:246, code: x = x.flatten(2).transpose(1, 2)
    flatten_1 = x_21.flatten(2);  x_21 = None
    x_23 = flatten_1.transpose(1, 2);  flatten_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_0_1_norm1 = self.L__mod___blocks_0_1_norm1(x_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_0_1_attn_q = self.L__mod___blocks_0_1_attn_q(l__mod___blocks_0_1_norm1)
    reshape_5 = l__mod___blocks_0_1_attn_q.reshape(8, 3136, 1, 64);  l__mod___blocks_0_1_attn_q = None
    q_1 = reshape_5.permute(0, 2, 1, 3);  reshape_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_5 = l__mod___blocks_0_1_norm1.permute(0, 2, 1);  l__mod___blocks_0_1_norm1 = None
    x_24 = permute_5.reshape(8, 64, 56, 56);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_0_1_attn_sr = self.L__mod___blocks_0_1_attn_sr(x_24);  x_24 = None
    reshape_7 = l__mod___blocks_0_1_attn_sr.reshape(8, 64, -1);  l__mod___blocks_0_1_attn_sr = None
    x_25 = reshape_7.permute(0, 2, 1);  reshape_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_26 = self.L__mod___blocks_0_1_attn_norm(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_0_1_attn_kv = self.L__mod___blocks_0_1_attn_kv(x_26);  x_26 = None
    reshape_8 = l__mod___blocks_0_1_attn_kv.reshape(8, -1, 2, 1, 64);  l__mod___blocks_0_1_attn_kv = None
    kv_1 = reshape_8.permute(2, 0, 3, 1, 4);  reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_1 = kv_1.unbind(0);  kv_1 = None
    k_1 = unbind_1[0]
    v_1 = unbind_1[1];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_27 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v_1, dropout_p = 0.0);  q_1 = k_1 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_4 = x_27.transpose(1, 2);  x_27 = None
    x_28 = transpose_4.reshape(8, 3136, 64);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_29 = self.L__mod___blocks_0_1_attn_proj(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_30 = self.L__mod___blocks_0_1_attn_proj_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_0_1_drop_path1 = self.L__mod___blocks_0_1_drop_path1(x_30);  x_30 = None
    x_31 = x_23 + l__mod___blocks_0_1_drop_path1;  x_23 = l__mod___blocks_0_1_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_0_1_norm2 = self.L__mod___blocks_0_1_norm2(x_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_32 = self.L__mod___blocks_0_1_mlp_fc1(l__mod___blocks_0_1_norm2);  l__mod___blocks_0_1_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_33 = self.L__mod___blocks_0_1_mlp_act(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_34 = self.L__mod___blocks_0_1_mlp_drop1(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_35 = self.L__mod___blocks_0_1_mlp_norm(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_36 = self.L__mod___blocks_0_1_mlp_fc2(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_37 = self.L__mod___blocks_0_1_mlp_drop2(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_0_1_drop_path2 = self.L__mod___blocks_0_1_drop_path2(x_37);  x_37 = None
    x_39 = x_31 + l__mod___blocks_0_1_drop_path2;  x_31 = l__mod___blocks_0_1_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_0_2_norm1 = self.L__mod___blocks_0_2_norm1(x_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_0_2_attn_q = self.L__mod___blocks_0_2_attn_q(l__mod___blocks_0_2_norm1)
    reshape_10 = l__mod___blocks_0_2_attn_q.reshape(8, 3136, 1, 64);  l__mod___blocks_0_2_attn_q = None
    q_2 = reshape_10.permute(0, 2, 1, 3);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_9 = l__mod___blocks_0_2_norm1.permute(0, 2, 1);  l__mod___blocks_0_2_norm1 = None
    x_40 = permute_9.reshape(8, 64, 56, 56);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_0_2_attn_sr = self.L__mod___blocks_0_2_attn_sr(x_40);  x_40 = None
    reshape_12 = l__mod___blocks_0_2_attn_sr.reshape(8, 64, -1);  l__mod___blocks_0_2_attn_sr = None
    x_41 = reshape_12.permute(0, 2, 1);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_42 = self.L__mod___blocks_0_2_attn_norm(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_0_2_attn_kv = self.L__mod___blocks_0_2_attn_kv(x_42);  x_42 = None
    reshape_13 = l__mod___blocks_0_2_attn_kv.reshape(8, -1, 2, 1, 64);  l__mod___blocks_0_2_attn_kv = None
    kv_2 = reshape_13.permute(2, 0, 3, 1, 4);  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_2 = kv_2.unbind(0);  kv_2 = None
    k_2 = unbind_2[0]
    v_2 = unbind_2[1];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_43 = torch._C._nn.scaled_dot_product_attention(q_2, k_2, v_2, dropout_p = 0.0);  q_2 = k_2 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_5 = x_43.transpose(1, 2);  x_43 = None
    x_44 = transpose_5.reshape(8, 3136, 64);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_45 = self.L__mod___blocks_0_2_attn_proj(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_46 = self.L__mod___blocks_0_2_attn_proj_drop(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_0_2_drop_path1 = self.L__mod___blocks_0_2_drop_path1(x_46);  x_46 = None
    x_47 = x_39 + l__mod___blocks_0_2_drop_path1;  x_39 = l__mod___blocks_0_2_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_0_2_norm2 = self.L__mod___blocks_0_2_norm2(x_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_48 = self.L__mod___blocks_0_2_mlp_fc1(l__mod___blocks_0_2_norm2);  l__mod___blocks_0_2_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_49 = self.L__mod___blocks_0_2_mlp_act(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_50 = self.L__mod___blocks_0_2_mlp_drop1(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_51 = self.L__mod___blocks_0_2_mlp_norm(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_52 = self.L__mod___blocks_0_2_mlp_fc2(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_53 = self.L__mod___blocks_0_2_mlp_drop2(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_0_2_drop_path2 = self.L__mod___blocks_0_2_drop_path2(x_53);  x_53 = None
    x_55 = x_47 + l__mod___blocks_0_2_drop_path2;  x_47 = l__mod___blocks_0_2_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    reshape_15 = x_55.reshape(8, 56, 56, -1);  x_55 = None
    permute_12 = reshape_15.permute(0, 3, 1, 2);  reshape_15 = None
    x_56 = permute_12.contiguous();  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    l__mod___patch_embeds_1_proj = self.L__mod___patch_embeds_1_proj(x_56);  x_56 = None
    flatten_2 = l__mod___patch_embeds_1_proj.flatten(2);  l__mod___patch_embeds_1_proj = None
    x_57 = flatten_2.transpose(1, 2);  flatten_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    x_59 = self.L__mod___patch_embeds_1_norm(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    x_60 = self.L__mod___pos_drops_1(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_1_0_norm1 = self.L__mod___blocks_1_0_norm1(x_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_0_attn_q = self.L__mod___blocks_1_0_attn_q(l__mod___blocks_1_0_norm1)
    reshape_16 = l__mod___blocks_1_0_attn_q.reshape(8, 784, 2, 64);  l__mod___blocks_1_0_attn_q = None
    q_3 = reshape_16.permute(0, 2, 1, 3);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_14 = l__mod___blocks_1_0_norm1.permute(0, 2, 1);  l__mod___blocks_1_0_norm1 = None
    x_61 = permute_14.reshape(8, 128, 28, 28);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_1_0_attn_sr = self.L__mod___blocks_1_0_attn_sr(x_61);  x_61 = None
    reshape_18 = l__mod___blocks_1_0_attn_sr.reshape(8, 128, -1);  l__mod___blocks_1_0_attn_sr = None
    x_62 = reshape_18.permute(0, 2, 1);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_63 = self.L__mod___blocks_1_0_attn_norm(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_1_0_attn_kv = self.L__mod___blocks_1_0_attn_kv(x_63);  x_63 = None
    reshape_19 = l__mod___blocks_1_0_attn_kv.reshape(8, -1, 2, 2, 64);  l__mod___blocks_1_0_attn_kv = None
    kv_3 = reshape_19.permute(2, 0, 3, 1, 4);  reshape_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_3 = kv_3.unbind(0);  kv_3 = None
    k_3 = unbind_3[0]
    v_3 = unbind_3[1];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_64 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_3, dropout_p = 0.0);  q_3 = k_3 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_7 = x_64.transpose(1, 2);  x_64 = None
    x_65 = transpose_7.reshape(8, 784, 128);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_66 = self.L__mod___blocks_1_0_attn_proj(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_67 = self.L__mod___blocks_1_0_attn_proj_drop(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_1_0_drop_path1 = self.L__mod___blocks_1_0_drop_path1(x_67);  x_67 = None
    x_68 = x_60 + l__mod___blocks_1_0_drop_path1;  x_60 = l__mod___blocks_1_0_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_1_0_norm2 = self.L__mod___blocks_1_0_norm2(x_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_69 = self.L__mod___blocks_1_0_mlp_fc1(l__mod___blocks_1_0_norm2);  l__mod___blocks_1_0_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_70 = self.L__mod___blocks_1_0_mlp_act(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_71 = self.L__mod___blocks_1_0_mlp_drop1(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_72 = self.L__mod___blocks_1_0_mlp_norm(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_73 = self.L__mod___blocks_1_0_mlp_fc2(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_74 = self.L__mod___blocks_1_0_mlp_drop2(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_1_0_drop_path2 = self.L__mod___blocks_1_0_drop_path2(x_74);  x_74 = None
    x_76 = x_68 + l__mod___blocks_1_0_drop_path2;  x_68 = l__mod___blocks_1_0_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    transpose_8 = x_76.transpose(1, 2);  x_76 = None
    cnn_feat_token_1 = transpose_8.view(8, 128, 28, 28);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    x_77 = self.L__mod___pos_block_1_proj_0(cnn_feat_token_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    x_77 += cnn_feat_token_1;  x_78 = x_77;  x_77 = cnn_feat_token_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:246, code: x = x.flatten(2).transpose(1, 2)
    flatten_3 = x_78.flatten(2);  x_78 = None
    x_80 = flatten_3.transpose(1, 2);  flatten_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_1_1_norm1 = self.L__mod___blocks_1_1_norm1(x_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_1_attn_q = self.L__mod___blocks_1_1_attn_q(l__mod___blocks_1_1_norm1)
    reshape_21 = l__mod___blocks_1_1_attn_q.reshape(8, 784, 2, 64);  l__mod___blocks_1_1_attn_q = None
    q_4 = reshape_21.permute(0, 2, 1, 3);  reshape_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_18 = l__mod___blocks_1_1_norm1.permute(0, 2, 1);  l__mod___blocks_1_1_norm1 = None
    x_81 = permute_18.reshape(8, 128, 28, 28);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_1_1_attn_sr = self.L__mod___blocks_1_1_attn_sr(x_81);  x_81 = None
    reshape_23 = l__mod___blocks_1_1_attn_sr.reshape(8, 128, -1);  l__mod___blocks_1_1_attn_sr = None
    x_82 = reshape_23.permute(0, 2, 1);  reshape_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_83 = self.L__mod___blocks_1_1_attn_norm(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_1_1_attn_kv = self.L__mod___blocks_1_1_attn_kv(x_83);  x_83 = None
    reshape_24 = l__mod___blocks_1_1_attn_kv.reshape(8, -1, 2, 2, 64);  l__mod___blocks_1_1_attn_kv = None
    kv_4 = reshape_24.permute(2, 0, 3, 1, 4);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_4 = kv_4.unbind(0);  kv_4 = None
    k_4 = unbind_4[0]
    v_4 = unbind_4[1];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_84 = torch._C._nn.scaled_dot_product_attention(q_4, k_4, v_4, dropout_p = 0.0);  q_4 = k_4 = v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_10 = x_84.transpose(1, 2);  x_84 = None
    x_85 = transpose_10.reshape(8, 784, 128);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_86 = self.L__mod___blocks_1_1_attn_proj(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_87 = self.L__mod___blocks_1_1_attn_proj_drop(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_1_1_drop_path1 = self.L__mod___blocks_1_1_drop_path1(x_87);  x_87 = None
    x_88 = x_80 + l__mod___blocks_1_1_drop_path1;  x_80 = l__mod___blocks_1_1_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_1_1_norm2 = self.L__mod___blocks_1_1_norm2(x_88)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_89 = self.L__mod___blocks_1_1_mlp_fc1(l__mod___blocks_1_1_norm2);  l__mod___blocks_1_1_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_90 = self.L__mod___blocks_1_1_mlp_act(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_91 = self.L__mod___blocks_1_1_mlp_drop1(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_92 = self.L__mod___blocks_1_1_mlp_norm(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_93 = self.L__mod___blocks_1_1_mlp_fc2(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_94 = self.L__mod___blocks_1_1_mlp_drop2(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_1_1_drop_path2 = self.L__mod___blocks_1_1_drop_path2(x_94);  x_94 = None
    x_96 = x_88 + l__mod___blocks_1_1_drop_path2;  x_88 = l__mod___blocks_1_1_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_1_2_norm1 = self.L__mod___blocks_1_2_norm1(x_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_2_attn_q = self.L__mod___blocks_1_2_attn_q(l__mod___blocks_1_2_norm1)
    reshape_26 = l__mod___blocks_1_2_attn_q.reshape(8, 784, 2, 64);  l__mod___blocks_1_2_attn_q = None
    q_5 = reshape_26.permute(0, 2, 1, 3);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_22 = l__mod___blocks_1_2_norm1.permute(0, 2, 1);  l__mod___blocks_1_2_norm1 = None
    x_97 = permute_22.reshape(8, 128, 28, 28);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_1_2_attn_sr = self.L__mod___blocks_1_2_attn_sr(x_97);  x_97 = None
    reshape_28 = l__mod___blocks_1_2_attn_sr.reshape(8, 128, -1);  l__mod___blocks_1_2_attn_sr = None
    x_98 = reshape_28.permute(0, 2, 1);  reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_99 = self.L__mod___blocks_1_2_attn_norm(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_1_2_attn_kv = self.L__mod___blocks_1_2_attn_kv(x_99);  x_99 = None
    reshape_29 = l__mod___blocks_1_2_attn_kv.reshape(8, -1, 2, 2, 64);  l__mod___blocks_1_2_attn_kv = None
    kv_5 = reshape_29.permute(2, 0, 3, 1, 4);  reshape_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_5 = kv_5.unbind(0);  kv_5 = None
    k_5 = unbind_5[0]
    v_5 = unbind_5[1];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_100 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_5, dropout_p = 0.0);  q_5 = k_5 = v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_11 = x_100.transpose(1, 2);  x_100 = None
    x_101 = transpose_11.reshape(8, 784, 128);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_102 = self.L__mod___blocks_1_2_attn_proj(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_103 = self.L__mod___blocks_1_2_attn_proj_drop(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_1_2_drop_path1 = self.L__mod___blocks_1_2_drop_path1(x_103);  x_103 = None
    x_104 = x_96 + l__mod___blocks_1_2_drop_path1;  x_96 = l__mod___blocks_1_2_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_1_2_norm2 = self.L__mod___blocks_1_2_norm2(x_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_105 = self.L__mod___blocks_1_2_mlp_fc1(l__mod___blocks_1_2_norm2);  l__mod___blocks_1_2_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_106 = self.L__mod___blocks_1_2_mlp_act(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_107 = self.L__mod___blocks_1_2_mlp_drop1(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_108 = self.L__mod___blocks_1_2_mlp_norm(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_109 = self.L__mod___blocks_1_2_mlp_fc2(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_110 = self.L__mod___blocks_1_2_mlp_drop2(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_1_2_drop_path2 = self.L__mod___blocks_1_2_drop_path2(x_110);  x_110 = None
    x_112 = x_104 + l__mod___blocks_1_2_drop_path2;  x_104 = l__mod___blocks_1_2_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_1_3_norm1 = self.L__mod___blocks_1_3_norm1(x_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_3_attn_q = self.L__mod___blocks_1_3_attn_q(l__mod___blocks_1_3_norm1)
    reshape_31 = l__mod___blocks_1_3_attn_q.reshape(8, 784, 2, 64);  l__mod___blocks_1_3_attn_q = None
    q_6 = reshape_31.permute(0, 2, 1, 3);  reshape_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_26 = l__mod___blocks_1_3_norm1.permute(0, 2, 1);  l__mod___blocks_1_3_norm1 = None
    x_113 = permute_26.reshape(8, 128, 28, 28);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_1_3_attn_sr = self.L__mod___blocks_1_3_attn_sr(x_113);  x_113 = None
    reshape_33 = l__mod___blocks_1_3_attn_sr.reshape(8, 128, -1);  l__mod___blocks_1_3_attn_sr = None
    x_114 = reshape_33.permute(0, 2, 1);  reshape_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_115 = self.L__mod___blocks_1_3_attn_norm(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_1_3_attn_kv = self.L__mod___blocks_1_3_attn_kv(x_115);  x_115 = None
    reshape_34 = l__mod___blocks_1_3_attn_kv.reshape(8, -1, 2, 2, 64);  l__mod___blocks_1_3_attn_kv = None
    kv_6 = reshape_34.permute(2, 0, 3, 1, 4);  reshape_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_6 = kv_6.unbind(0);  kv_6 = None
    k_6 = unbind_6[0]
    v_6 = unbind_6[1];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_116 = torch._C._nn.scaled_dot_product_attention(q_6, k_6, v_6, dropout_p = 0.0);  q_6 = k_6 = v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_12 = x_116.transpose(1, 2);  x_116 = None
    x_117 = transpose_12.reshape(8, 784, 128);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_118 = self.L__mod___blocks_1_3_attn_proj(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_119 = self.L__mod___blocks_1_3_attn_proj_drop(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_1_3_drop_path1 = self.L__mod___blocks_1_3_drop_path1(x_119);  x_119 = None
    x_120 = x_112 + l__mod___blocks_1_3_drop_path1;  x_112 = l__mod___blocks_1_3_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_1_3_norm2 = self.L__mod___blocks_1_3_norm2(x_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_121 = self.L__mod___blocks_1_3_mlp_fc1(l__mod___blocks_1_3_norm2);  l__mod___blocks_1_3_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_122 = self.L__mod___blocks_1_3_mlp_act(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_123 = self.L__mod___blocks_1_3_mlp_drop1(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_124 = self.L__mod___blocks_1_3_mlp_norm(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_125 = self.L__mod___blocks_1_3_mlp_fc2(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_126 = self.L__mod___blocks_1_3_mlp_drop2(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_1_3_drop_path2 = self.L__mod___blocks_1_3_drop_path2(x_126);  x_126 = None
    x_128 = x_120 + l__mod___blocks_1_3_drop_path2;  x_120 = l__mod___blocks_1_3_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    reshape_36 = x_128.reshape(8, 28, 28, -1);  x_128 = None
    permute_29 = reshape_36.permute(0, 3, 1, 2);  reshape_36 = None
    x_129 = permute_29.contiguous();  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    l__mod___patch_embeds_2_proj = self.L__mod___patch_embeds_2_proj(x_129);  x_129 = None
    flatten_4 = l__mod___patch_embeds_2_proj.flatten(2);  l__mod___patch_embeds_2_proj = None
    x_130 = flatten_4.transpose(1, 2);  flatten_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    x_132 = self.L__mod___patch_embeds_2_norm(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    x_133 = self.L__mod___pos_drops_2(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_0_norm1 = self.L__mod___blocks_2_0_norm1(x_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_0_attn_q = self.L__mod___blocks_2_0_attn_q(l__mod___blocks_2_0_norm1)
    reshape_37 = l__mod___blocks_2_0_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_0_attn_q = None
    q_7 = reshape_37.permute(0, 2, 1, 3);  reshape_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_31 = l__mod___blocks_2_0_norm1.permute(0, 2, 1);  l__mod___blocks_2_0_norm1 = None
    x_134 = permute_31.reshape(8, 320, 14, 14);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_0_attn_sr = self.L__mod___blocks_2_0_attn_sr(x_134);  x_134 = None
    reshape_39 = l__mod___blocks_2_0_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_0_attn_sr = None
    x_135 = reshape_39.permute(0, 2, 1);  reshape_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_136 = self.L__mod___blocks_2_0_attn_norm(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_0_attn_kv = self.L__mod___blocks_2_0_attn_kv(x_136);  x_136 = None
    reshape_40 = l__mod___blocks_2_0_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_0_attn_kv = None
    kv_7 = reshape_40.permute(2, 0, 3, 1, 4);  reshape_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_7 = kv_7.unbind(0);  kv_7 = None
    k_7 = unbind_7[0]
    v_7 = unbind_7[1];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_137 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_7, dropout_p = 0.0);  q_7 = k_7 = v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_14 = x_137.transpose(1, 2);  x_137 = None
    x_138 = transpose_14.reshape(8, 196, 320);  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_139 = self.L__mod___blocks_2_0_attn_proj(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_140 = self.L__mod___blocks_2_0_attn_proj_drop(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_0_drop_path1 = self.L__mod___blocks_2_0_drop_path1(x_140);  x_140 = None
    x_141 = x_133 + l__mod___blocks_2_0_drop_path1;  x_133 = l__mod___blocks_2_0_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_0_norm2 = self.L__mod___blocks_2_0_norm2(x_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_142 = self.L__mod___blocks_2_0_mlp_fc1(l__mod___blocks_2_0_norm2);  l__mod___blocks_2_0_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_143 = self.L__mod___blocks_2_0_mlp_act(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_144 = self.L__mod___blocks_2_0_mlp_drop1(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_145 = self.L__mod___blocks_2_0_mlp_norm(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_146 = self.L__mod___blocks_2_0_mlp_fc2(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_147 = self.L__mod___blocks_2_0_mlp_drop2(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_0_drop_path2 = self.L__mod___blocks_2_0_drop_path2(x_147);  x_147 = None
    x_149 = x_141 + l__mod___blocks_2_0_drop_path2;  x_141 = l__mod___blocks_2_0_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    transpose_15 = x_149.transpose(1, 2);  x_149 = None
    cnn_feat_token_2 = transpose_15.view(8, 320, 14, 14);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    x_150 = self.L__mod___pos_block_2_proj_0(cnn_feat_token_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    x_150 += cnn_feat_token_2;  x_151 = x_150;  x_150 = cnn_feat_token_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:246, code: x = x.flatten(2).transpose(1, 2)
    flatten_5 = x_151.flatten(2);  x_151 = None
    x_153 = flatten_5.transpose(1, 2);  flatten_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_1_norm1 = self.L__mod___blocks_2_1_norm1(x_153)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_1_attn_q = self.L__mod___blocks_2_1_attn_q(l__mod___blocks_2_1_norm1)
    reshape_42 = l__mod___blocks_2_1_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_1_attn_q = None
    q_8 = reshape_42.permute(0, 2, 1, 3);  reshape_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_35 = l__mod___blocks_2_1_norm1.permute(0, 2, 1);  l__mod___blocks_2_1_norm1 = None
    x_154 = permute_35.reshape(8, 320, 14, 14);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_1_attn_sr = self.L__mod___blocks_2_1_attn_sr(x_154);  x_154 = None
    reshape_44 = l__mod___blocks_2_1_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_1_attn_sr = None
    x_155 = reshape_44.permute(0, 2, 1);  reshape_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_156 = self.L__mod___blocks_2_1_attn_norm(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_1_attn_kv = self.L__mod___blocks_2_1_attn_kv(x_156);  x_156 = None
    reshape_45 = l__mod___blocks_2_1_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_1_attn_kv = None
    kv_8 = reshape_45.permute(2, 0, 3, 1, 4);  reshape_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_8 = kv_8.unbind(0);  kv_8 = None
    k_8 = unbind_8[0]
    v_8 = unbind_8[1];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_157 = torch._C._nn.scaled_dot_product_attention(q_8, k_8, v_8, dropout_p = 0.0);  q_8 = k_8 = v_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_17 = x_157.transpose(1, 2);  x_157 = None
    x_158 = transpose_17.reshape(8, 196, 320);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_159 = self.L__mod___blocks_2_1_attn_proj(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_160 = self.L__mod___blocks_2_1_attn_proj_drop(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_1_drop_path1 = self.L__mod___blocks_2_1_drop_path1(x_160);  x_160 = None
    x_161 = x_153 + l__mod___blocks_2_1_drop_path1;  x_153 = l__mod___blocks_2_1_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_1_norm2 = self.L__mod___blocks_2_1_norm2(x_161)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_162 = self.L__mod___blocks_2_1_mlp_fc1(l__mod___blocks_2_1_norm2);  l__mod___blocks_2_1_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_163 = self.L__mod___blocks_2_1_mlp_act(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_164 = self.L__mod___blocks_2_1_mlp_drop1(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_165 = self.L__mod___blocks_2_1_mlp_norm(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_166 = self.L__mod___blocks_2_1_mlp_fc2(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_167 = self.L__mod___blocks_2_1_mlp_drop2(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_1_drop_path2 = self.L__mod___blocks_2_1_drop_path2(x_167);  x_167 = None
    x_169 = x_161 + l__mod___blocks_2_1_drop_path2;  x_161 = l__mod___blocks_2_1_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_2_norm1 = self.L__mod___blocks_2_2_norm1(x_169)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_2_attn_q = self.L__mod___blocks_2_2_attn_q(l__mod___blocks_2_2_norm1)
    reshape_47 = l__mod___blocks_2_2_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_2_attn_q = None
    q_9 = reshape_47.permute(0, 2, 1, 3);  reshape_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_39 = l__mod___blocks_2_2_norm1.permute(0, 2, 1);  l__mod___blocks_2_2_norm1 = None
    x_170 = permute_39.reshape(8, 320, 14, 14);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_2_attn_sr = self.L__mod___blocks_2_2_attn_sr(x_170);  x_170 = None
    reshape_49 = l__mod___blocks_2_2_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_2_attn_sr = None
    x_171 = reshape_49.permute(0, 2, 1);  reshape_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_172 = self.L__mod___blocks_2_2_attn_norm(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_2_attn_kv = self.L__mod___blocks_2_2_attn_kv(x_172);  x_172 = None
    reshape_50 = l__mod___blocks_2_2_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_2_attn_kv = None
    kv_9 = reshape_50.permute(2, 0, 3, 1, 4);  reshape_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_9 = kv_9.unbind(0);  kv_9 = None
    k_9 = unbind_9[0]
    v_9 = unbind_9[1];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_173 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_9, dropout_p = 0.0);  q_9 = k_9 = v_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_18 = x_173.transpose(1, 2);  x_173 = None
    x_174 = transpose_18.reshape(8, 196, 320);  transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_175 = self.L__mod___blocks_2_2_attn_proj(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_176 = self.L__mod___blocks_2_2_attn_proj_drop(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_2_drop_path1 = self.L__mod___blocks_2_2_drop_path1(x_176);  x_176 = None
    x_177 = x_169 + l__mod___blocks_2_2_drop_path1;  x_169 = l__mod___blocks_2_2_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_2_norm2 = self.L__mod___blocks_2_2_norm2(x_177)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_178 = self.L__mod___blocks_2_2_mlp_fc1(l__mod___blocks_2_2_norm2);  l__mod___blocks_2_2_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_179 = self.L__mod___blocks_2_2_mlp_act(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_180 = self.L__mod___blocks_2_2_mlp_drop1(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_181 = self.L__mod___blocks_2_2_mlp_norm(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_182 = self.L__mod___blocks_2_2_mlp_fc2(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_183 = self.L__mod___blocks_2_2_mlp_drop2(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_2_drop_path2 = self.L__mod___blocks_2_2_drop_path2(x_183);  x_183 = None
    x_185 = x_177 + l__mod___blocks_2_2_drop_path2;  x_177 = l__mod___blocks_2_2_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_3_norm1 = self.L__mod___blocks_2_3_norm1(x_185)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_3_attn_q = self.L__mod___blocks_2_3_attn_q(l__mod___blocks_2_3_norm1)
    reshape_52 = l__mod___blocks_2_3_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_3_attn_q = None
    q_10 = reshape_52.permute(0, 2, 1, 3);  reshape_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_43 = l__mod___blocks_2_3_norm1.permute(0, 2, 1);  l__mod___blocks_2_3_norm1 = None
    x_186 = permute_43.reshape(8, 320, 14, 14);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_3_attn_sr = self.L__mod___blocks_2_3_attn_sr(x_186);  x_186 = None
    reshape_54 = l__mod___blocks_2_3_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_3_attn_sr = None
    x_187 = reshape_54.permute(0, 2, 1);  reshape_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_188 = self.L__mod___blocks_2_3_attn_norm(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_3_attn_kv = self.L__mod___blocks_2_3_attn_kv(x_188);  x_188 = None
    reshape_55 = l__mod___blocks_2_3_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_3_attn_kv = None
    kv_10 = reshape_55.permute(2, 0, 3, 1, 4);  reshape_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_10 = kv_10.unbind(0);  kv_10 = None
    k_10 = unbind_10[0]
    v_10 = unbind_10[1];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_189 = torch._C._nn.scaled_dot_product_attention(q_10, k_10, v_10, dropout_p = 0.0);  q_10 = k_10 = v_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_19 = x_189.transpose(1, 2);  x_189 = None
    x_190 = transpose_19.reshape(8, 196, 320);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_191 = self.L__mod___blocks_2_3_attn_proj(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_192 = self.L__mod___blocks_2_3_attn_proj_drop(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_3_drop_path1 = self.L__mod___blocks_2_3_drop_path1(x_192);  x_192 = None
    x_193 = x_185 + l__mod___blocks_2_3_drop_path1;  x_185 = l__mod___blocks_2_3_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_3_norm2 = self.L__mod___blocks_2_3_norm2(x_193)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_194 = self.L__mod___blocks_2_3_mlp_fc1(l__mod___blocks_2_3_norm2);  l__mod___blocks_2_3_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_195 = self.L__mod___blocks_2_3_mlp_act(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_196 = self.L__mod___blocks_2_3_mlp_drop1(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_197 = self.L__mod___blocks_2_3_mlp_norm(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_198 = self.L__mod___blocks_2_3_mlp_fc2(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_199 = self.L__mod___blocks_2_3_mlp_drop2(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_3_drop_path2 = self.L__mod___blocks_2_3_drop_path2(x_199);  x_199 = None
    x_201 = x_193 + l__mod___blocks_2_3_drop_path2;  x_193 = l__mod___blocks_2_3_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_4_norm1 = self.L__mod___blocks_2_4_norm1(x_201)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_4_attn_q = self.L__mod___blocks_2_4_attn_q(l__mod___blocks_2_4_norm1)
    reshape_57 = l__mod___blocks_2_4_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_4_attn_q = None
    q_11 = reshape_57.permute(0, 2, 1, 3);  reshape_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_47 = l__mod___blocks_2_4_norm1.permute(0, 2, 1);  l__mod___blocks_2_4_norm1 = None
    x_202 = permute_47.reshape(8, 320, 14, 14);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_4_attn_sr = self.L__mod___blocks_2_4_attn_sr(x_202);  x_202 = None
    reshape_59 = l__mod___blocks_2_4_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_4_attn_sr = None
    x_203 = reshape_59.permute(0, 2, 1);  reshape_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_204 = self.L__mod___blocks_2_4_attn_norm(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_4_attn_kv = self.L__mod___blocks_2_4_attn_kv(x_204);  x_204 = None
    reshape_60 = l__mod___blocks_2_4_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_4_attn_kv = None
    kv_11 = reshape_60.permute(2, 0, 3, 1, 4);  reshape_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_11 = kv_11.unbind(0);  kv_11 = None
    k_11 = unbind_11[0]
    v_11 = unbind_11[1];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_205 = torch._C._nn.scaled_dot_product_attention(q_11, k_11, v_11, dropout_p = 0.0);  q_11 = k_11 = v_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_20 = x_205.transpose(1, 2);  x_205 = None
    x_206 = transpose_20.reshape(8, 196, 320);  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_207 = self.L__mod___blocks_2_4_attn_proj(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_208 = self.L__mod___blocks_2_4_attn_proj_drop(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_4_drop_path1 = self.L__mod___blocks_2_4_drop_path1(x_208);  x_208 = None
    x_209 = x_201 + l__mod___blocks_2_4_drop_path1;  x_201 = l__mod___blocks_2_4_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_4_norm2 = self.L__mod___blocks_2_4_norm2(x_209)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_210 = self.L__mod___blocks_2_4_mlp_fc1(l__mod___blocks_2_4_norm2);  l__mod___blocks_2_4_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_211 = self.L__mod___blocks_2_4_mlp_act(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_212 = self.L__mod___blocks_2_4_mlp_drop1(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_213 = self.L__mod___blocks_2_4_mlp_norm(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_214 = self.L__mod___blocks_2_4_mlp_fc2(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_215 = self.L__mod___blocks_2_4_mlp_drop2(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_4_drop_path2 = self.L__mod___blocks_2_4_drop_path2(x_215);  x_215 = None
    x_217 = x_209 + l__mod___blocks_2_4_drop_path2;  x_209 = l__mod___blocks_2_4_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_5_norm1 = self.L__mod___blocks_2_5_norm1(x_217)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_5_attn_q = self.L__mod___blocks_2_5_attn_q(l__mod___blocks_2_5_norm1)
    reshape_62 = l__mod___blocks_2_5_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_5_attn_q = None
    q_12 = reshape_62.permute(0, 2, 1, 3);  reshape_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_51 = l__mod___blocks_2_5_norm1.permute(0, 2, 1);  l__mod___blocks_2_5_norm1 = None
    x_218 = permute_51.reshape(8, 320, 14, 14);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_5_attn_sr = self.L__mod___blocks_2_5_attn_sr(x_218);  x_218 = None
    reshape_64 = l__mod___blocks_2_5_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_5_attn_sr = None
    x_219 = reshape_64.permute(0, 2, 1);  reshape_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_220 = self.L__mod___blocks_2_5_attn_norm(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_5_attn_kv = self.L__mod___blocks_2_5_attn_kv(x_220);  x_220 = None
    reshape_65 = l__mod___blocks_2_5_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_5_attn_kv = None
    kv_12 = reshape_65.permute(2, 0, 3, 1, 4);  reshape_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_12 = kv_12.unbind(0);  kv_12 = None
    k_12 = unbind_12[0]
    v_12 = unbind_12[1];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_221 = torch._C._nn.scaled_dot_product_attention(q_12, k_12, v_12, dropout_p = 0.0);  q_12 = k_12 = v_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_21 = x_221.transpose(1, 2);  x_221 = None
    x_222 = transpose_21.reshape(8, 196, 320);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_223 = self.L__mod___blocks_2_5_attn_proj(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_224 = self.L__mod___blocks_2_5_attn_proj_drop(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_5_drop_path1 = self.L__mod___blocks_2_5_drop_path1(x_224);  x_224 = None
    x_225 = x_217 + l__mod___blocks_2_5_drop_path1;  x_217 = l__mod___blocks_2_5_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_5_norm2 = self.L__mod___blocks_2_5_norm2(x_225)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_226 = self.L__mod___blocks_2_5_mlp_fc1(l__mod___blocks_2_5_norm2);  l__mod___blocks_2_5_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_227 = self.L__mod___blocks_2_5_mlp_act(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_228 = self.L__mod___blocks_2_5_mlp_drop1(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_229 = self.L__mod___blocks_2_5_mlp_norm(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_230 = self.L__mod___blocks_2_5_mlp_fc2(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_231 = self.L__mod___blocks_2_5_mlp_drop2(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_5_drop_path2 = self.L__mod___blocks_2_5_drop_path2(x_231);  x_231 = None
    x_233 = x_225 + l__mod___blocks_2_5_drop_path2;  x_225 = l__mod___blocks_2_5_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_6_norm1 = self.L__mod___blocks_2_6_norm1(x_233)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_6_attn_q = self.L__mod___blocks_2_6_attn_q(l__mod___blocks_2_6_norm1)
    reshape_67 = l__mod___blocks_2_6_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_6_attn_q = None
    q_13 = reshape_67.permute(0, 2, 1, 3);  reshape_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_55 = l__mod___blocks_2_6_norm1.permute(0, 2, 1);  l__mod___blocks_2_6_norm1 = None
    x_234 = permute_55.reshape(8, 320, 14, 14);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_6_attn_sr = self.L__mod___blocks_2_6_attn_sr(x_234);  x_234 = None
    reshape_69 = l__mod___blocks_2_6_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_6_attn_sr = None
    x_235 = reshape_69.permute(0, 2, 1);  reshape_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_236 = self.L__mod___blocks_2_6_attn_norm(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_6_attn_kv = self.L__mod___blocks_2_6_attn_kv(x_236);  x_236 = None
    reshape_70 = l__mod___blocks_2_6_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_6_attn_kv = None
    kv_13 = reshape_70.permute(2, 0, 3, 1, 4);  reshape_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_13 = kv_13.unbind(0);  kv_13 = None
    k_13 = unbind_13[0]
    v_13 = unbind_13[1];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_237 = torch._C._nn.scaled_dot_product_attention(q_13, k_13, v_13, dropout_p = 0.0);  q_13 = k_13 = v_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_22 = x_237.transpose(1, 2);  x_237 = None
    x_238 = transpose_22.reshape(8, 196, 320);  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_239 = self.L__mod___blocks_2_6_attn_proj(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_240 = self.L__mod___blocks_2_6_attn_proj_drop(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_6_drop_path1 = self.L__mod___blocks_2_6_drop_path1(x_240);  x_240 = None
    x_241 = x_233 + l__mod___blocks_2_6_drop_path1;  x_233 = l__mod___blocks_2_6_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_6_norm2 = self.L__mod___blocks_2_6_norm2(x_241)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_242 = self.L__mod___blocks_2_6_mlp_fc1(l__mod___blocks_2_6_norm2);  l__mod___blocks_2_6_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_243 = self.L__mod___blocks_2_6_mlp_act(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_244 = self.L__mod___blocks_2_6_mlp_drop1(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_245 = self.L__mod___blocks_2_6_mlp_norm(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_246 = self.L__mod___blocks_2_6_mlp_fc2(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_247 = self.L__mod___blocks_2_6_mlp_drop2(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_6_drop_path2 = self.L__mod___blocks_2_6_drop_path2(x_247);  x_247 = None
    x_249 = x_241 + l__mod___blocks_2_6_drop_path2;  x_241 = l__mod___blocks_2_6_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_7_norm1 = self.L__mod___blocks_2_7_norm1(x_249)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_7_attn_q = self.L__mod___blocks_2_7_attn_q(l__mod___blocks_2_7_norm1)
    reshape_72 = l__mod___blocks_2_7_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_7_attn_q = None
    q_14 = reshape_72.permute(0, 2, 1, 3);  reshape_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_59 = l__mod___blocks_2_7_norm1.permute(0, 2, 1);  l__mod___blocks_2_7_norm1 = None
    x_250 = permute_59.reshape(8, 320, 14, 14);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_7_attn_sr = self.L__mod___blocks_2_7_attn_sr(x_250);  x_250 = None
    reshape_74 = l__mod___blocks_2_7_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_7_attn_sr = None
    x_251 = reshape_74.permute(0, 2, 1);  reshape_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_252 = self.L__mod___blocks_2_7_attn_norm(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_7_attn_kv = self.L__mod___blocks_2_7_attn_kv(x_252);  x_252 = None
    reshape_75 = l__mod___blocks_2_7_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_7_attn_kv = None
    kv_14 = reshape_75.permute(2, 0, 3, 1, 4);  reshape_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_14 = kv_14.unbind(0);  kv_14 = None
    k_14 = unbind_14[0]
    v_14 = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_253 = torch._C._nn.scaled_dot_product_attention(q_14, k_14, v_14, dropout_p = 0.0);  q_14 = k_14 = v_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_23 = x_253.transpose(1, 2);  x_253 = None
    x_254 = transpose_23.reshape(8, 196, 320);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_255 = self.L__mod___blocks_2_7_attn_proj(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_256 = self.L__mod___blocks_2_7_attn_proj_drop(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_7_drop_path1 = self.L__mod___blocks_2_7_drop_path1(x_256);  x_256 = None
    x_257 = x_249 + l__mod___blocks_2_7_drop_path1;  x_249 = l__mod___blocks_2_7_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_7_norm2 = self.L__mod___blocks_2_7_norm2(x_257)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_258 = self.L__mod___blocks_2_7_mlp_fc1(l__mod___blocks_2_7_norm2);  l__mod___blocks_2_7_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_259 = self.L__mod___blocks_2_7_mlp_act(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_260 = self.L__mod___blocks_2_7_mlp_drop1(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_261 = self.L__mod___blocks_2_7_mlp_norm(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_262 = self.L__mod___blocks_2_7_mlp_fc2(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_263 = self.L__mod___blocks_2_7_mlp_drop2(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_7_drop_path2 = self.L__mod___blocks_2_7_drop_path2(x_263);  x_263 = None
    x_265 = x_257 + l__mod___blocks_2_7_drop_path2;  x_257 = l__mod___blocks_2_7_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_8_norm1 = self.L__mod___blocks_2_8_norm1(x_265)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_8_attn_q = self.L__mod___blocks_2_8_attn_q(l__mod___blocks_2_8_norm1)
    reshape_77 = l__mod___blocks_2_8_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_8_attn_q = None
    q_15 = reshape_77.permute(0, 2, 1, 3);  reshape_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_63 = l__mod___blocks_2_8_norm1.permute(0, 2, 1);  l__mod___blocks_2_8_norm1 = None
    x_266 = permute_63.reshape(8, 320, 14, 14);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_8_attn_sr = self.L__mod___blocks_2_8_attn_sr(x_266);  x_266 = None
    reshape_79 = l__mod___blocks_2_8_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_8_attn_sr = None
    x_267 = reshape_79.permute(0, 2, 1);  reshape_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_268 = self.L__mod___blocks_2_8_attn_norm(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_8_attn_kv = self.L__mod___blocks_2_8_attn_kv(x_268);  x_268 = None
    reshape_80 = l__mod___blocks_2_8_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_8_attn_kv = None
    kv_15 = reshape_80.permute(2, 0, 3, 1, 4);  reshape_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_15 = kv_15.unbind(0);  kv_15 = None
    k_15 = unbind_15[0]
    v_15 = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_269 = torch._C._nn.scaled_dot_product_attention(q_15, k_15, v_15, dropout_p = 0.0);  q_15 = k_15 = v_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_24 = x_269.transpose(1, 2);  x_269 = None
    x_270 = transpose_24.reshape(8, 196, 320);  transpose_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_271 = self.L__mod___blocks_2_8_attn_proj(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_272 = self.L__mod___blocks_2_8_attn_proj_drop(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_8_drop_path1 = self.L__mod___blocks_2_8_drop_path1(x_272);  x_272 = None
    x_273 = x_265 + l__mod___blocks_2_8_drop_path1;  x_265 = l__mod___blocks_2_8_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_8_norm2 = self.L__mod___blocks_2_8_norm2(x_273)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_274 = self.L__mod___blocks_2_8_mlp_fc1(l__mod___blocks_2_8_norm2);  l__mod___blocks_2_8_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_275 = self.L__mod___blocks_2_8_mlp_act(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_276 = self.L__mod___blocks_2_8_mlp_drop1(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_277 = self.L__mod___blocks_2_8_mlp_norm(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_278 = self.L__mod___blocks_2_8_mlp_fc2(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_279 = self.L__mod___blocks_2_8_mlp_drop2(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_8_drop_path2 = self.L__mod___blocks_2_8_drop_path2(x_279);  x_279 = None
    x_281 = x_273 + l__mod___blocks_2_8_drop_path2;  x_273 = l__mod___blocks_2_8_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_9_norm1 = self.L__mod___blocks_2_9_norm1(x_281)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_9_attn_q = self.L__mod___blocks_2_9_attn_q(l__mod___blocks_2_9_norm1)
    reshape_82 = l__mod___blocks_2_9_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_9_attn_q = None
    q_16 = reshape_82.permute(0, 2, 1, 3);  reshape_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_67 = l__mod___blocks_2_9_norm1.permute(0, 2, 1);  l__mod___blocks_2_9_norm1 = None
    x_282 = permute_67.reshape(8, 320, 14, 14);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_9_attn_sr = self.L__mod___blocks_2_9_attn_sr(x_282);  x_282 = None
    reshape_84 = l__mod___blocks_2_9_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_9_attn_sr = None
    x_283 = reshape_84.permute(0, 2, 1);  reshape_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_284 = self.L__mod___blocks_2_9_attn_norm(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_9_attn_kv = self.L__mod___blocks_2_9_attn_kv(x_284);  x_284 = None
    reshape_85 = l__mod___blocks_2_9_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_9_attn_kv = None
    kv_16 = reshape_85.permute(2, 0, 3, 1, 4);  reshape_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_16 = kv_16.unbind(0);  kv_16 = None
    k_16 = unbind_16[0]
    v_16 = unbind_16[1];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_285 = torch._C._nn.scaled_dot_product_attention(q_16, k_16, v_16, dropout_p = 0.0);  q_16 = k_16 = v_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_25 = x_285.transpose(1, 2);  x_285 = None
    x_286 = transpose_25.reshape(8, 196, 320);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_287 = self.L__mod___blocks_2_9_attn_proj(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_288 = self.L__mod___blocks_2_9_attn_proj_drop(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_9_drop_path1 = self.L__mod___blocks_2_9_drop_path1(x_288);  x_288 = None
    x_289 = x_281 + l__mod___blocks_2_9_drop_path1;  x_281 = l__mod___blocks_2_9_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_9_norm2 = self.L__mod___blocks_2_9_norm2(x_289)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_290 = self.L__mod___blocks_2_9_mlp_fc1(l__mod___blocks_2_9_norm2);  l__mod___blocks_2_9_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_291 = self.L__mod___blocks_2_9_mlp_act(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_292 = self.L__mod___blocks_2_9_mlp_drop1(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_293 = self.L__mod___blocks_2_9_mlp_norm(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_294 = self.L__mod___blocks_2_9_mlp_fc2(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_295 = self.L__mod___blocks_2_9_mlp_drop2(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_9_drop_path2 = self.L__mod___blocks_2_9_drop_path2(x_295);  x_295 = None
    x_297 = x_289 + l__mod___blocks_2_9_drop_path2;  x_289 = l__mod___blocks_2_9_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_10_norm1 = self.L__mod___blocks_2_10_norm1(x_297)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_10_attn_q = self.L__mod___blocks_2_10_attn_q(l__mod___blocks_2_10_norm1)
    reshape_87 = l__mod___blocks_2_10_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_10_attn_q = None
    q_17 = reshape_87.permute(0, 2, 1, 3);  reshape_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_71 = l__mod___blocks_2_10_norm1.permute(0, 2, 1);  l__mod___blocks_2_10_norm1 = None
    x_298 = permute_71.reshape(8, 320, 14, 14);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_10_attn_sr = self.L__mod___blocks_2_10_attn_sr(x_298);  x_298 = None
    reshape_89 = l__mod___blocks_2_10_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_10_attn_sr = None
    x_299 = reshape_89.permute(0, 2, 1);  reshape_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_300 = self.L__mod___blocks_2_10_attn_norm(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_10_attn_kv = self.L__mod___blocks_2_10_attn_kv(x_300);  x_300 = None
    reshape_90 = l__mod___blocks_2_10_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_10_attn_kv = None
    kv_17 = reshape_90.permute(2, 0, 3, 1, 4);  reshape_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_17 = kv_17.unbind(0);  kv_17 = None
    k_17 = unbind_17[0]
    v_17 = unbind_17[1];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_301 = torch._C._nn.scaled_dot_product_attention(q_17, k_17, v_17, dropout_p = 0.0);  q_17 = k_17 = v_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_26 = x_301.transpose(1, 2);  x_301 = None
    x_302 = transpose_26.reshape(8, 196, 320);  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_303 = self.L__mod___blocks_2_10_attn_proj(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_304 = self.L__mod___blocks_2_10_attn_proj_drop(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_10_drop_path1 = self.L__mod___blocks_2_10_drop_path1(x_304);  x_304 = None
    x_305 = x_297 + l__mod___blocks_2_10_drop_path1;  x_297 = l__mod___blocks_2_10_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_10_norm2 = self.L__mod___blocks_2_10_norm2(x_305)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_306 = self.L__mod___blocks_2_10_mlp_fc1(l__mod___blocks_2_10_norm2);  l__mod___blocks_2_10_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_307 = self.L__mod___blocks_2_10_mlp_act(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_308 = self.L__mod___blocks_2_10_mlp_drop1(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_309 = self.L__mod___blocks_2_10_mlp_norm(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_310 = self.L__mod___blocks_2_10_mlp_fc2(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_311 = self.L__mod___blocks_2_10_mlp_drop2(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_10_drop_path2 = self.L__mod___blocks_2_10_drop_path2(x_311);  x_311 = None
    x_313 = x_305 + l__mod___blocks_2_10_drop_path2;  x_305 = l__mod___blocks_2_10_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_11_norm1 = self.L__mod___blocks_2_11_norm1(x_313)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_11_attn_q = self.L__mod___blocks_2_11_attn_q(l__mod___blocks_2_11_norm1)
    reshape_92 = l__mod___blocks_2_11_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_11_attn_q = None
    q_18 = reshape_92.permute(0, 2, 1, 3);  reshape_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_75 = l__mod___blocks_2_11_norm1.permute(0, 2, 1);  l__mod___blocks_2_11_norm1 = None
    x_314 = permute_75.reshape(8, 320, 14, 14);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_11_attn_sr = self.L__mod___blocks_2_11_attn_sr(x_314);  x_314 = None
    reshape_94 = l__mod___blocks_2_11_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_11_attn_sr = None
    x_315 = reshape_94.permute(0, 2, 1);  reshape_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_316 = self.L__mod___blocks_2_11_attn_norm(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_11_attn_kv = self.L__mod___blocks_2_11_attn_kv(x_316);  x_316 = None
    reshape_95 = l__mod___blocks_2_11_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_11_attn_kv = None
    kv_18 = reshape_95.permute(2, 0, 3, 1, 4);  reshape_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_18 = kv_18.unbind(0);  kv_18 = None
    k_18 = unbind_18[0]
    v_18 = unbind_18[1];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_317 = torch._C._nn.scaled_dot_product_attention(q_18, k_18, v_18, dropout_p = 0.0);  q_18 = k_18 = v_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_27 = x_317.transpose(1, 2);  x_317 = None
    x_318 = transpose_27.reshape(8, 196, 320);  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_319 = self.L__mod___blocks_2_11_attn_proj(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_320 = self.L__mod___blocks_2_11_attn_proj_drop(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_11_drop_path1 = self.L__mod___blocks_2_11_drop_path1(x_320);  x_320 = None
    x_321 = x_313 + l__mod___blocks_2_11_drop_path1;  x_313 = l__mod___blocks_2_11_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_11_norm2 = self.L__mod___blocks_2_11_norm2(x_321)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_322 = self.L__mod___blocks_2_11_mlp_fc1(l__mod___blocks_2_11_norm2);  l__mod___blocks_2_11_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_323 = self.L__mod___blocks_2_11_mlp_act(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_324 = self.L__mod___blocks_2_11_mlp_drop1(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_325 = self.L__mod___blocks_2_11_mlp_norm(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_326 = self.L__mod___blocks_2_11_mlp_fc2(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_327 = self.L__mod___blocks_2_11_mlp_drop2(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_11_drop_path2 = self.L__mod___blocks_2_11_drop_path2(x_327);  x_327 = None
    x_329 = x_321 + l__mod___blocks_2_11_drop_path2;  x_321 = l__mod___blocks_2_11_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_12_norm1 = self.L__mod___blocks_2_12_norm1(x_329)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_12_attn_q = self.L__mod___blocks_2_12_attn_q(l__mod___blocks_2_12_norm1)
    reshape_97 = l__mod___blocks_2_12_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_12_attn_q = None
    q_19 = reshape_97.permute(0, 2, 1, 3);  reshape_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_79 = l__mod___blocks_2_12_norm1.permute(0, 2, 1);  l__mod___blocks_2_12_norm1 = None
    x_330 = permute_79.reshape(8, 320, 14, 14);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_12_attn_sr = self.L__mod___blocks_2_12_attn_sr(x_330);  x_330 = None
    reshape_99 = l__mod___blocks_2_12_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_12_attn_sr = None
    x_331 = reshape_99.permute(0, 2, 1);  reshape_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_332 = self.L__mod___blocks_2_12_attn_norm(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_12_attn_kv = self.L__mod___blocks_2_12_attn_kv(x_332);  x_332 = None
    reshape_100 = l__mod___blocks_2_12_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_12_attn_kv = None
    kv_19 = reshape_100.permute(2, 0, 3, 1, 4);  reshape_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_19 = kv_19.unbind(0);  kv_19 = None
    k_19 = unbind_19[0]
    v_19 = unbind_19[1];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_333 = torch._C._nn.scaled_dot_product_attention(q_19, k_19, v_19, dropout_p = 0.0);  q_19 = k_19 = v_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_28 = x_333.transpose(1, 2);  x_333 = None
    x_334 = transpose_28.reshape(8, 196, 320);  transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_335 = self.L__mod___blocks_2_12_attn_proj(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_336 = self.L__mod___blocks_2_12_attn_proj_drop(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_12_drop_path1 = self.L__mod___blocks_2_12_drop_path1(x_336);  x_336 = None
    x_337 = x_329 + l__mod___blocks_2_12_drop_path1;  x_329 = l__mod___blocks_2_12_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_12_norm2 = self.L__mod___blocks_2_12_norm2(x_337)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_338 = self.L__mod___blocks_2_12_mlp_fc1(l__mod___blocks_2_12_norm2);  l__mod___blocks_2_12_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_339 = self.L__mod___blocks_2_12_mlp_act(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_340 = self.L__mod___blocks_2_12_mlp_drop1(x_339);  x_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_341 = self.L__mod___blocks_2_12_mlp_norm(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_342 = self.L__mod___blocks_2_12_mlp_fc2(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_343 = self.L__mod___blocks_2_12_mlp_drop2(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_12_drop_path2 = self.L__mod___blocks_2_12_drop_path2(x_343);  x_343 = None
    x_345 = x_337 + l__mod___blocks_2_12_drop_path2;  x_337 = l__mod___blocks_2_12_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_13_norm1 = self.L__mod___blocks_2_13_norm1(x_345)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_13_attn_q = self.L__mod___blocks_2_13_attn_q(l__mod___blocks_2_13_norm1)
    reshape_102 = l__mod___blocks_2_13_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_13_attn_q = None
    q_20 = reshape_102.permute(0, 2, 1, 3);  reshape_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_83 = l__mod___blocks_2_13_norm1.permute(0, 2, 1);  l__mod___blocks_2_13_norm1 = None
    x_346 = permute_83.reshape(8, 320, 14, 14);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_13_attn_sr = self.L__mod___blocks_2_13_attn_sr(x_346);  x_346 = None
    reshape_104 = l__mod___blocks_2_13_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_13_attn_sr = None
    x_347 = reshape_104.permute(0, 2, 1);  reshape_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_348 = self.L__mod___blocks_2_13_attn_norm(x_347);  x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_13_attn_kv = self.L__mod___blocks_2_13_attn_kv(x_348);  x_348 = None
    reshape_105 = l__mod___blocks_2_13_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_13_attn_kv = None
    kv_20 = reshape_105.permute(2, 0, 3, 1, 4);  reshape_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_20 = kv_20.unbind(0);  kv_20 = None
    k_20 = unbind_20[0]
    v_20 = unbind_20[1];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_349 = torch._C._nn.scaled_dot_product_attention(q_20, k_20, v_20, dropout_p = 0.0);  q_20 = k_20 = v_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_29 = x_349.transpose(1, 2);  x_349 = None
    x_350 = transpose_29.reshape(8, 196, 320);  transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_351 = self.L__mod___blocks_2_13_attn_proj(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_352 = self.L__mod___blocks_2_13_attn_proj_drop(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_13_drop_path1 = self.L__mod___blocks_2_13_drop_path1(x_352);  x_352 = None
    x_353 = x_345 + l__mod___blocks_2_13_drop_path1;  x_345 = l__mod___blocks_2_13_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_13_norm2 = self.L__mod___blocks_2_13_norm2(x_353)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_354 = self.L__mod___blocks_2_13_mlp_fc1(l__mod___blocks_2_13_norm2);  l__mod___blocks_2_13_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_355 = self.L__mod___blocks_2_13_mlp_act(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_356 = self.L__mod___blocks_2_13_mlp_drop1(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_357 = self.L__mod___blocks_2_13_mlp_norm(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_358 = self.L__mod___blocks_2_13_mlp_fc2(x_357);  x_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_359 = self.L__mod___blocks_2_13_mlp_drop2(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_13_drop_path2 = self.L__mod___blocks_2_13_drop_path2(x_359);  x_359 = None
    x_361 = x_353 + l__mod___blocks_2_13_drop_path2;  x_353 = l__mod___blocks_2_13_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_14_norm1 = self.L__mod___blocks_2_14_norm1(x_361)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_14_attn_q = self.L__mod___blocks_2_14_attn_q(l__mod___blocks_2_14_norm1)
    reshape_107 = l__mod___blocks_2_14_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_14_attn_q = None
    q_21 = reshape_107.permute(0, 2, 1, 3);  reshape_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_87 = l__mod___blocks_2_14_norm1.permute(0, 2, 1);  l__mod___blocks_2_14_norm1 = None
    x_362 = permute_87.reshape(8, 320, 14, 14);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_14_attn_sr = self.L__mod___blocks_2_14_attn_sr(x_362);  x_362 = None
    reshape_109 = l__mod___blocks_2_14_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_14_attn_sr = None
    x_363 = reshape_109.permute(0, 2, 1);  reshape_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_364 = self.L__mod___blocks_2_14_attn_norm(x_363);  x_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_14_attn_kv = self.L__mod___blocks_2_14_attn_kv(x_364);  x_364 = None
    reshape_110 = l__mod___blocks_2_14_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_14_attn_kv = None
    kv_21 = reshape_110.permute(2, 0, 3, 1, 4);  reshape_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_21 = kv_21.unbind(0);  kv_21 = None
    k_21 = unbind_21[0]
    v_21 = unbind_21[1];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_365 = torch._C._nn.scaled_dot_product_attention(q_21, k_21, v_21, dropout_p = 0.0);  q_21 = k_21 = v_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_30 = x_365.transpose(1, 2);  x_365 = None
    x_366 = transpose_30.reshape(8, 196, 320);  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_367 = self.L__mod___blocks_2_14_attn_proj(x_366);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_368 = self.L__mod___blocks_2_14_attn_proj_drop(x_367);  x_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_14_drop_path1 = self.L__mod___blocks_2_14_drop_path1(x_368);  x_368 = None
    x_369 = x_361 + l__mod___blocks_2_14_drop_path1;  x_361 = l__mod___blocks_2_14_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_14_norm2 = self.L__mod___blocks_2_14_norm2(x_369)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_370 = self.L__mod___blocks_2_14_mlp_fc1(l__mod___blocks_2_14_norm2);  l__mod___blocks_2_14_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_371 = self.L__mod___blocks_2_14_mlp_act(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_372 = self.L__mod___blocks_2_14_mlp_drop1(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_373 = self.L__mod___blocks_2_14_mlp_norm(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_374 = self.L__mod___blocks_2_14_mlp_fc2(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_375 = self.L__mod___blocks_2_14_mlp_drop2(x_374);  x_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_14_drop_path2 = self.L__mod___blocks_2_14_drop_path2(x_375);  x_375 = None
    x_377 = x_369 + l__mod___blocks_2_14_drop_path2;  x_369 = l__mod___blocks_2_14_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_15_norm1 = self.L__mod___blocks_2_15_norm1(x_377)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_15_attn_q = self.L__mod___blocks_2_15_attn_q(l__mod___blocks_2_15_norm1)
    reshape_112 = l__mod___blocks_2_15_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_15_attn_q = None
    q_22 = reshape_112.permute(0, 2, 1, 3);  reshape_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_91 = l__mod___blocks_2_15_norm1.permute(0, 2, 1);  l__mod___blocks_2_15_norm1 = None
    x_378 = permute_91.reshape(8, 320, 14, 14);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_15_attn_sr = self.L__mod___blocks_2_15_attn_sr(x_378);  x_378 = None
    reshape_114 = l__mod___blocks_2_15_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_15_attn_sr = None
    x_379 = reshape_114.permute(0, 2, 1);  reshape_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_380 = self.L__mod___blocks_2_15_attn_norm(x_379);  x_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_15_attn_kv = self.L__mod___blocks_2_15_attn_kv(x_380);  x_380 = None
    reshape_115 = l__mod___blocks_2_15_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_15_attn_kv = None
    kv_22 = reshape_115.permute(2, 0, 3, 1, 4);  reshape_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_22 = kv_22.unbind(0);  kv_22 = None
    k_22 = unbind_22[0]
    v_22 = unbind_22[1];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_381 = torch._C._nn.scaled_dot_product_attention(q_22, k_22, v_22, dropout_p = 0.0);  q_22 = k_22 = v_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_31 = x_381.transpose(1, 2);  x_381 = None
    x_382 = transpose_31.reshape(8, 196, 320);  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_383 = self.L__mod___blocks_2_15_attn_proj(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_384 = self.L__mod___blocks_2_15_attn_proj_drop(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_15_drop_path1 = self.L__mod___blocks_2_15_drop_path1(x_384);  x_384 = None
    x_385 = x_377 + l__mod___blocks_2_15_drop_path1;  x_377 = l__mod___blocks_2_15_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_15_norm2 = self.L__mod___blocks_2_15_norm2(x_385)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_386 = self.L__mod___blocks_2_15_mlp_fc1(l__mod___blocks_2_15_norm2);  l__mod___blocks_2_15_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_387 = self.L__mod___blocks_2_15_mlp_act(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_388 = self.L__mod___blocks_2_15_mlp_drop1(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_389 = self.L__mod___blocks_2_15_mlp_norm(x_388);  x_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_390 = self.L__mod___blocks_2_15_mlp_fc2(x_389);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_391 = self.L__mod___blocks_2_15_mlp_drop2(x_390);  x_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_15_drop_path2 = self.L__mod___blocks_2_15_drop_path2(x_391);  x_391 = None
    x_393 = x_385 + l__mod___blocks_2_15_drop_path2;  x_385 = l__mod___blocks_2_15_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_16_norm1 = self.L__mod___blocks_2_16_norm1(x_393)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_16_attn_q = self.L__mod___blocks_2_16_attn_q(l__mod___blocks_2_16_norm1)
    reshape_117 = l__mod___blocks_2_16_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_16_attn_q = None
    q_23 = reshape_117.permute(0, 2, 1, 3);  reshape_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_95 = l__mod___blocks_2_16_norm1.permute(0, 2, 1);  l__mod___blocks_2_16_norm1 = None
    x_394 = permute_95.reshape(8, 320, 14, 14);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_16_attn_sr = self.L__mod___blocks_2_16_attn_sr(x_394);  x_394 = None
    reshape_119 = l__mod___blocks_2_16_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_16_attn_sr = None
    x_395 = reshape_119.permute(0, 2, 1);  reshape_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_396 = self.L__mod___blocks_2_16_attn_norm(x_395);  x_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_16_attn_kv = self.L__mod___blocks_2_16_attn_kv(x_396);  x_396 = None
    reshape_120 = l__mod___blocks_2_16_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_16_attn_kv = None
    kv_23 = reshape_120.permute(2, 0, 3, 1, 4);  reshape_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_23 = kv_23.unbind(0);  kv_23 = None
    k_23 = unbind_23[0]
    v_23 = unbind_23[1];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_397 = torch._C._nn.scaled_dot_product_attention(q_23, k_23, v_23, dropout_p = 0.0);  q_23 = k_23 = v_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_32 = x_397.transpose(1, 2);  x_397 = None
    x_398 = transpose_32.reshape(8, 196, 320);  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_399 = self.L__mod___blocks_2_16_attn_proj(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_400 = self.L__mod___blocks_2_16_attn_proj_drop(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_16_drop_path1 = self.L__mod___blocks_2_16_drop_path1(x_400);  x_400 = None
    x_401 = x_393 + l__mod___blocks_2_16_drop_path1;  x_393 = l__mod___blocks_2_16_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_16_norm2 = self.L__mod___blocks_2_16_norm2(x_401)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_402 = self.L__mod___blocks_2_16_mlp_fc1(l__mod___blocks_2_16_norm2);  l__mod___blocks_2_16_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_403 = self.L__mod___blocks_2_16_mlp_act(x_402);  x_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_404 = self.L__mod___blocks_2_16_mlp_drop1(x_403);  x_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_405 = self.L__mod___blocks_2_16_mlp_norm(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_406 = self.L__mod___blocks_2_16_mlp_fc2(x_405);  x_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_407 = self.L__mod___blocks_2_16_mlp_drop2(x_406);  x_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_16_drop_path2 = self.L__mod___blocks_2_16_drop_path2(x_407);  x_407 = None
    x_409 = x_401 + l__mod___blocks_2_16_drop_path2;  x_401 = l__mod___blocks_2_16_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_17_norm1 = self.L__mod___blocks_2_17_norm1(x_409)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_17_attn_q = self.L__mod___blocks_2_17_attn_q(l__mod___blocks_2_17_norm1)
    reshape_122 = l__mod___blocks_2_17_attn_q.reshape(8, 196, 5, 64);  l__mod___blocks_2_17_attn_q = None
    q_24 = reshape_122.permute(0, 2, 1, 3);  reshape_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    permute_99 = l__mod___blocks_2_17_norm1.permute(0, 2, 1);  l__mod___blocks_2_17_norm1 = None
    x_410 = permute_99.reshape(8, 320, 14, 14);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    l__mod___blocks_2_17_attn_sr = self.L__mod___blocks_2_17_attn_sr(x_410);  x_410 = None
    reshape_124 = l__mod___blocks_2_17_attn_sr.reshape(8, 320, -1);  l__mod___blocks_2_17_attn_sr = None
    x_411 = reshape_124.permute(0, 2, 1);  reshape_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    x_412 = self.L__mod___blocks_2_17_attn_norm(x_411);  x_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_17_attn_kv = self.L__mod___blocks_2_17_attn_kv(x_412);  x_412 = None
    reshape_125 = l__mod___blocks_2_17_attn_kv.reshape(8, -1, 2, 5, 64);  l__mod___blocks_2_17_attn_kv = None
    kv_24 = reshape_125.permute(2, 0, 3, 1, 4);  reshape_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_24 = kv_24.unbind(0);  kv_24 = None
    k_24 = unbind_24[0]
    v_24 = unbind_24[1];  unbind_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_413 = torch._C._nn.scaled_dot_product_attention(q_24, k_24, v_24, dropout_p = 0.0);  q_24 = k_24 = v_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_33 = x_413.transpose(1, 2);  x_413 = None
    x_414 = transpose_33.reshape(8, 196, 320);  transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_415 = self.L__mod___blocks_2_17_attn_proj(x_414);  x_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_416 = self.L__mod___blocks_2_17_attn_proj_drop(x_415);  x_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_2_17_drop_path1 = self.L__mod___blocks_2_17_drop_path1(x_416);  x_416 = None
    x_417 = x_409 + l__mod___blocks_2_17_drop_path1;  x_409 = l__mod___blocks_2_17_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_17_norm2 = self.L__mod___blocks_2_17_norm2(x_417)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_418 = self.L__mod___blocks_2_17_mlp_fc1(l__mod___blocks_2_17_norm2);  l__mod___blocks_2_17_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_419 = self.L__mod___blocks_2_17_mlp_act(x_418);  x_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_420 = self.L__mod___blocks_2_17_mlp_drop1(x_419);  x_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_421 = self.L__mod___blocks_2_17_mlp_norm(x_420);  x_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_422 = self.L__mod___blocks_2_17_mlp_fc2(x_421);  x_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_423 = self.L__mod___blocks_2_17_mlp_drop2(x_422);  x_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_2_17_drop_path2 = self.L__mod___blocks_2_17_drop_path2(x_423);  x_423 = None
    x_425 = x_417 + l__mod___blocks_2_17_drop_path2;  x_417 = l__mod___blocks_2_17_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    reshape_127 = x_425.reshape(8, 14, 14, -1);  x_425 = None
    permute_102 = reshape_127.permute(0, 3, 1, 2);  reshape_127 = None
    x_426 = permute_102.contiguous();  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    l__mod___patch_embeds_3_proj = self.L__mod___patch_embeds_3_proj(x_426);  x_426 = None
    flatten_6 = l__mod___patch_embeds_3_proj.flatten(2);  l__mod___patch_embeds_3_proj = None
    x_427 = flatten_6.transpose(1, 2);  flatten_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    x_429 = self.L__mod___patch_embeds_3_norm(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:409, code: x = drop(x)
    x_430 = self.L__mod___pos_drops_3(x_429);  x_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_3_0_norm1 = self.L__mod___blocks_3_0_norm1(x_430)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_3_0_attn_q = self.L__mod___blocks_3_0_attn_q(l__mod___blocks_3_0_norm1)
    reshape_128 = l__mod___blocks_3_0_attn_q.reshape(8, 49, 8, 64);  l__mod___blocks_3_0_attn_q = None
    q_25 = reshape_128.permute(0, 2, 1, 3);  reshape_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_3_0_attn_kv = self.L__mod___blocks_3_0_attn_kv(l__mod___blocks_3_0_norm1);  l__mod___blocks_3_0_norm1 = None
    reshape_129 = l__mod___blocks_3_0_attn_kv.reshape(8, -1, 2, 8, 64);  l__mod___blocks_3_0_attn_kv = None
    kv_25 = reshape_129.permute(2, 0, 3, 1, 4);  reshape_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_25 = kv_25.unbind(0);  kv_25 = None
    k_25 = unbind_25[0]
    v_25 = unbind_25[1];  unbind_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_431 = torch._C._nn.scaled_dot_product_attention(q_25, k_25, v_25, dropout_p = 0.0);  q_25 = k_25 = v_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_35 = x_431.transpose(1, 2);  x_431 = None
    x_432 = transpose_35.reshape(8, 49, 512);  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_433 = self.L__mod___blocks_3_0_attn_proj(x_432);  x_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_434 = self.L__mod___blocks_3_0_attn_proj_drop(x_433);  x_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_3_0_drop_path1 = self.L__mod___blocks_3_0_drop_path1(x_434);  x_434 = None
    x_435 = x_430 + l__mod___blocks_3_0_drop_path1;  x_430 = l__mod___blocks_3_0_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_3_0_norm2 = self.L__mod___blocks_3_0_norm2(x_435)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_436 = self.L__mod___blocks_3_0_mlp_fc1(l__mod___blocks_3_0_norm2);  l__mod___blocks_3_0_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_437 = self.L__mod___blocks_3_0_mlp_act(x_436);  x_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_438 = self.L__mod___blocks_3_0_mlp_drop1(x_437);  x_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_439 = self.L__mod___blocks_3_0_mlp_norm(x_438);  x_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_440 = self.L__mod___blocks_3_0_mlp_fc2(x_439);  x_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_441 = self.L__mod___blocks_3_0_mlp_drop2(x_440);  x_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_3_0_drop_path2 = self.L__mod___blocks_3_0_drop_path2(x_441);  x_441 = None
    x_443 = x_435 + l__mod___blocks_3_0_drop_path2;  x_435 = l__mod___blocks_3_0_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    transpose_36 = x_443.transpose(1, 2);  x_443 = None
    cnn_feat_token_3 = transpose_36.view(8, 512, 7, 7);  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    x_444 = self.L__mod___pos_block_3_proj_0(cnn_feat_token_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:245, code: x += cnn_feat_token
    x_444 += cnn_feat_token_3;  x_445 = x_444;  x_444 = cnn_feat_token_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:246, code: x = x.flatten(2).transpose(1, 2)
    flatten_7 = x_445.flatten(2);  x_445 = None
    x_447 = flatten_7.transpose(1, 2);  flatten_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_3_1_norm1 = self.L__mod___blocks_3_1_norm1(x_447)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_3_1_attn_q = self.L__mod___blocks_3_1_attn_q(l__mod___blocks_3_1_norm1)
    reshape_131 = l__mod___blocks_3_1_attn_q.reshape(8, 49, 8, 64);  l__mod___blocks_3_1_attn_q = None
    q_26 = reshape_131.permute(0, 2, 1, 3);  reshape_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_3_1_attn_kv = self.L__mod___blocks_3_1_attn_kv(l__mod___blocks_3_1_norm1);  l__mod___blocks_3_1_norm1 = None
    reshape_132 = l__mod___blocks_3_1_attn_kv.reshape(8, -1, 2, 8, 64);  l__mod___blocks_3_1_attn_kv = None
    kv_26 = reshape_132.permute(2, 0, 3, 1, 4);  reshape_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_26 = kv_26.unbind(0);  kv_26 = None
    k_26 = unbind_26[0]
    v_26 = unbind_26[1];  unbind_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_448 = torch._C._nn.scaled_dot_product_attention(q_26, k_26, v_26, dropout_p = 0.0);  q_26 = k_26 = v_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_38 = x_448.transpose(1, 2);  x_448 = None
    x_449 = transpose_38.reshape(8, 49, 512);  transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_450 = self.L__mod___blocks_3_1_attn_proj(x_449);  x_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_451 = self.L__mod___blocks_3_1_attn_proj_drop(x_450);  x_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_3_1_drop_path1 = self.L__mod___blocks_3_1_drop_path1(x_451);  x_451 = None
    x_452 = x_447 + l__mod___blocks_3_1_drop_path1;  x_447 = l__mod___blocks_3_1_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_3_1_norm2 = self.L__mod___blocks_3_1_norm2(x_452)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_453 = self.L__mod___blocks_3_1_mlp_fc1(l__mod___blocks_3_1_norm2);  l__mod___blocks_3_1_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_454 = self.L__mod___blocks_3_1_mlp_act(x_453);  x_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_455 = self.L__mod___blocks_3_1_mlp_drop1(x_454);  x_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_456 = self.L__mod___blocks_3_1_mlp_norm(x_455);  x_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_457 = self.L__mod___blocks_3_1_mlp_fc2(x_456);  x_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_458 = self.L__mod___blocks_3_1_mlp_drop2(x_457);  x_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_3_1_drop_path2 = self.L__mod___blocks_3_1_drop_path2(x_458);  x_458 = None
    x_460 = x_452 + l__mod___blocks_3_1_drop_path2;  x_452 = l__mod___blocks_3_1_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_3_2_norm1 = self.L__mod___blocks_3_2_norm1(x_460)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_3_2_attn_q = self.L__mod___blocks_3_2_attn_q(l__mod___blocks_3_2_norm1)
    reshape_134 = l__mod___blocks_3_2_attn_q.reshape(8, 49, 8, 64);  l__mod___blocks_3_2_attn_q = None
    q_27 = reshape_134.permute(0, 2, 1, 3);  reshape_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    l__mod___blocks_3_2_attn_kv = self.L__mod___blocks_3_2_attn_kv(l__mod___blocks_3_2_norm1);  l__mod___blocks_3_2_norm1 = None
    reshape_135 = l__mod___blocks_3_2_attn_kv.reshape(8, -1, 2, 8, 64);  l__mod___blocks_3_2_attn_kv = None
    kv_27 = reshape_135.permute(2, 0, 3, 1, 4);  reshape_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    unbind_27 = kv_27.unbind(0);  kv_27 = None
    k_27 = unbind_27[0]
    v_27 = unbind_27[1];  unbind_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    x_461 = torch._C._nn.scaled_dot_product_attention(q_27, k_27, v_27, dropout_p = 0.0);  q_27 = k_27 = v_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_39 = x_461.transpose(1, 2);  x_461 = None
    x_462 = transpose_39.reshape(8, 49, 512);  transpose_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    x_463 = self.L__mod___blocks_3_2_attn_proj(x_462);  x_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:186, code: x = self.proj_drop(x)
    x_464 = self.L__mod___blocks_3_2_attn_proj_drop(x_463);  x_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    l__mod___blocks_3_2_drop_path1 = self.L__mod___blocks_3_2_drop_path1(x_464);  x_464 = None
    x_465 = x_460 + l__mod___blocks_3_2_drop_path1;  x_460 = l__mod___blocks_3_2_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_3_2_norm2 = self.L__mod___blocks_3_2_norm2(x_465)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_466 = self.L__mod___blocks_3_2_mlp_fc1(l__mod___blocks_3_2_norm2);  l__mod___blocks_3_2_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_467 = self.L__mod___blocks_3_2_mlp_act(x_466);  x_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_468 = self.L__mod___blocks_3_2_mlp_drop1(x_467);  x_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_469 = self.L__mod___blocks_3_2_mlp_norm(x_468);  x_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_470 = self.L__mod___blocks_3_2_mlp_fc2(x_469);  x_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_471 = self.L__mod___blocks_3_2_mlp_drop2(x_470);  x_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    l__mod___blocks_3_2_drop_path2 = self.L__mod___blocks_3_2_drop_path2(x_471);  x_471 = None
    x_473 = x_465 + l__mod___blocks_3_2_drop_path2;  x_465 = l__mod___blocks_3_2_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:416, code: x = self.norm(x)
    x_475 = self.L__mod___norm(x_473);  x_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:421, code: x = x.mean(dim=1)
    x_476 = x_475.mean(dim = 1);  x_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:422, code: x = self.head_drop(x)
    x_477 = self.L__mod___head_drop(x_476);  x_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:423, code: return x if pre_logits else self.head(x)
    x_478 = self.L__mod___head(x_477);  x_477 = None
    return (x_478,)
    