from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:309, code: pixel_embed = self.pixel_embed(x, self.pixel_pos)
    l__mod___pixel_pos = self.L__mod___pixel_pos
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:181, code: x = self.proj(x)
    x = self.L__mod___pixel_embed_proj(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:182, code: x = self.unfold(x)
    x_1 = self.L__mod___pixel_embed_unfold(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:183, code: x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
    transpose = x_1.transpose(1, 2);  x_1 = None
    x_2 = transpose.reshape(1568, 24, 4, 4);  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:184, code: x = x + pixel_pos
    x_3 = x_2 + l__mod___pixel_pos;  x_2 = l__mod___pixel_pos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:185, code: x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
    reshape_1 = x_3.reshape(1568, 24, -1);  x_3 = None
    pixel_embed = reshape_1.transpose(1, 2);  reshape_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:311, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    reshape_2 = pixel_embed.reshape(8, 196, -1)
    l__mod___norm1_proj = self.L__mod___norm1_proj(reshape_2);  reshape_2 = None
    l__mod___proj = self.L__mod___proj(l__mod___norm1_proj);  l__mod___norm1_proj = None
    patch_embed = self.L__mod___norm2_proj(l__mod___proj);  l__mod___proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:312, code: patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
    l__mod___cls_token = self.L__mod___cls_token
    expand = l__mod___cls_token.expand(8, -1, -1);  l__mod___cls_token = None
    patch_embed_1 = torch.cat((expand, patch_embed), dim = 1);  expand = patch_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:313, code: patch_embed = patch_embed + self.patch_pos
    l__mod___patch_pos = self.L__mod___patch_pos
    patch_embed_2 = patch_embed_1 + l__mod___patch_pos;  patch_embed_1 = l__mod___patch_pos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:314, code: patch_embed = self.pos_drop(patch_embed)
    patch_embed_3 = self.L__mod___pos_drop(patch_embed_2);  patch_embed_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_0_norm_in = self.L__mod___blocks_0_norm_in(pixel_embed)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_0_attn_in_qk = self.L__mod___blocks_0_attn_in_qk(l__mod___blocks_0_norm_in)
    reshape_3 = l__mod___blocks_0_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_0_attn_in_qk = None
    qk = reshape_3.permute(2, 0, 3, 1, 4);  reshape_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind = qk.unbind(0);  qk = None
    q = unbind[0]
    k = unbind[1];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_0_attn_in_v = self.L__mod___blocks_0_attn_in_v(l__mod___blocks_0_norm_in);  l__mod___blocks_0_norm_in = None
    reshape_4 = l__mod___blocks_0_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_0_attn_in_v = None
    v = reshape_4.permute(0, 2, 1, 3);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_2 = k.transpose(-2, -1);  k = None
    matmul = q @ transpose_2;  q = transpose_2 = None
    attn = matmul * 0.408248290463863;  matmul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_1 = attn.softmax(dim = -1);  attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_2 = self.L__mod___blocks_0_attn_in_attn_drop(attn_1);  attn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_1 = attn_2 @ v;  attn_2 = v = None
    transpose_3 = matmul_1.transpose(1, 2);  matmul_1 = None
    x_5 = transpose_3.reshape(1568, 16, -1);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_6 = self.L__mod___blocks_0_attn_in_proj(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_7 = self.L__mod___blocks_0_attn_in_proj_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_0_drop_path = self.L__mod___blocks_0_drop_path(x_7);  x_7 = None
    pixel_embed_1 = pixel_embed + l__mod___blocks_0_drop_path;  pixel_embed = l__mod___blocks_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_0_norm_mlp_in = self.L__mod___blocks_0_norm_mlp_in(pixel_embed_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_8 = self.L__mod___blocks_0_mlp_in_fc1(l__mod___blocks_0_norm_mlp_in);  l__mod___blocks_0_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_9 = self.L__mod___blocks_0_mlp_in_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_10 = self.L__mod___blocks_0_mlp_in_drop1(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_11 = self.L__mod___blocks_0_mlp_in_norm(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_12 = self.L__mod___blocks_0_mlp_in_fc2(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_13 = self.L__mod___blocks_0_mlp_in_drop2(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_0_drop_path_1 = self.L__mod___blocks_0_drop_path(x_13);  x_13 = None
    pixel_embed_3 = pixel_embed_1 + l__mod___blocks_0_drop_path_1;  pixel_embed_1 = l__mod___blocks_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_2 = patch_embed_3[(slice(None, None, None), slice(0, 1, None))]
    getitem_3 = patch_embed_3[(slice(None, None, None), slice(1, None, None))];  patch_embed_3 = None
    l__mod___blocks_0_norm1_proj = self.L__mod___blocks_0_norm1_proj(pixel_embed_3)
    reshape_6 = l__mod___blocks_0_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_0_norm1_proj = None
    l__mod___blocks_0_proj = self.L__mod___blocks_0_proj(reshape_6);  reshape_6 = None
    add_4 = getitem_3 + l__mod___blocks_0_proj;  getitem_3 = l__mod___blocks_0_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_4 = torch.cat([getitem_2, add_4], dim = 1);  getitem_2 = add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_0_norm_out = self.L__mod___blocks_0_norm_out(patch_embed_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_0_attn_out_qk = self.L__mod___blocks_0_attn_out_qk(l__mod___blocks_0_norm_out)
    reshape_7 = l__mod___blocks_0_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_0_attn_out_qk = None
    qk_1 = reshape_7.permute(2, 0, 3, 1, 4);  reshape_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = qk_1.unbind(0);  qk_1 = None
    q_1 = unbind_1[0]
    k_1 = unbind_1[1];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_0_attn_out_v = self.L__mod___blocks_0_attn_out_v(l__mod___blocks_0_norm_out);  l__mod___blocks_0_norm_out = None
    reshape_8 = l__mod___blocks_0_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_0_attn_out_v = None
    v_1 = reshape_8.permute(0, 2, 1, 3);  reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_4 = k_1.transpose(-2, -1);  k_1 = None
    matmul_2 = q_1 @ transpose_4;  q_1 = transpose_4 = None
    attn_3 = matmul_2 * 0.125;  matmul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_4 = attn_3.softmax(dim = -1);  attn_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_5 = self.L__mod___blocks_0_attn_out_attn_drop(attn_4);  attn_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_3 = attn_5 @ v_1;  attn_5 = v_1 = None
    transpose_5 = matmul_3.transpose(1, 2);  matmul_3 = None
    x_14 = transpose_5.reshape(8, 197, -1);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_15 = self.L__mod___blocks_0_attn_out_proj(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_16 = self.L__mod___blocks_0_attn_out_proj_drop(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_0_drop_path_2 = self.L__mod___blocks_0_drop_path(x_16);  x_16 = None
    patch_embed_5 = patch_embed_4 + l__mod___blocks_0_drop_path_2;  patch_embed_4 = l__mod___blocks_0_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_0_norm_mlp = self.L__mod___blocks_0_norm_mlp(patch_embed_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_17 = self.L__mod___blocks_0_mlp_fc1(l__mod___blocks_0_norm_mlp);  l__mod___blocks_0_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_18 = self.L__mod___blocks_0_mlp_act(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_19 = self.L__mod___blocks_0_mlp_drop1(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_20 = self.L__mod___blocks_0_mlp_norm(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_21 = self.L__mod___blocks_0_mlp_fc2(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_22 = self.L__mod___blocks_0_mlp_drop2(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_0_drop_path_3 = self.L__mod___blocks_0_drop_path(x_22);  x_22 = None
    patch_embed_7 = patch_embed_5 + l__mod___blocks_0_drop_path_3;  patch_embed_5 = l__mod___blocks_0_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_1_norm_in = self.L__mod___blocks_1_norm_in(pixel_embed_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_1_attn_in_qk = self.L__mod___blocks_1_attn_in_qk(l__mod___blocks_1_norm_in)
    reshape_10 = l__mod___blocks_1_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_1_attn_in_qk = None
    qk_2 = reshape_10.permute(2, 0, 3, 1, 4);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = qk_2.unbind(0);  qk_2 = None
    q_2 = unbind_2[0]
    k_2 = unbind_2[1];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_1_attn_in_v = self.L__mod___blocks_1_attn_in_v(l__mod___blocks_1_norm_in);  l__mod___blocks_1_norm_in = None
    reshape_11 = l__mod___blocks_1_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_1_attn_in_v = None
    v_2 = reshape_11.permute(0, 2, 1, 3);  reshape_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_6 = k_2.transpose(-2, -1);  k_2 = None
    matmul_4 = q_2 @ transpose_6;  q_2 = transpose_6 = None
    attn_6 = matmul_4 * 0.408248290463863;  matmul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_7 = attn_6.softmax(dim = -1);  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_8 = self.L__mod___blocks_1_attn_in_attn_drop(attn_7);  attn_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_5 = attn_8 @ v_2;  attn_8 = v_2 = None
    transpose_7 = matmul_5.transpose(1, 2);  matmul_5 = None
    x_23 = transpose_7.reshape(1568, 16, -1);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_24 = self.L__mod___blocks_1_attn_in_proj(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_25 = self.L__mod___blocks_1_attn_in_proj_drop(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_1_drop_path = self.L__mod___blocks_1_drop_path(x_25);  x_25 = None
    pixel_embed_4 = pixel_embed_3 + l__mod___blocks_1_drop_path;  pixel_embed_3 = l__mod___blocks_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_1_norm_mlp_in = self.L__mod___blocks_1_norm_mlp_in(pixel_embed_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_26 = self.L__mod___blocks_1_mlp_in_fc1(l__mod___blocks_1_norm_mlp_in);  l__mod___blocks_1_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_27 = self.L__mod___blocks_1_mlp_in_act(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_28 = self.L__mod___blocks_1_mlp_in_drop1(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_29 = self.L__mod___blocks_1_mlp_in_norm(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_30 = self.L__mod___blocks_1_mlp_in_fc2(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_31 = self.L__mod___blocks_1_mlp_in_drop2(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_1_drop_path_1 = self.L__mod___blocks_1_drop_path(x_31);  x_31 = None
    pixel_embed_6 = pixel_embed_4 + l__mod___blocks_1_drop_path_1;  pixel_embed_4 = l__mod___blocks_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_8 = patch_embed_7[(slice(None, None, None), slice(0, 1, None))]
    getitem_9 = patch_embed_7[(slice(None, None, None), slice(1, None, None))];  patch_embed_7 = None
    l__mod___blocks_1_norm1_proj = self.L__mod___blocks_1_norm1_proj(pixel_embed_6)
    reshape_13 = l__mod___blocks_1_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_1_norm1_proj = None
    l__mod___blocks_1_proj = self.L__mod___blocks_1_proj(reshape_13);  reshape_13 = None
    add_9 = getitem_9 + l__mod___blocks_1_proj;  getitem_9 = l__mod___blocks_1_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_8 = torch.cat([getitem_8, add_9], dim = 1);  getitem_8 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_1_norm_out = self.L__mod___blocks_1_norm_out(patch_embed_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_1_attn_out_qk = self.L__mod___blocks_1_attn_out_qk(l__mod___blocks_1_norm_out)
    reshape_14 = l__mod___blocks_1_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_1_attn_out_qk = None
    qk_3 = reshape_14.permute(2, 0, 3, 1, 4);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = qk_3.unbind(0);  qk_3 = None
    q_3 = unbind_3[0]
    k_3 = unbind_3[1];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_1_attn_out_v = self.L__mod___blocks_1_attn_out_v(l__mod___blocks_1_norm_out);  l__mod___blocks_1_norm_out = None
    reshape_15 = l__mod___blocks_1_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_1_attn_out_v = None
    v_3 = reshape_15.permute(0, 2, 1, 3);  reshape_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_8 = k_3.transpose(-2, -1);  k_3 = None
    matmul_6 = q_3 @ transpose_8;  q_3 = transpose_8 = None
    attn_9 = matmul_6 * 0.125;  matmul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_10 = attn_9.softmax(dim = -1);  attn_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_11 = self.L__mod___blocks_1_attn_out_attn_drop(attn_10);  attn_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_7 = attn_11 @ v_3;  attn_11 = v_3 = None
    transpose_9 = matmul_7.transpose(1, 2);  matmul_7 = None
    x_32 = transpose_9.reshape(8, 197, -1);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_33 = self.L__mod___blocks_1_attn_out_proj(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_34 = self.L__mod___blocks_1_attn_out_proj_drop(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_1_drop_path_2 = self.L__mod___blocks_1_drop_path(x_34);  x_34 = None
    patch_embed_9 = patch_embed_8 + l__mod___blocks_1_drop_path_2;  patch_embed_8 = l__mod___blocks_1_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_1_norm_mlp = self.L__mod___blocks_1_norm_mlp(patch_embed_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_35 = self.L__mod___blocks_1_mlp_fc1(l__mod___blocks_1_norm_mlp);  l__mod___blocks_1_norm_mlp = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_1_drop_path_3 = self.L__mod___blocks_1_drop_path(x_40);  x_40 = None
    patch_embed_11 = patch_embed_9 + l__mod___blocks_1_drop_path_3;  patch_embed_9 = l__mod___blocks_1_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_2_norm_in = self.L__mod___blocks_2_norm_in(pixel_embed_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_attn_in_qk = self.L__mod___blocks_2_attn_in_qk(l__mod___blocks_2_norm_in)
    reshape_17 = l__mod___blocks_2_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_2_attn_in_qk = None
    qk_4 = reshape_17.permute(2, 0, 3, 1, 4);  reshape_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = qk_4.unbind(0);  qk_4 = None
    q_4 = unbind_4[0]
    k_4 = unbind_4[1];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_2_attn_in_v = self.L__mod___blocks_2_attn_in_v(l__mod___blocks_2_norm_in);  l__mod___blocks_2_norm_in = None
    reshape_18 = l__mod___blocks_2_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_2_attn_in_v = None
    v_4 = reshape_18.permute(0, 2, 1, 3);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_10 = k_4.transpose(-2, -1);  k_4 = None
    matmul_8 = q_4 @ transpose_10;  q_4 = transpose_10 = None
    attn_12 = matmul_8 * 0.408248290463863;  matmul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_13 = attn_12.softmax(dim = -1);  attn_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_14 = self.L__mod___blocks_2_attn_in_attn_drop(attn_13);  attn_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_9 = attn_14 @ v_4;  attn_14 = v_4 = None
    transpose_11 = matmul_9.transpose(1, 2);  matmul_9 = None
    x_41 = transpose_11.reshape(1568, 16, -1);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_42 = self.L__mod___blocks_2_attn_in_proj(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_43 = self.L__mod___blocks_2_attn_in_proj_drop(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_2_drop_path = self.L__mod___blocks_2_drop_path(x_43);  x_43 = None
    pixel_embed_7 = pixel_embed_6 + l__mod___blocks_2_drop_path;  pixel_embed_6 = l__mod___blocks_2_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_2_norm_mlp_in = self.L__mod___blocks_2_norm_mlp_in(pixel_embed_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_44 = self.L__mod___blocks_2_mlp_in_fc1(l__mod___blocks_2_norm_mlp_in);  l__mod___blocks_2_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_45 = self.L__mod___blocks_2_mlp_in_act(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_46 = self.L__mod___blocks_2_mlp_in_drop1(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_47 = self.L__mod___blocks_2_mlp_in_norm(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_48 = self.L__mod___blocks_2_mlp_in_fc2(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_49 = self.L__mod___blocks_2_mlp_in_drop2(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_2_drop_path_1 = self.L__mod___blocks_2_drop_path(x_49);  x_49 = None
    pixel_embed_9 = pixel_embed_7 + l__mod___blocks_2_drop_path_1;  pixel_embed_7 = l__mod___blocks_2_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_14 = patch_embed_11[(slice(None, None, None), slice(0, 1, None))]
    getitem_15 = patch_embed_11[(slice(None, None, None), slice(1, None, None))];  patch_embed_11 = None
    l__mod___blocks_2_norm1_proj = self.L__mod___blocks_2_norm1_proj(pixel_embed_9)
    reshape_20 = l__mod___blocks_2_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_2_norm1_proj = None
    l__mod___blocks_2_proj = self.L__mod___blocks_2_proj(reshape_20);  reshape_20 = None
    add_14 = getitem_15 + l__mod___blocks_2_proj;  getitem_15 = l__mod___blocks_2_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_12 = torch.cat([getitem_14, add_14], dim = 1);  getitem_14 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_2_norm_out = self.L__mod___blocks_2_norm_out(patch_embed_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_2_attn_out_qk = self.L__mod___blocks_2_attn_out_qk(l__mod___blocks_2_norm_out)
    reshape_21 = l__mod___blocks_2_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_2_attn_out_qk = None
    qk_5 = reshape_21.permute(2, 0, 3, 1, 4);  reshape_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = qk_5.unbind(0);  qk_5 = None
    q_5 = unbind_5[0]
    k_5 = unbind_5[1];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_2_attn_out_v = self.L__mod___blocks_2_attn_out_v(l__mod___blocks_2_norm_out);  l__mod___blocks_2_norm_out = None
    reshape_22 = l__mod___blocks_2_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_2_attn_out_v = None
    v_5 = reshape_22.permute(0, 2, 1, 3);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_12 = k_5.transpose(-2, -1);  k_5 = None
    matmul_10 = q_5 @ transpose_12;  q_5 = transpose_12 = None
    attn_15 = matmul_10 * 0.125;  matmul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_16 = attn_15.softmax(dim = -1);  attn_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_17 = self.L__mod___blocks_2_attn_out_attn_drop(attn_16);  attn_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_11 = attn_17 @ v_5;  attn_17 = v_5 = None
    transpose_13 = matmul_11.transpose(1, 2);  matmul_11 = None
    x_50 = transpose_13.reshape(8, 197, -1);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_51 = self.L__mod___blocks_2_attn_out_proj(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_52 = self.L__mod___blocks_2_attn_out_proj_drop(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_2_drop_path_2 = self.L__mod___blocks_2_drop_path(x_52);  x_52 = None
    patch_embed_13 = patch_embed_12 + l__mod___blocks_2_drop_path_2;  patch_embed_12 = l__mod___blocks_2_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_2_norm_mlp = self.L__mod___blocks_2_norm_mlp(patch_embed_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_53 = self.L__mod___blocks_2_mlp_fc1(l__mod___blocks_2_norm_mlp);  l__mod___blocks_2_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_54 = self.L__mod___blocks_2_mlp_act(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_55 = self.L__mod___blocks_2_mlp_drop1(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_56 = self.L__mod___blocks_2_mlp_norm(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_57 = self.L__mod___blocks_2_mlp_fc2(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_58 = self.L__mod___blocks_2_mlp_drop2(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_2_drop_path_3 = self.L__mod___blocks_2_drop_path(x_58);  x_58 = None
    patch_embed_15 = patch_embed_13 + l__mod___blocks_2_drop_path_3;  patch_embed_13 = l__mod___blocks_2_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_3_norm_in = self.L__mod___blocks_3_norm_in(pixel_embed_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_3_attn_in_qk = self.L__mod___blocks_3_attn_in_qk(l__mod___blocks_3_norm_in)
    reshape_24 = l__mod___blocks_3_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_3_attn_in_qk = None
    qk_6 = reshape_24.permute(2, 0, 3, 1, 4);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = qk_6.unbind(0);  qk_6 = None
    q_6 = unbind_6[0]
    k_6 = unbind_6[1];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_3_attn_in_v = self.L__mod___blocks_3_attn_in_v(l__mod___blocks_3_norm_in);  l__mod___blocks_3_norm_in = None
    reshape_25 = l__mod___blocks_3_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_3_attn_in_v = None
    v_6 = reshape_25.permute(0, 2, 1, 3);  reshape_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_14 = k_6.transpose(-2, -1);  k_6 = None
    matmul_12 = q_6 @ transpose_14;  q_6 = transpose_14 = None
    attn_18 = matmul_12 * 0.408248290463863;  matmul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_19 = attn_18.softmax(dim = -1);  attn_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_20 = self.L__mod___blocks_3_attn_in_attn_drop(attn_19);  attn_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_13 = attn_20 @ v_6;  attn_20 = v_6 = None
    transpose_15 = matmul_13.transpose(1, 2);  matmul_13 = None
    x_59 = transpose_15.reshape(1568, 16, -1);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_60 = self.L__mod___blocks_3_attn_in_proj(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_61 = self.L__mod___blocks_3_attn_in_proj_drop(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_3_drop_path = self.L__mod___blocks_3_drop_path(x_61);  x_61 = None
    pixel_embed_10 = pixel_embed_9 + l__mod___blocks_3_drop_path;  pixel_embed_9 = l__mod___blocks_3_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_3_norm_mlp_in = self.L__mod___blocks_3_norm_mlp_in(pixel_embed_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_62 = self.L__mod___blocks_3_mlp_in_fc1(l__mod___blocks_3_norm_mlp_in);  l__mod___blocks_3_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_63 = self.L__mod___blocks_3_mlp_in_act(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_64 = self.L__mod___blocks_3_mlp_in_drop1(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_65 = self.L__mod___blocks_3_mlp_in_norm(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_66 = self.L__mod___blocks_3_mlp_in_fc2(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_67 = self.L__mod___blocks_3_mlp_in_drop2(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_3_drop_path_1 = self.L__mod___blocks_3_drop_path(x_67);  x_67 = None
    pixel_embed_12 = pixel_embed_10 + l__mod___blocks_3_drop_path_1;  pixel_embed_10 = l__mod___blocks_3_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_20 = patch_embed_15[(slice(None, None, None), slice(0, 1, None))]
    getitem_21 = patch_embed_15[(slice(None, None, None), slice(1, None, None))];  patch_embed_15 = None
    l__mod___blocks_3_norm1_proj = self.L__mod___blocks_3_norm1_proj(pixel_embed_12)
    reshape_27 = l__mod___blocks_3_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_3_norm1_proj = None
    l__mod___blocks_3_proj = self.L__mod___blocks_3_proj(reshape_27);  reshape_27 = None
    add_19 = getitem_21 + l__mod___blocks_3_proj;  getitem_21 = l__mod___blocks_3_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_16 = torch.cat([getitem_20, add_19], dim = 1);  getitem_20 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_3_norm_out = self.L__mod___blocks_3_norm_out(patch_embed_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_3_attn_out_qk = self.L__mod___blocks_3_attn_out_qk(l__mod___blocks_3_norm_out)
    reshape_28 = l__mod___blocks_3_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_3_attn_out_qk = None
    qk_7 = reshape_28.permute(2, 0, 3, 1, 4);  reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = qk_7.unbind(0);  qk_7 = None
    q_7 = unbind_7[0]
    k_7 = unbind_7[1];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_3_attn_out_v = self.L__mod___blocks_3_attn_out_v(l__mod___blocks_3_norm_out);  l__mod___blocks_3_norm_out = None
    reshape_29 = l__mod___blocks_3_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_3_attn_out_v = None
    v_7 = reshape_29.permute(0, 2, 1, 3);  reshape_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_16 = k_7.transpose(-2, -1);  k_7 = None
    matmul_14 = q_7 @ transpose_16;  q_7 = transpose_16 = None
    attn_21 = matmul_14 * 0.125;  matmul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_22 = attn_21.softmax(dim = -1);  attn_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_23 = self.L__mod___blocks_3_attn_out_attn_drop(attn_22);  attn_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_15 = attn_23 @ v_7;  attn_23 = v_7 = None
    transpose_17 = matmul_15.transpose(1, 2);  matmul_15 = None
    x_68 = transpose_17.reshape(8, 197, -1);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_69 = self.L__mod___blocks_3_attn_out_proj(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_70 = self.L__mod___blocks_3_attn_out_proj_drop(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_3_drop_path_2 = self.L__mod___blocks_3_drop_path(x_70);  x_70 = None
    patch_embed_17 = patch_embed_16 + l__mod___blocks_3_drop_path_2;  patch_embed_16 = l__mod___blocks_3_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_3_norm_mlp = self.L__mod___blocks_3_norm_mlp(patch_embed_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_71 = self.L__mod___blocks_3_mlp_fc1(l__mod___blocks_3_norm_mlp);  l__mod___blocks_3_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_72 = self.L__mod___blocks_3_mlp_act(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_73 = self.L__mod___blocks_3_mlp_drop1(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_74 = self.L__mod___blocks_3_mlp_norm(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_75 = self.L__mod___blocks_3_mlp_fc2(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_76 = self.L__mod___blocks_3_mlp_drop2(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_3_drop_path_3 = self.L__mod___blocks_3_drop_path(x_76);  x_76 = None
    patch_embed_19 = patch_embed_17 + l__mod___blocks_3_drop_path_3;  patch_embed_17 = l__mod___blocks_3_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_4_norm_in = self.L__mod___blocks_4_norm_in(pixel_embed_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_4_attn_in_qk = self.L__mod___blocks_4_attn_in_qk(l__mod___blocks_4_norm_in)
    reshape_31 = l__mod___blocks_4_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_4_attn_in_qk = None
    qk_8 = reshape_31.permute(2, 0, 3, 1, 4);  reshape_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = qk_8.unbind(0);  qk_8 = None
    q_8 = unbind_8[0]
    k_8 = unbind_8[1];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_4_attn_in_v = self.L__mod___blocks_4_attn_in_v(l__mod___blocks_4_norm_in);  l__mod___blocks_4_norm_in = None
    reshape_32 = l__mod___blocks_4_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_4_attn_in_v = None
    v_8 = reshape_32.permute(0, 2, 1, 3);  reshape_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_18 = k_8.transpose(-2, -1);  k_8 = None
    matmul_16 = q_8 @ transpose_18;  q_8 = transpose_18 = None
    attn_24 = matmul_16 * 0.408248290463863;  matmul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_25 = attn_24.softmax(dim = -1);  attn_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_26 = self.L__mod___blocks_4_attn_in_attn_drop(attn_25);  attn_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_17 = attn_26 @ v_8;  attn_26 = v_8 = None
    transpose_19 = matmul_17.transpose(1, 2);  matmul_17 = None
    x_77 = transpose_19.reshape(1568, 16, -1);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_78 = self.L__mod___blocks_4_attn_in_proj(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_79 = self.L__mod___blocks_4_attn_in_proj_drop(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_4_drop_path = self.L__mod___blocks_4_drop_path(x_79);  x_79 = None
    pixel_embed_13 = pixel_embed_12 + l__mod___blocks_4_drop_path;  pixel_embed_12 = l__mod___blocks_4_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_4_norm_mlp_in = self.L__mod___blocks_4_norm_mlp_in(pixel_embed_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_80 = self.L__mod___blocks_4_mlp_in_fc1(l__mod___blocks_4_norm_mlp_in);  l__mod___blocks_4_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_81 = self.L__mod___blocks_4_mlp_in_act(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_82 = self.L__mod___blocks_4_mlp_in_drop1(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_83 = self.L__mod___blocks_4_mlp_in_norm(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_84 = self.L__mod___blocks_4_mlp_in_fc2(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_85 = self.L__mod___blocks_4_mlp_in_drop2(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_4_drop_path_1 = self.L__mod___blocks_4_drop_path(x_85);  x_85 = None
    pixel_embed_15 = pixel_embed_13 + l__mod___blocks_4_drop_path_1;  pixel_embed_13 = l__mod___blocks_4_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_26 = patch_embed_19[(slice(None, None, None), slice(0, 1, None))]
    getitem_27 = patch_embed_19[(slice(None, None, None), slice(1, None, None))];  patch_embed_19 = None
    l__mod___blocks_4_norm1_proj = self.L__mod___blocks_4_norm1_proj(pixel_embed_15)
    reshape_34 = l__mod___blocks_4_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_4_norm1_proj = None
    l__mod___blocks_4_proj = self.L__mod___blocks_4_proj(reshape_34);  reshape_34 = None
    add_24 = getitem_27 + l__mod___blocks_4_proj;  getitem_27 = l__mod___blocks_4_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_20 = torch.cat([getitem_26, add_24], dim = 1);  getitem_26 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_4_norm_out = self.L__mod___blocks_4_norm_out(patch_embed_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_4_attn_out_qk = self.L__mod___blocks_4_attn_out_qk(l__mod___blocks_4_norm_out)
    reshape_35 = l__mod___blocks_4_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_4_attn_out_qk = None
    qk_9 = reshape_35.permute(2, 0, 3, 1, 4);  reshape_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = qk_9.unbind(0);  qk_9 = None
    q_9 = unbind_9[0]
    k_9 = unbind_9[1];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_4_attn_out_v = self.L__mod___blocks_4_attn_out_v(l__mod___blocks_4_norm_out);  l__mod___blocks_4_norm_out = None
    reshape_36 = l__mod___blocks_4_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_4_attn_out_v = None
    v_9 = reshape_36.permute(0, 2, 1, 3);  reshape_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_20 = k_9.transpose(-2, -1);  k_9 = None
    matmul_18 = q_9 @ transpose_20;  q_9 = transpose_20 = None
    attn_27 = matmul_18 * 0.125;  matmul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_28 = attn_27.softmax(dim = -1);  attn_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_29 = self.L__mod___blocks_4_attn_out_attn_drop(attn_28);  attn_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_19 = attn_29 @ v_9;  attn_29 = v_9 = None
    transpose_21 = matmul_19.transpose(1, 2);  matmul_19 = None
    x_86 = transpose_21.reshape(8, 197, -1);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_87 = self.L__mod___blocks_4_attn_out_proj(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_88 = self.L__mod___blocks_4_attn_out_proj_drop(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_4_drop_path_2 = self.L__mod___blocks_4_drop_path(x_88);  x_88 = None
    patch_embed_21 = patch_embed_20 + l__mod___blocks_4_drop_path_2;  patch_embed_20 = l__mod___blocks_4_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_4_norm_mlp = self.L__mod___blocks_4_norm_mlp(patch_embed_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_89 = self.L__mod___blocks_4_mlp_fc1(l__mod___blocks_4_norm_mlp);  l__mod___blocks_4_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_90 = self.L__mod___blocks_4_mlp_act(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_91 = self.L__mod___blocks_4_mlp_drop1(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_92 = self.L__mod___blocks_4_mlp_norm(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_93 = self.L__mod___blocks_4_mlp_fc2(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_94 = self.L__mod___blocks_4_mlp_drop2(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_4_drop_path_3 = self.L__mod___blocks_4_drop_path(x_94);  x_94 = None
    patch_embed_23 = patch_embed_21 + l__mod___blocks_4_drop_path_3;  patch_embed_21 = l__mod___blocks_4_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_5_norm_in = self.L__mod___blocks_5_norm_in(pixel_embed_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_5_attn_in_qk = self.L__mod___blocks_5_attn_in_qk(l__mod___blocks_5_norm_in)
    reshape_38 = l__mod___blocks_5_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_5_attn_in_qk = None
    qk_10 = reshape_38.permute(2, 0, 3, 1, 4);  reshape_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = qk_10.unbind(0);  qk_10 = None
    q_10 = unbind_10[0]
    k_10 = unbind_10[1];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_5_attn_in_v = self.L__mod___blocks_5_attn_in_v(l__mod___blocks_5_norm_in);  l__mod___blocks_5_norm_in = None
    reshape_39 = l__mod___blocks_5_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_5_attn_in_v = None
    v_10 = reshape_39.permute(0, 2, 1, 3);  reshape_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_22 = k_10.transpose(-2, -1);  k_10 = None
    matmul_20 = q_10 @ transpose_22;  q_10 = transpose_22 = None
    attn_30 = matmul_20 * 0.408248290463863;  matmul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_31 = attn_30.softmax(dim = -1);  attn_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_32 = self.L__mod___blocks_5_attn_in_attn_drop(attn_31);  attn_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_21 = attn_32 @ v_10;  attn_32 = v_10 = None
    transpose_23 = matmul_21.transpose(1, 2);  matmul_21 = None
    x_95 = transpose_23.reshape(1568, 16, -1);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_96 = self.L__mod___blocks_5_attn_in_proj(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_97 = self.L__mod___blocks_5_attn_in_proj_drop(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_5_drop_path = self.L__mod___blocks_5_drop_path(x_97);  x_97 = None
    pixel_embed_16 = pixel_embed_15 + l__mod___blocks_5_drop_path;  pixel_embed_15 = l__mod___blocks_5_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_5_norm_mlp_in = self.L__mod___blocks_5_norm_mlp_in(pixel_embed_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_98 = self.L__mod___blocks_5_mlp_in_fc1(l__mod___blocks_5_norm_mlp_in);  l__mod___blocks_5_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_99 = self.L__mod___blocks_5_mlp_in_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_100 = self.L__mod___blocks_5_mlp_in_drop1(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_101 = self.L__mod___blocks_5_mlp_in_norm(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_102 = self.L__mod___blocks_5_mlp_in_fc2(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_103 = self.L__mod___blocks_5_mlp_in_drop2(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_5_drop_path_1 = self.L__mod___blocks_5_drop_path(x_103);  x_103 = None
    pixel_embed_18 = pixel_embed_16 + l__mod___blocks_5_drop_path_1;  pixel_embed_16 = l__mod___blocks_5_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_32 = patch_embed_23[(slice(None, None, None), slice(0, 1, None))]
    getitem_33 = patch_embed_23[(slice(None, None, None), slice(1, None, None))];  patch_embed_23 = None
    l__mod___blocks_5_norm1_proj = self.L__mod___blocks_5_norm1_proj(pixel_embed_18)
    reshape_41 = l__mod___blocks_5_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_5_norm1_proj = None
    l__mod___blocks_5_proj = self.L__mod___blocks_5_proj(reshape_41);  reshape_41 = None
    add_29 = getitem_33 + l__mod___blocks_5_proj;  getitem_33 = l__mod___blocks_5_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_24 = torch.cat([getitem_32, add_29], dim = 1);  getitem_32 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_5_norm_out = self.L__mod___blocks_5_norm_out(patch_embed_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_5_attn_out_qk = self.L__mod___blocks_5_attn_out_qk(l__mod___blocks_5_norm_out)
    reshape_42 = l__mod___blocks_5_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_5_attn_out_qk = None
    qk_11 = reshape_42.permute(2, 0, 3, 1, 4);  reshape_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = qk_11.unbind(0);  qk_11 = None
    q_11 = unbind_11[0]
    k_11 = unbind_11[1];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_5_attn_out_v = self.L__mod___blocks_5_attn_out_v(l__mod___blocks_5_norm_out);  l__mod___blocks_5_norm_out = None
    reshape_43 = l__mod___blocks_5_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_5_attn_out_v = None
    v_11 = reshape_43.permute(0, 2, 1, 3);  reshape_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_24 = k_11.transpose(-2, -1);  k_11 = None
    matmul_22 = q_11 @ transpose_24;  q_11 = transpose_24 = None
    attn_33 = matmul_22 * 0.125;  matmul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_34 = attn_33.softmax(dim = -1);  attn_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_35 = self.L__mod___blocks_5_attn_out_attn_drop(attn_34);  attn_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_23 = attn_35 @ v_11;  attn_35 = v_11 = None
    transpose_25 = matmul_23.transpose(1, 2);  matmul_23 = None
    x_104 = transpose_25.reshape(8, 197, -1);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_105 = self.L__mod___blocks_5_attn_out_proj(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_106 = self.L__mod___blocks_5_attn_out_proj_drop(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_5_drop_path_2 = self.L__mod___blocks_5_drop_path(x_106);  x_106 = None
    patch_embed_25 = patch_embed_24 + l__mod___blocks_5_drop_path_2;  patch_embed_24 = l__mod___blocks_5_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_5_norm_mlp = self.L__mod___blocks_5_norm_mlp(patch_embed_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_107 = self.L__mod___blocks_5_mlp_fc1(l__mod___blocks_5_norm_mlp);  l__mod___blocks_5_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_108 = self.L__mod___blocks_5_mlp_act(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_109 = self.L__mod___blocks_5_mlp_drop1(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_110 = self.L__mod___blocks_5_mlp_norm(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_111 = self.L__mod___blocks_5_mlp_fc2(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_112 = self.L__mod___blocks_5_mlp_drop2(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_5_drop_path_3 = self.L__mod___blocks_5_drop_path(x_112);  x_112 = None
    patch_embed_27 = patch_embed_25 + l__mod___blocks_5_drop_path_3;  patch_embed_25 = l__mod___blocks_5_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_6_norm_in = self.L__mod___blocks_6_norm_in(pixel_embed_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_6_attn_in_qk = self.L__mod___blocks_6_attn_in_qk(l__mod___blocks_6_norm_in)
    reshape_45 = l__mod___blocks_6_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_6_attn_in_qk = None
    qk_12 = reshape_45.permute(2, 0, 3, 1, 4);  reshape_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = qk_12.unbind(0);  qk_12 = None
    q_12 = unbind_12[0]
    k_12 = unbind_12[1];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_6_attn_in_v = self.L__mod___blocks_6_attn_in_v(l__mod___blocks_6_norm_in);  l__mod___blocks_6_norm_in = None
    reshape_46 = l__mod___blocks_6_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_6_attn_in_v = None
    v_12 = reshape_46.permute(0, 2, 1, 3);  reshape_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_26 = k_12.transpose(-2, -1);  k_12 = None
    matmul_24 = q_12 @ transpose_26;  q_12 = transpose_26 = None
    attn_36 = matmul_24 * 0.408248290463863;  matmul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_37 = attn_36.softmax(dim = -1);  attn_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_38 = self.L__mod___blocks_6_attn_in_attn_drop(attn_37);  attn_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_25 = attn_38 @ v_12;  attn_38 = v_12 = None
    transpose_27 = matmul_25.transpose(1, 2);  matmul_25 = None
    x_113 = transpose_27.reshape(1568, 16, -1);  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_114 = self.L__mod___blocks_6_attn_in_proj(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_115 = self.L__mod___blocks_6_attn_in_proj_drop(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_6_drop_path = self.L__mod___blocks_6_drop_path(x_115);  x_115 = None
    pixel_embed_19 = pixel_embed_18 + l__mod___blocks_6_drop_path;  pixel_embed_18 = l__mod___blocks_6_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_6_norm_mlp_in = self.L__mod___blocks_6_norm_mlp_in(pixel_embed_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_116 = self.L__mod___blocks_6_mlp_in_fc1(l__mod___blocks_6_norm_mlp_in);  l__mod___blocks_6_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_117 = self.L__mod___blocks_6_mlp_in_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_118 = self.L__mod___blocks_6_mlp_in_drop1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_119 = self.L__mod___blocks_6_mlp_in_norm(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_120 = self.L__mod___blocks_6_mlp_in_fc2(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_121 = self.L__mod___blocks_6_mlp_in_drop2(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_6_drop_path_1 = self.L__mod___blocks_6_drop_path(x_121);  x_121 = None
    pixel_embed_21 = pixel_embed_19 + l__mod___blocks_6_drop_path_1;  pixel_embed_19 = l__mod___blocks_6_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_38 = patch_embed_27[(slice(None, None, None), slice(0, 1, None))]
    getitem_39 = patch_embed_27[(slice(None, None, None), slice(1, None, None))];  patch_embed_27 = None
    l__mod___blocks_6_norm1_proj = self.L__mod___blocks_6_norm1_proj(pixel_embed_21)
    reshape_48 = l__mod___blocks_6_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_6_norm1_proj = None
    l__mod___blocks_6_proj = self.L__mod___blocks_6_proj(reshape_48);  reshape_48 = None
    add_34 = getitem_39 + l__mod___blocks_6_proj;  getitem_39 = l__mod___blocks_6_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_28 = torch.cat([getitem_38, add_34], dim = 1);  getitem_38 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_6_norm_out = self.L__mod___blocks_6_norm_out(patch_embed_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_6_attn_out_qk = self.L__mod___blocks_6_attn_out_qk(l__mod___blocks_6_norm_out)
    reshape_49 = l__mod___blocks_6_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_6_attn_out_qk = None
    qk_13 = reshape_49.permute(2, 0, 3, 1, 4);  reshape_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = qk_13.unbind(0);  qk_13 = None
    q_13 = unbind_13[0]
    k_13 = unbind_13[1];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_6_attn_out_v = self.L__mod___blocks_6_attn_out_v(l__mod___blocks_6_norm_out);  l__mod___blocks_6_norm_out = None
    reshape_50 = l__mod___blocks_6_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_6_attn_out_v = None
    v_13 = reshape_50.permute(0, 2, 1, 3);  reshape_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_28 = k_13.transpose(-2, -1);  k_13 = None
    matmul_26 = q_13 @ transpose_28;  q_13 = transpose_28 = None
    attn_39 = matmul_26 * 0.125;  matmul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_40 = attn_39.softmax(dim = -1);  attn_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_41 = self.L__mod___blocks_6_attn_out_attn_drop(attn_40);  attn_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_27 = attn_41 @ v_13;  attn_41 = v_13 = None
    transpose_29 = matmul_27.transpose(1, 2);  matmul_27 = None
    x_122 = transpose_29.reshape(8, 197, -1);  transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_123 = self.L__mod___blocks_6_attn_out_proj(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_124 = self.L__mod___blocks_6_attn_out_proj_drop(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_6_drop_path_2 = self.L__mod___blocks_6_drop_path(x_124);  x_124 = None
    patch_embed_29 = patch_embed_28 + l__mod___blocks_6_drop_path_2;  patch_embed_28 = l__mod___blocks_6_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_6_norm_mlp = self.L__mod___blocks_6_norm_mlp(patch_embed_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_125 = self.L__mod___blocks_6_mlp_fc1(l__mod___blocks_6_norm_mlp);  l__mod___blocks_6_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_126 = self.L__mod___blocks_6_mlp_act(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_127 = self.L__mod___blocks_6_mlp_drop1(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_128 = self.L__mod___blocks_6_mlp_norm(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_129 = self.L__mod___blocks_6_mlp_fc2(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_130 = self.L__mod___blocks_6_mlp_drop2(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_6_drop_path_3 = self.L__mod___blocks_6_drop_path(x_130);  x_130 = None
    patch_embed_31 = patch_embed_29 + l__mod___blocks_6_drop_path_3;  patch_embed_29 = l__mod___blocks_6_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_7_norm_in = self.L__mod___blocks_7_norm_in(pixel_embed_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_7_attn_in_qk = self.L__mod___blocks_7_attn_in_qk(l__mod___blocks_7_norm_in)
    reshape_52 = l__mod___blocks_7_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_7_attn_in_qk = None
    qk_14 = reshape_52.permute(2, 0, 3, 1, 4);  reshape_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = qk_14.unbind(0);  qk_14 = None
    q_14 = unbind_14[0]
    k_14 = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_7_attn_in_v = self.L__mod___blocks_7_attn_in_v(l__mod___blocks_7_norm_in);  l__mod___blocks_7_norm_in = None
    reshape_53 = l__mod___blocks_7_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_7_attn_in_v = None
    v_14 = reshape_53.permute(0, 2, 1, 3);  reshape_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_30 = k_14.transpose(-2, -1);  k_14 = None
    matmul_28 = q_14 @ transpose_30;  q_14 = transpose_30 = None
    attn_42 = matmul_28 * 0.408248290463863;  matmul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_43 = attn_42.softmax(dim = -1);  attn_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_44 = self.L__mod___blocks_7_attn_in_attn_drop(attn_43);  attn_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_29 = attn_44 @ v_14;  attn_44 = v_14 = None
    transpose_31 = matmul_29.transpose(1, 2);  matmul_29 = None
    x_131 = transpose_31.reshape(1568, 16, -1);  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_132 = self.L__mod___blocks_7_attn_in_proj(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_133 = self.L__mod___blocks_7_attn_in_proj_drop(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_7_drop_path = self.L__mod___blocks_7_drop_path(x_133);  x_133 = None
    pixel_embed_22 = pixel_embed_21 + l__mod___blocks_7_drop_path;  pixel_embed_21 = l__mod___blocks_7_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_7_norm_mlp_in = self.L__mod___blocks_7_norm_mlp_in(pixel_embed_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_134 = self.L__mod___blocks_7_mlp_in_fc1(l__mod___blocks_7_norm_mlp_in);  l__mod___blocks_7_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_135 = self.L__mod___blocks_7_mlp_in_act(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_136 = self.L__mod___blocks_7_mlp_in_drop1(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_137 = self.L__mod___blocks_7_mlp_in_norm(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_138 = self.L__mod___blocks_7_mlp_in_fc2(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_139 = self.L__mod___blocks_7_mlp_in_drop2(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_7_drop_path_1 = self.L__mod___blocks_7_drop_path(x_139);  x_139 = None
    pixel_embed_24 = pixel_embed_22 + l__mod___blocks_7_drop_path_1;  pixel_embed_22 = l__mod___blocks_7_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_44 = patch_embed_31[(slice(None, None, None), slice(0, 1, None))]
    getitem_45 = patch_embed_31[(slice(None, None, None), slice(1, None, None))];  patch_embed_31 = None
    l__mod___blocks_7_norm1_proj = self.L__mod___blocks_7_norm1_proj(pixel_embed_24)
    reshape_55 = l__mod___blocks_7_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_7_norm1_proj = None
    l__mod___blocks_7_proj = self.L__mod___blocks_7_proj(reshape_55);  reshape_55 = None
    add_39 = getitem_45 + l__mod___blocks_7_proj;  getitem_45 = l__mod___blocks_7_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_32 = torch.cat([getitem_44, add_39], dim = 1);  getitem_44 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_7_norm_out = self.L__mod___blocks_7_norm_out(patch_embed_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_7_attn_out_qk = self.L__mod___blocks_7_attn_out_qk(l__mod___blocks_7_norm_out)
    reshape_56 = l__mod___blocks_7_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_7_attn_out_qk = None
    qk_15 = reshape_56.permute(2, 0, 3, 1, 4);  reshape_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = qk_15.unbind(0);  qk_15 = None
    q_15 = unbind_15[0]
    k_15 = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_7_attn_out_v = self.L__mod___blocks_7_attn_out_v(l__mod___blocks_7_norm_out);  l__mod___blocks_7_norm_out = None
    reshape_57 = l__mod___blocks_7_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_7_attn_out_v = None
    v_15 = reshape_57.permute(0, 2, 1, 3);  reshape_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_32 = k_15.transpose(-2, -1);  k_15 = None
    matmul_30 = q_15 @ transpose_32;  q_15 = transpose_32 = None
    attn_45 = matmul_30 * 0.125;  matmul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_46 = attn_45.softmax(dim = -1);  attn_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_47 = self.L__mod___blocks_7_attn_out_attn_drop(attn_46);  attn_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_31 = attn_47 @ v_15;  attn_47 = v_15 = None
    transpose_33 = matmul_31.transpose(1, 2);  matmul_31 = None
    x_140 = transpose_33.reshape(8, 197, -1);  transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_141 = self.L__mod___blocks_7_attn_out_proj(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_142 = self.L__mod___blocks_7_attn_out_proj_drop(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_7_drop_path_2 = self.L__mod___blocks_7_drop_path(x_142);  x_142 = None
    patch_embed_33 = patch_embed_32 + l__mod___blocks_7_drop_path_2;  patch_embed_32 = l__mod___blocks_7_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_7_norm_mlp = self.L__mod___blocks_7_norm_mlp(patch_embed_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_143 = self.L__mod___blocks_7_mlp_fc1(l__mod___blocks_7_norm_mlp);  l__mod___blocks_7_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_144 = self.L__mod___blocks_7_mlp_act(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_145 = self.L__mod___blocks_7_mlp_drop1(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_146 = self.L__mod___blocks_7_mlp_norm(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_147 = self.L__mod___blocks_7_mlp_fc2(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_148 = self.L__mod___blocks_7_mlp_drop2(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_7_drop_path_3 = self.L__mod___blocks_7_drop_path(x_148);  x_148 = None
    patch_embed_35 = patch_embed_33 + l__mod___blocks_7_drop_path_3;  patch_embed_33 = l__mod___blocks_7_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_8_norm_in = self.L__mod___blocks_8_norm_in(pixel_embed_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_8_attn_in_qk = self.L__mod___blocks_8_attn_in_qk(l__mod___blocks_8_norm_in)
    reshape_59 = l__mod___blocks_8_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_8_attn_in_qk = None
    qk_16 = reshape_59.permute(2, 0, 3, 1, 4);  reshape_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = qk_16.unbind(0);  qk_16 = None
    q_16 = unbind_16[0]
    k_16 = unbind_16[1];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_8_attn_in_v = self.L__mod___blocks_8_attn_in_v(l__mod___blocks_8_norm_in);  l__mod___blocks_8_norm_in = None
    reshape_60 = l__mod___blocks_8_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_8_attn_in_v = None
    v_16 = reshape_60.permute(0, 2, 1, 3);  reshape_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_34 = k_16.transpose(-2, -1);  k_16 = None
    matmul_32 = q_16 @ transpose_34;  q_16 = transpose_34 = None
    attn_48 = matmul_32 * 0.408248290463863;  matmul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_49 = attn_48.softmax(dim = -1);  attn_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_50 = self.L__mod___blocks_8_attn_in_attn_drop(attn_49);  attn_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_33 = attn_50 @ v_16;  attn_50 = v_16 = None
    transpose_35 = matmul_33.transpose(1, 2);  matmul_33 = None
    x_149 = transpose_35.reshape(1568, 16, -1);  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_150 = self.L__mod___blocks_8_attn_in_proj(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_151 = self.L__mod___blocks_8_attn_in_proj_drop(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_8_drop_path = self.L__mod___blocks_8_drop_path(x_151);  x_151 = None
    pixel_embed_25 = pixel_embed_24 + l__mod___blocks_8_drop_path;  pixel_embed_24 = l__mod___blocks_8_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_8_norm_mlp_in = self.L__mod___blocks_8_norm_mlp_in(pixel_embed_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_152 = self.L__mod___blocks_8_mlp_in_fc1(l__mod___blocks_8_norm_mlp_in);  l__mod___blocks_8_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_153 = self.L__mod___blocks_8_mlp_in_act(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_154 = self.L__mod___blocks_8_mlp_in_drop1(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_155 = self.L__mod___blocks_8_mlp_in_norm(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_156 = self.L__mod___blocks_8_mlp_in_fc2(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_157 = self.L__mod___blocks_8_mlp_in_drop2(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_8_drop_path_1 = self.L__mod___blocks_8_drop_path(x_157);  x_157 = None
    pixel_embed_27 = pixel_embed_25 + l__mod___blocks_8_drop_path_1;  pixel_embed_25 = l__mod___blocks_8_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_50 = patch_embed_35[(slice(None, None, None), slice(0, 1, None))]
    getitem_51 = patch_embed_35[(slice(None, None, None), slice(1, None, None))];  patch_embed_35 = None
    l__mod___blocks_8_norm1_proj = self.L__mod___blocks_8_norm1_proj(pixel_embed_27)
    reshape_62 = l__mod___blocks_8_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_8_norm1_proj = None
    l__mod___blocks_8_proj = self.L__mod___blocks_8_proj(reshape_62);  reshape_62 = None
    add_44 = getitem_51 + l__mod___blocks_8_proj;  getitem_51 = l__mod___blocks_8_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_36 = torch.cat([getitem_50, add_44], dim = 1);  getitem_50 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_8_norm_out = self.L__mod___blocks_8_norm_out(patch_embed_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_8_attn_out_qk = self.L__mod___blocks_8_attn_out_qk(l__mod___blocks_8_norm_out)
    reshape_63 = l__mod___blocks_8_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_8_attn_out_qk = None
    qk_17 = reshape_63.permute(2, 0, 3, 1, 4);  reshape_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = qk_17.unbind(0);  qk_17 = None
    q_17 = unbind_17[0]
    k_17 = unbind_17[1];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_8_attn_out_v = self.L__mod___blocks_8_attn_out_v(l__mod___blocks_8_norm_out);  l__mod___blocks_8_norm_out = None
    reshape_64 = l__mod___blocks_8_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_8_attn_out_v = None
    v_17 = reshape_64.permute(0, 2, 1, 3);  reshape_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_36 = k_17.transpose(-2, -1);  k_17 = None
    matmul_34 = q_17 @ transpose_36;  q_17 = transpose_36 = None
    attn_51 = matmul_34 * 0.125;  matmul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_52 = attn_51.softmax(dim = -1);  attn_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_53 = self.L__mod___blocks_8_attn_out_attn_drop(attn_52);  attn_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_35 = attn_53 @ v_17;  attn_53 = v_17 = None
    transpose_37 = matmul_35.transpose(1, 2);  matmul_35 = None
    x_158 = transpose_37.reshape(8, 197, -1);  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_159 = self.L__mod___blocks_8_attn_out_proj(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_160 = self.L__mod___blocks_8_attn_out_proj_drop(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_8_drop_path_2 = self.L__mod___blocks_8_drop_path(x_160);  x_160 = None
    patch_embed_37 = patch_embed_36 + l__mod___blocks_8_drop_path_2;  patch_embed_36 = l__mod___blocks_8_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_8_norm_mlp = self.L__mod___blocks_8_norm_mlp(patch_embed_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_161 = self.L__mod___blocks_8_mlp_fc1(l__mod___blocks_8_norm_mlp);  l__mod___blocks_8_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_162 = self.L__mod___blocks_8_mlp_act(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_163 = self.L__mod___blocks_8_mlp_drop1(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_164 = self.L__mod___blocks_8_mlp_norm(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_165 = self.L__mod___blocks_8_mlp_fc2(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_166 = self.L__mod___blocks_8_mlp_drop2(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_8_drop_path_3 = self.L__mod___blocks_8_drop_path(x_166);  x_166 = None
    patch_embed_39 = patch_embed_37 + l__mod___blocks_8_drop_path_3;  patch_embed_37 = l__mod___blocks_8_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_9_norm_in = self.L__mod___blocks_9_norm_in(pixel_embed_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_9_attn_in_qk = self.L__mod___blocks_9_attn_in_qk(l__mod___blocks_9_norm_in)
    reshape_66 = l__mod___blocks_9_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_9_attn_in_qk = None
    qk_18 = reshape_66.permute(2, 0, 3, 1, 4);  reshape_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = qk_18.unbind(0);  qk_18 = None
    q_18 = unbind_18[0]
    k_18 = unbind_18[1];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_9_attn_in_v = self.L__mod___blocks_9_attn_in_v(l__mod___blocks_9_norm_in);  l__mod___blocks_9_norm_in = None
    reshape_67 = l__mod___blocks_9_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_9_attn_in_v = None
    v_18 = reshape_67.permute(0, 2, 1, 3);  reshape_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_38 = k_18.transpose(-2, -1);  k_18 = None
    matmul_36 = q_18 @ transpose_38;  q_18 = transpose_38 = None
    attn_54 = matmul_36 * 0.408248290463863;  matmul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_55 = attn_54.softmax(dim = -1);  attn_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_56 = self.L__mod___blocks_9_attn_in_attn_drop(attn_55);  attn_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_37 = attn_56 @ v_18;  attn_56 = v_18 = None
    transpose_39 = matmul_37.transpose(1, 2);  matmul_37 = None
    x_167 = transpose_39.reshape(1568, 16, -1);  transpose_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_168 = self.L__mod___blocks_9_attn_in_proj(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_169 = self.L__mod___blocks_9_attn_in_proj_drop(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_9_drop_path = self.L__mod___blocks_9_drop_path(x_169);  x_169 = None
    pixel_embed_28 = pixel_embed_27 + l__mod___blocks_9_drop_path;  pixel_embed_27 = l__mod___blocks_9_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_9_norm_mlp_in = self.L__mod___blocks_9_norm_mlp_in(pixel_embed_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_170 = self.L__mod___blocks_9_mlp_in_fc1(l__mod___blocks_9_norm_mlp_in);  l__mod___blocks_9_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_171 = self.L__mod___blocks_9_mlp_in_act(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_172 = self.L__mod___blocks_9_mlp_in_drop1(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_173 = self.L__mod___blocks_9_mlp_in_norm(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_174 = self.L__mod___blocks_9_mlp_in_fc2(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_175 = self.L__mod___blocks_9_mlp_in_drop2(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_9_drop_path_1 = self.L__mod___blocks_9_drop_path(x_175);  x_175 = None
    pixel_embed_30 = pixel_embed_28 + l__mod___blocks_9_drop_path_1;  pixel_embed_28 = l__mod___blocks_9_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_56 = patch_embed_39[(slice(None, None, None), slice(0, 1, None))]
    getitem_57 = patch_embed_39[(slice(None, None, None), slice(1, None, None))];  patch_embed_39 = None
    l__mod___blocks_9_norm1_proj = self.L__mod___blocks_9_norm1_proj(pixel_embed_30)
    reshape_69 = l__mod___blocks_9_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_9_norm1_proj = None
    l__mod___blocks_9_proj = self.L__mod___blocks_9_proj(reshape_69);  reshape_69 = None
    add_49 = getitem_57 + l__mod___blocks_9_proj;  getitem_57 = l__mod___blocks_9_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_40 = torch.cat([getitem_56, add_49], dim = 1);  getitem_56 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_9_norm_out = self.L__mod___blocks_9_norm_out(patch_embed_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_9_attn_out_qk = self.L__mod___blocks_9_attn_out_qk(l__mod___blocks_9_norm_out)
    reshape_70 = l__mod___blocks_9_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_9_attn_out_qk = None
    qk_19 = reshape_70.permute(2, 0, 3, 1, 4);  reshape_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = qk_19.unbind(0);  qk_19 = None
    q_19 = unbind_19[0]
    k_19 = unbind_19[1];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_9_attn_out_v = self.L__mod___blocks_9_attn_out_v(l__mod___blocks_9_norm_out);  l__mod___blocks_9_norm_out = None
    reshape_71 = l__mod___blocks_9_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_9_attn_out_v = None
    v_19 = reshape_71.permute(0, 2, 1, 3);  reshape_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_40 = k_19.transpose(-2, -1);  k_19 = None
    matmul_38 = q_19 @ transpose_40;  q_19 = transpose_40 = None
    attn_57 = matmul_38 * 0.125;  matmul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_58 = attn_57.softmax(dim = -1);  attn_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_59 = self.L__mod___blocks_9_attn_out_attn_drop(attn_58);  attn_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_39 = attn_59 @ v_19;  attn_59 = v_19 = None
    transpose_41 = matmul_39.transpose(1, 2);  matmul_39 = None
    x_176 = transpose_41.reshape(8, 197, -1);  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_177 = self.L__mod___blocks_9_attn_out_proj(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_178 = self.L__mod___blocks_9_attn_out_proj_drop(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_9_drop_path_2 = self.L__mod___blocks_9_drop_path(x_178);  x_178 = None
    patch_embed_41 = patch_embed_40 + l__mod___blocks_9_drop_path_2;  patch_embed_40 = l__mod___blocks_9_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_9_norm_mlp = self.L__mod___blocks_9_norm_mlp(patch_embed_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_179 = self.L__mod___blocks_9_mlp_fc1(l__mod___blocks_9_norm_mlp);  l__mod___blocks_9_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_180 = self.L__mod___blocks_9_mlp_act(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_181 = self.L__mod___blocks_9_mlp_drop1(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_182 = self.L__mod___blocks_9_mlp_norm(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_183 = self.L__mod___blocks_9_mlp_fc2(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_184 = self.L__mod___blocks_9_mlp_drop2(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_9_drop_path_3 = self.L__mod___blocks_9_drop_path(x_184);  x_184 = None
    patch_embed_43 = patch_embed_41 + l__mod___blocks_9_drop_path_3;  patch_embed_41 = l__mod___blocks_9_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_10_norm_in = self.L__mod___blocks_10_norm_in(pixel_embed_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_10_attn_in_qk = self.L__mod___blocks_10_attn_in_qk(l__mod___blocks_10_norm_in)
    reshape_73 = l__mod___blocks_10_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_10_attn_in_qk = None
    qk_20 = reshape_73.permute(2, 0, 3, 1, 4);  reshape_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = qk_20.unbind(0);  qk_20 = None
    q_20 = unbind_20[0]
    k_20 = unbind_20[1];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_10_attn_in_v = self.L__mod___blocks_10_attn_in_v(l__mod___blocks_10_norm_in);  l__mod___blocks_10_norm_in = None
    reshape_74 = l__mod___blocks_10_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_10_attn_in_v = None
    v_20 = reshape_74.permute(0, 2, 1, 3);  reshape_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_42 = k_20.transpose(-2, -1);  k_20 = None
    matmul_40 = q_20 @ transpose_42;  q_20 = transpose_42 = None
    attn_60 = matmul_40 * 0.408248290463863;  matmul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_61 = attn_60.softmax(dim = -1);  attn_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_62 = self.L__mod___blocks_10_attn_in_attn_drop(attn_61);  attn_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_41 = attn_62 @ v_20;  attn_62 = v_20 = None
    transpose_43 = matmul_41.transpose(1, 2);  matmul_41 = None
    x_185 = transpose_43.reshape(1568, 16, -1);  transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_186 = self.L__mod___blocks_10_attn_in_proj(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_187 = self.L__mod___blocks_10_attn_in_proj_drop(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_10_drop_path = self.L__mod___blocks_10_drop_path(x_187);  x_187 = None
    pixel_embed_31 = pixel_embed_30 + l__mod___blocks_10_drop_path;  pixel_embed_30 = l__mod___blocks_10_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_10_norm_mlp_in = self.L__mod___blocks_10_norm_mlp_in(pixel_embed_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_188 = self.L__mod___blocks_10_mlp_in_fc1(l__mod___blocks_10_norm_mlp_in);  l__mod___blocks_10_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_189 = self.L__mod___blocks_10_mlp_in_act(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_190 = self.L__mod___blocks_10_mlp_in_drop1(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_191 = self.L__mod___blocks_10_mlp_in_norm(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_192 = self.L__mod___blocks_10_mlp_in_fc2(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_193 = self.L__mod___blocks_10_mlp_in_drop2(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_10_drop_path_1 = self.L__mod___blocks_10_drop_path(x_193);  x_193 = None
    pixel_embed_33 = pixel_embed_31 + l__mod___blocks_10_drop_path_1;  pixel_embed_31 = l__mod___blocks_10_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_62 = patch_embed_43[(slice(None, None, None), slice(0, 1, None))]
    getitem_63 = patch_embed_43[(slice(None, None, None), slice(1, None, None))];  patch_embed_43 = None
    l__mod___blocks_10_norm1_proj = self.L__mod___blocks_10_norm1_proj(pixel_embed_33)
    reshape_76 = l__mod___blocks_10_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_10_norm1_proj = None
    l__mod___blocks_10_proj = self.L__mod___blocks_10_proj(reshape_76);  reshape_76 = None
    add_54 = getitem_63 + l__mod___blocks_10_proj;  getitem_63 = l__mod___blocks_10_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_44 = torch.cat([getitem_62, add_54], dim = 1);  getitem_62 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_10_norm_out = self.L__mod___blocks_10_norm_out(patch_embed_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_10_attn_out_qk = self.L__mod___blocks_10_attn_out_qk(l__mod___blocks_10_norm_out)
    reshape_77 = l__mod___blocks_10_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_10_attn_out_qk = None
    qk_21 = reshape_77.permute(2, 0, 3, 1, 4);  reshape_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = qk_21.unbind(0);  qk_21 = None
    q_21 = unbind_21[0]
    k_21 = unbind_21[1];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_10_attn_out_v = self.L__mod___blocks_10_attn_out_v(l__mod___blocks_10_norm_out);  l__mod___blocks_10_norm_out = None
    reshape_78 = l__mod___blocks_10_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_10_attn_out_v = None
    v_21 = reshape_78.permute(0, 2, 1, 3);  reshape_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_44 = k_21.transpose(-2, -1);  k_21 = None
    matmul_42 = q_21 @ transpose_44;  q_21 = transpose_44 = None
    attn_63 = matmul_42 * 0.125;  matmul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_64 = attn_63.softmax(dim = -1);  attn_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_65 = self.L__mod___blocks_10_attn_out_attn_drop(attn_64);  attn_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_43 = attn_65 @ v_21;  attn_65 = v_21 = None
    transpose_45 = matmul_43.transpose(1, 2);  matmul_43 = None
    x_194 = transpose_45.reshape(8, 197, -1);  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_195 = self.L__mod___blocks_10_attn_out_proj(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_196 = self.L__mod___blocks_10_attn_out_proj_drop(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_10_drop_path_2 = self.L__mod___blocks_10_drop_path(x_196);  x_196 = None
    patch_embed_45 = patch_embed_44 + l__mod___blocks_10_drop_path_2;  patch_embed_44 = l__mod___blocks_10_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_10_norm_mlp = self.L__mod___blocks_10_norm_mlp(patch_embed_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_197 = self.L__mod___blocks_10_mlp_fc1(l__mod___blocks_10_norm_mlp);  l__mod___blocks_10_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_198 = self.L__mod___blocks_10_mlp_act(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_199 = self.L__mod___blocks_10_mlp_drop1(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_200 = self.L__mod___blocks_10_mlp_norm(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_201 = self.L__mod___blocks_10_mlp_fc2(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_202 = self.L__mod___blocks_10_mlp_drop2(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_10_drop_path_3 = self.L__mod___blocks_10_drop_path(x_202);  x_202 = None
    patch_embed_47 = patch_embed_45 + l__mod___blocks_10_drop_path_3;  patch_embed_45 = l__mod___blocks_10_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_11_norm_in = self.L__mod___blocks_11_norm_in(pixel_embed_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_11_attn_in_qk = self.L__mod___blocks_11_attn_in_qk(l__mod___blocks_11_norm_in)
    reshape_80 = l__mod___blocks_11_attn_in_qk.reshape(1568, 16, 2, 4, 6);  l__mod___blocks_11_attn_in_qk = None
    qk_22 = reshape_80.permute(2, 0, 3, 1, 4);  reshape_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = qk_22.unbind(0);  qk_22 = None
    q_22 = unbind_22[0]
    k_22 = unbind_22[1];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_11_attn_in_v = self.L__mod___blocks_11_attn_in_v(l__mod___blocks_11_norm_in);  l__mod___blocks_11_norm_in = None
    reshape_81 = l__mod___blocks_11_attn_in_v.reshape(1568, 16, 4, -1);  l__mod___blocks_11_attn_in_v = None
    v_22 = reshape_81.permute(0, 2, 1, 3);  reshape_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_46 = k_22.transpose(-2, -1);  k_22 = None
    matmul_44 = q_22 @ transpose_46;  q_22 = transpose_46 = None
    attn_66 = matmul_44 * 0.408248290463863;  matmul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_67 = attn_66.softmax(dim = -1);  attn_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_68 = self.L__mod___blocks_11_attn_in_attn_drop(attn_67);  attn_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_45 = attn_68 @ v_22;  attn_68 = v_22 = None
    transpose_47 = matmul_45.transpose(1, 2);  matmul_45 = None
    x_203 = transpose_47.reshape(1568, 16, -1);  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_204 = self.L__mod___blocks_11_attn_in_proj(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_205 = self.L__mod___blocks_11_attn_in_proj_drop(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    l__mod___blocks_11_drop_path = self.L__mod___blocks_11_drop_path(x_205);  x_205 = None
    pixel_embed_34 = pixel_embed_33 + l__mod___blocks_11_drop_path;  pixel_embed_33 = l__mod___blocks_11_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_11_norm_mlp_in = self.L__mod___blocks_11_norm_mlp_in(pixel_embed_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_206 = self.L__mod___blocks_11_mlp_in_fc1(l__mod___blocks_11_norm_mlp_in);  l__mod___blocks_11_norm_mlp_in = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_207 = self.L__mod___blocks_11_mlp_in_act(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_208 = self.L__mod___blocks_11_mlp_in_drop1(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_209 = self.L__mod___blocks_11_mlp_in_norm(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_210 = self.L__mod___blocks_11_mlp_in_fc2(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_211 = self.L__mod___blocks_11_mlp_in_drop2(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    l__mod___blocks_11_drop_path_1 = self.L__mod___blocks_11_drop_path(x_211);  x_211 = None
    pixel_embed_36 = pixel_embed_34 + l__mod___blocks_11_drop_path_1;  pixel_embed_34 = l__mod___blocks_11_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    getitem_68 = patch_embed_47[(slice(None, None, None), slice(0, 1, None))]
    getitem_69 = patch_embed_47[(slice(None, None, None), slice(1, None, None))];  patch_embed_47 = None
    l__mod___blocks_11_norm1_proj = self.L__mod___blocks_11_norm1_proj(pixel_embed_36);  pixel_embed_36 = None
    reshape_83 = l__mod___blocks_11_norm1_proj.reshape(8, 196, -1);  l__mod___blocks_11_norm1_proj = None
    l__mod___blocks_11_proj = self.L__mod___blocks_11_proj(reshape_83);  reshape_83 = None
    add_59 = getitem_69 + l__mod___blocks_11_proj;  getitem_69 = l__mod___blocks_11_proj = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    patch_embed_48 = torch.cat([getitem_68, add_59], dim = 1);  getitem_68 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_11_norm_out = self.L__mod___blocks_11_norm_out(patch_embed_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___blocks_11_attn_out_qk = self.L__mod___blocks_11_attn_out_qk(l__mod___blocks_11_norm_out)
    reshape_84 = l__mod___blocks_11_attn_out_qk.reshape(8, 197, 2, 6, 64);  l__mod___blocks_11_attn_out_qk = None
    qk_23 = reshape_84.permute(2, 0, 3, 1, 4);  reshape_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = qk_23.unbind(0);  qk_23 = None
    q_23 = unbind_23[0]
    k_23 = unbind_23[1];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    l__mod___blocks_11_attn_out_v = self.L__mod___blocks_11_attn_out_v(l__mod___blocks_11_norm_out);  l__mod___blocks_11_norm_out = None
    reshape_85 = l__mod___blocks_11_attn_out_v.reshape(8, 197, 6, -1);  l__mod___blocks_11_attn_out_v = None
    v_23 = reshape_85.permute(0, 2, 1, 3);  reshape_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_48 = k_23.transpose(-2, -1);  k_23 = None
    matmul_46 = q_23 @ transpose_48;  q_23 = transpose_48 = None
    attn_69 = matmul_46 * 0.125;  matmul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    attn_70 = attn_69.softmax(dim = -1);  attn_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:71, code: attn = self.attn_drop(attn)
    attn_71 = self.L__mod___blocks_11_attn_out_attn_drop(attn_70);  attn_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    matmul_47 = attn_71 @ v_23;  attn_71 = v_23 = None
    transpose_49 = matmul_47.transpose(1, 2);  matmul_47 = None
    x_212 = transpose_49.reshape(8, 197, -1);  transpose_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    x_213 = self.L__mod___blocks_11_attn_out_proj(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:75, code: x = self.proj_drop(x)
    x_214 = self.L__mod___blocks_11_attn_out_proj_drop(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    l__mod___blocks_11_drop_path_2 = self.L__mod___blocks_11_drop_path(x_214);  x_214 = None
    patch_embed_49 = patch_embed_48 + l__mod___blocks_11_drop_path_2;  patch_embed_48 = l__mod___blocks_11_drop_path_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_11_norm_mlp = self.L__mod___blocks_11_norm_mlp(patch_embed_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_215 = self.L__mod___blocks_11_mlp_fc1(l__mod___blocks_11_norm_mlp);  l__mod___blocks_11_norm_mlp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_216 = self.L__mod___blocks_11_mlp_act(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_217 = self.L__mod___blocks_11_mlp_drop1(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_218 = self.L__mod___blocks_11_mlp_norm(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_219 = self.L__mod___blocks_11_mlp_fc2(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_220 = self.L__mod___blocks_11_mlp_drop2(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    l__mod___blocks_11_drop_path_3 = self.L__mod___blocks_11_drop_path(x_220);  x_220 = None
    patch_embed_51 = patch_embed_49 + l__mod___blocks_11_drop_path_3;  patch_embed_49 = l__mod___blocks_11_drop_path_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:323, code: patch_embed = self.norm(patch_embed)
    x_221 = self.L__mod___norm(patch_embed_51);  patch_embed_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:328, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    x_222 = x_221[(slice(None, None, None), 0)];  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:329, code: x = self.head_drop(x)
    x_223 = self.L__mod___head_drop(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:330, code: return x if pre_logits else self.head(x)
    x_224 = self.L__mod___head(x_223);  x_223 = None
    return (x_224,)
    