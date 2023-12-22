from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x = self.L__mod___patch_embed_proj(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/format.py:43, code: x = x.permute(0, 2, 3, 1)
    x_2 = x.permute(0, 2, 3, 1);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_4 = self.L__mod___patch_embed_norm(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:438, code: x = self.downsample(x)
    x_5 = self.getattr_L__mod___layers___0___downsample(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x = self.getattr_getattr_L__mod___layers___0___blocks___0___norm1(x_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_1 = torch.nn.functional.pad(shifted_x, (0, 0, 0, 0, 0, 0));  shifted_x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_6 = shifted_x_1.view(8, 8, 7, 8, 7, 128);  shifted_x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_1 = x_6.permute(0, 1, 3, 2, 4, 5);  x_6 = None
    contiguous = permute_1.contiguous();  permute_1 = None
    x_windows = contiguous.view(-1, 7, 7, 128);  contiguous = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_1 = x_windows.view(-1, 49, 128);  x_windows = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___0___blocks___0___attn_qkv = self.getattr_getattr_L__mod___layers___0___blocks___0___attn_qkv(x_windows_1);  x_windows_1 = None
    reshape = getattr_getattr_l__mod___layers___0___blocks___0___attn_qkv.reshape(512, 49, 3, 4, -1);  getattr_getattr_l__mod___layers___0___blocks___0___attn_qkv = None
    qkv = reshape.permute(2, 0, 3, 1, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_1 = q * 0.1767766952966369;  q = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose = k.transpose(-2, -1);  k = None
    attn = q_1 @ transpose;  q_1 = transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___0___blocks___0___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___0___blocks___0___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___0___blocks___0___attn_relative_position_index = self.getattr_getattr_L__mod___layers___0___blocks___0___attn_relative_position_index
    view_3 = getattr_getattr_l__mod___layers___0___blocks___0___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___0___blocks___0___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_3 = getattr_getattr_l__mod___layers___0___blocks___0___attn_relative_position_bias_table[view_3];  getattr_getattr_l__mod___layers___0___blocks___0___attn_relative_position_bias_table = view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = getitem_3.view(49, 49, -1);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_3 = relative_position_bias.permute(2, 0, 1);  relative_position_bias = None
    relative_position_bias_1 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze = relative_position_bias_1.unsqueeze(0);  relative_position_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_1 = attn + unsqueeze;  attn = unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_2 = self.getattr_getattr_L__mod___layers___0___blocks___0___attn_softmax(attn_1);  attn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_3 = self.getattr_getattr_L__mod___layers___0___blocks___0___attn_attn_drop(attn_2);  attn_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_7 = attn_3 @ v;  attn_3 = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_1 = x_7.transpose(1, 2);  x_7 = None
    x_8 = transpose_1.reshape(512, 49, -1);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_9 = self.getattr_getattr_L__mod___layers___0___blocks___0___attn_proj(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows = self.getattr_getattr_L__mod___layers___0___blocks___0___attn_proj_drop(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_1 = attn_windows.view(-1, 7, 7, 128);  attn_windows = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_11 = attn_windows_1.view(-1, 8, 8, 7, 7, 128);  attn_windows_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_4 = x_11.permute(0, 1, 3, 2, 4, 5);  x_11 = None
    contiguous_2 = permute_4.contiguous();  permute_4 = None
    shifted_x_2 = contiguous_2.view(-1, 56, 56, 128);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_4 = shifted_x_2[(slice(None, None, None), slice(None, 56, None), slice(None, 56, None), slice(None, None, None))];  shifted_x_2 = None
    x_13 = getitem_4.contiguous();  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___0___blocks___0___drop_path1 = self.getattr_getattr_L__mod___layers___0___blocks___0___drop_path1(x_13);  x_13 = None
    x_14 = x_5 + getattr_getattr_l__mod___layers___0___blocks___0___drop_path1;  x_5 = getattr_getattr_l__mod___layers___0___blocks___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_15 = x_14.reshape(8, -1, 128);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___0___blocks___0___norm2 = self.getattr_getattr_L__mod___layers___0___blocks___0___norm2(x_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_16 = self.getattr_getattr_L__mod___layers___0___blocks___0___mlp_fc1(getattr_getattr_l__mod___layers___0___blocks___0___norm2);  getattr_getattr_l__mod___layers___0___blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_17 = self.getattr_getattr_L__mod___layers___0___blocks___0___mlp_act(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_18 = self.getattr_getattr_L__mod___layers___0___blocks___0___mlp_drop1(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_19 = self.getattr_getattr_L__mod___layers___0___blocks___0___mlp_norm(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_20 = self.getattr_getattr_L__mod___layers___0___blocks___0___mlp_fc2(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_21 = self.getattr_getattr_L__mod___layers___0___blocks___0___mlp_drop2(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___0___blocks___0___drop_path2 = self.getattr_getattr_L__mod___layers___0___blocks___0___drop_path2(x_21);  x_21 = None
    x_22 = x_15 + getattr_getattr_l__mod___layers___0___blocks___0___drop_path2;  x_15 = getattr_getattr_l__mod___layers___0___blocks___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_23 = x_22.reshape(8, 56, 56, 128);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___0___blocks___1___norm1 = self.getattr_getattr_L__mod___layers___0___blocks___1___norm1(x_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_4 = torch.roll(getattr_getattr_l__mod___layers___0___blocks___1___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___0___blocks___1___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_5 = torch.nn.functional.pad(shifted_x_4, (0, 0, 0, 0, 0, 0));  shifted_x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_24 = shifted_x_5.view(8, 8, 7, 8, 7, 128);  shifted_x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_5 = x_24.permute(0, 1, 3, 2, 4, 5);  x_24 = None
    contiguous_4 = permute_5.contiguous();  permute_5 = None
    x_windows_2 = contiguous_4.view(-1, 7, 7, 128);  contiguous_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_3 = x_windows_2.view(-1, 49, 128);  x_windows_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___0___blocks___1___attn_mask = self.getattr_getattr_L__mod___layers___0___blocks___1___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___0___blocks___1___attn_qkv = self.getattr_getattr_L__mod___layers___0___blocks___1___attn_qkv(x_windows_3);  x_windows_3 = None
    reshape_4 = getattr_getattr_l__mod___layers___0___blocks___1___attn_qkv.reshape(512, 49, 3, 4, -1);  getattr_getattr_l__mod___layers___0___blocks___1___attn_qkv = None
    qkv_1 = reshape_4.permute(2, 0, 3, 1, 4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_2 = unbind_1[0]
    k_1 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_3 = q_2 * 0.1767766952966369;  q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_2 = k_1.transpose(-2, -1);  k_1 = None
    attn_4 = q_3 @ transpose_2;  q_3 = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___0___blocks___1___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___0___blocks___1___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___0___blocks___1___attn_relative_position_index = self.getattr_getattr_L__mod___layers___0___blocks___1___attn_relative_position_index
    view_11 = getattr_getattr_l__mod___layers___0___blocks___1___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___0___blocks___1___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_8 = getattr_getattr_l__mod___layers___0___blocks___1___attn_relative_position_bias_table[view_11];  getattr_getattr_l__mod___layers___0___blocks___1___attn_relative_position_bias_table = view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_2 = getitem_8.view(49, 49, -1);  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_7 = relative_position_bias_2.permute(2, 0, 1);  relative_position_bias_2 = None
    relative_position_bias_3 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_1 = relative_position_bias_3.unsqueeze(0);  relative_position_bias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_5 = attn_4 + unsqueeze_1;  attn_4 = unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_13 = attn_5.view(-1, 64, 4, 49, 49);  attn_5 = None
    unsqueeze_2 = getattr_getattr_l__mod___layers___0___blocks___1___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___0___blocks___1___attn_mask = None
    unsqueeze_3 = unsqueeze_2.unsqueeze(0);  unsqueeze_2 = None
    attn_6 = view_13 + unsqueeze_3;  view_13 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_7 = attn_6.view(-1, 4, 49, 49);  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_8 = self.getattr_getattr_L__mod___layers___0___blocks___1___attn_softmax(attn_7);  attn_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_9 = self.getattr_getattr_L__mod___layers___0___blocks___1___attn_attn_drop(attn_8);  attn_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_25 = attn_9 @ v_1;  attn_9 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_3 = x_25.transpose(1, 2);  x_25 = None
    x_26 = transpose_3.reshape(512, 49, -1);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_27 = self.getattr_getattr_L__mod___layers___0___blocks___1___attn_proj(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_2 = self.getattr_getattr_L__mod___layers___0___blocks___1___attn_proj_drop(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_3 = attn_windows_2.view(-1, 7, 7, 128);  attn_windows_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_29 = attn_windows_3.view(-1, 8, 8, 7, 7, 128);  attn_windows_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_8 = x_29.permute(0, 1, 3, 2, 4, 5);  x_29 = None
    contiguous_6 = permute_8.contiguous();  permute_8 = None
    shifted_x_6 = contiguous_6.view(-1, 56, 56, 128);  contiguous_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_9 = shifted_x_6[(slice(None, None, None), slice(None, 56, None), slice(None, 56, None), slice(None, None, None))];  shifted_x_6 = None
    shifted_x_7 = getitem_9.contiguous();  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_31 = torch.roll(shifted_x_7, shifts = (3, 3), dims = (1, 2));  shifted_x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_32 = x_23 + x_31;  x_23 = x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_33 = x_32.reshape(8, -1, 128);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___0___blocks___1___norm2 = self.getattr_getattr_L__mod___layers___0___blocks___1___norm2(x_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_34 = self.getattr_getattr_L__mod___layers___0___blocks___1___mlp_fc1(getattr_getattr_l__mod___layers___0___blocks___1___norm2);  getattr_getattr_l__mod___layers___0___blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_35 = self.getattr_getattr_L__mod___layers___0___blocks___1___mlp_act(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_36 = self.getattr_getattr_L__mod___layers___0___blocks___1___mlp_drop1(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_37 = self.getattr_getattr_L__mod___layers___0___blocks___1___mlp_norm(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_38 = self.getattr_getattr_L__mod___layers___0___blocks___1___mlp_fc2(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_39 = self.getattr_getattr_L__mod___layers___0___blocks___1___mlp_drop2(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_40 = x_33 + x_39;  x_33 = x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_42 = x_40.reshape(8, 56, 56, 128);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    reshape_8 = x_42.reshape(8, 28, 2, 28, 2, 128);  x_42 = None
    permute_9 = reshape_8.permute(0, 1, 3, 4, 2, 5);  reshape_8 = None
    x_43 = permute_9.flatten(3);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    x_44 = self.getattr_L__mod___layers___1___downsample_norm(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    x_46 = self.getattr_L__mod___layers___1___downsample_reduction(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_8 = self.getattr_getattr_L__mod___layers___1___blocks___0___norm1(x_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_9 = torch.nn.functional.pad(shifted_x_8, (0, 0, 0, 0, 0, 0));  shifted_x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_47 = shifted_x_9.view(8, 4, 7, 4, 7, 256);  shifted_x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_10 = x_47.permute(0, 1, 3, 2, 4, 5);  x_47 = None
    contiguous_8 = permute_10.contiguous();  permute_10 = None
    x_windows_4 = contiguous_8.view(-1, 7, 7, 256);  contiguous_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_5 = x_windows_4.view(-1, 49, 256);  x_windows_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___1___blocks___0___attn_qkv = self.getattr_getattr_L__mod___layers___1___blocks___0___attn_qkv(x_windows_5);  x_windows_5 = None
    reshape_9 = getattr_getattr_l__mod___layers___1___blocks___0___attn_qkv.reshape(128, 49, 3, 8, -1);  getattr_getattr_l__mod___layers___1___blocks___0___attn_qkv = None
    qkv_2 = reshape_9.permute(2, 0, 3, 1, 4);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_4 = unbind_2[0]
    k_2 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_5 = q_4 * 0.1767766952966369;  q_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_4 = k_2.transpose(-2, -1);  k_2 = None
    attn_10 = q_5 @ transpose_4;  q_5 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___1___blocks___0___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___1___blocks___0___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___1___blocks___0___attn_relative_position_index = self.getattr_getattr_L__mod___layers___1___blocks___0___attn_relative_position_index
    view_21 = getattr_getattr_l__mod___layers___1___blocks___0___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___1___blocks___0___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_13 = getattr_getattr_l__mod___layers___1___blocks___0___attn_relative_position_bias_table[view_21];  getattr_getattr_l__mod___layers___1___blocks___0___attn_relative_position_bias_table = view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_4 = getitem_13.view(49, 49, -1);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_12 = relative_position_bias_4.permute(2, 0, 1);  relative_position_bias_4 = None
    relative_position_bias_5 = permute_12.contiguous();  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_4 = relative_position_bias_5.unsqueeze(0);  relative_position_bias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_11 = attn_10 + unsqueeze_4;  attn_10 = unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_12 = self.getattr_getattr_L__mod___layers___1___blocks___0___attn_softmax(attn_11);  attn_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_13 = self.getattr_getattr_L__mod___layers___1___blocks___0___attn_attn_drop(attn_12);  attn_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_48 = attn_13 @ v_2;  attn_13 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_5 = x_48.transpose(1, 2);  x_48 = None
    x_49 = transpose_5.reshape(128, 49, -1);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_50 = self.getattr_getattr_L__mod___layers___1___blocks___0___attn_proj(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_4 = self.getattr_getattr_L__mod___layers___1___blocks___0___attn_proj_drop(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_5 = attn_windows_4.view(-1, 7, 7, 256);  attn_windows_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_52 = attn_windows_5.view(-1, 4, 4, 7, 7, 256);  attn_windows_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_13 = x_52.permute(0, 1, 3, 2, 4, 5);  x_52 = None
    contiguous_10 = permute_13.contiguous();  permute_13 = None
    shifted_x_10 = contiguous_10.view(-1, 28, 28, 256);  contiguous_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_14 = shifted_x_10[(slice(None, None, None), slice(None, 28, None), slice(None, 28, None), slice(None, None, None))];  shifted_x_10 = None
    x_54 = getitem_14.contiguous();  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_55 = x_46 + x_54;  x_46 = x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_56 = x_55.reshape(8, -1, 256);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___1___blocks___0___norm2 = self.getattr_getattr_L__mod___layers___1___blocks___0___norm2(x_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_57 = self.getattr_getattr_L__mod___layers___1___blocks___0___mlp_fc1(getattr_getattr_l__mod___layers___1___blocks___0___norm2);  getattr_getattr_l__mod___layers___1___blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_58 = self.getattr_getattr_L__mod___layers___1___blocks___0___mlp_act(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_59 = self.getattr_getattr_L__mod___layers___1___blocks___0___mlp_drop1(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_60 = self.getattr_getattr_L__mod___layers___1___blocks___0___mlp_norm(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_61 = self.getattr_getattr_L__mod___layers___1___blocks___0___mlp_fc2(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_62 = self.getattr_getattr_L__mod___layers___1___blocks___0___mlp_drop2(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_63 = x_56 + x_62;  x_56 = x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_64 = x_63.reshape(8, 28, 28, 256);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___1___blocks___1___norm1 = self.getattr_getattr_L__mod___layers___1___blocks___1___norm1(x_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_12 = torch.roll(getattr_getattr_l__mod___layers___1___blocks___1___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___1___blocks___1___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_13 = torch.nn.functional.pad(shifted_x_12, (0, 0, 0, 0, 0, 0));  shifted_x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_65 = shifted_x_13.view(8, 4, 7, 4, 7, 256);  shifted_x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_14 = x_65.permute(0, 1, 3, 2, 4, 5);  x_65 = None
    contiguous_12 = permute_14.contiguous();  permute_14 = None
    x_windows_6 = contiguous_12.view(-1, 7, 7, 256);  contiguous_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_7 = x_windows_6.view(-1, 49, 256);  x_windows_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___1___blocks___1___attn_mask = self.getattr_getattr_L__mod___layers___1___blocks___1___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___1___blocks___1___attn_qkv = self.getattr_getattr_L__mod___layers___1___blocks___1___attn_qkv(x_windows_7);  x_windows_7 = None
    reshape_13 = getattr_getattr_l__mod___layers___1___blocks___1___attn_qkv.reshape(128, 49, 3, 8, -1);  getattr_getattr_l__mod___layers___1___blocks___1___attn_qkv = None
    qkv_3 = reshape_13.permute(2, 0, 3, 1, 4);  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_6 = unbind_3[0]
    k_3 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_7 = q_6 * 0.1767766952966369;  q_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_6 = k_3.transpose(-2, -1);  k_3 = None
    attn_14 = q_7 @ transpose_6;  q_7 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___1___blocks___1___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___1___blocks___1___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___1___blocks___1___attn_relative_position_index = self.getattr_getattr_L__mod___layers___1___blocks___1___attn_relative_position_index
    view_29 = getattr_getattr_l__mod___layers___1___blocks___1___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___1___blocks___1___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_18 = getattr_getattr_l__mod___layers___1___blocks___1___attn_relative_position_bias_table[view_29];  getattr_getattr_l__mod___layers___1___blocks___1___attn_relative_position_bias_table = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_6 = getitem_18.view(49, 49, -1);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_16 = relative_position_bias_6.permute(2, 0, 1);  relative_position_bias_6 = None
    relative_position_bias_7 = permute_16.contiguous();  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_5 = relative_position_bias_7.unsqueeze(0);  relative_position_bias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_15 = attn_14 + unsqueeze_5;  attn_14 = unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_31 = attn_15.view(-1, 16, 8, 49, 49);  attn_15 = None
    unsqueeze_6 = getattr_getattr_l__mod___layers___1___blocks___1___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___1___blocks___1___attn_mask = None
    unsqueeze_7 = unsqueeze_6.unsqueeze(0);  unsqueeze_6 = None
    attn_16 = view_31 + unsqueeze_7;  view_31 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_17 = attn_16.view(-1, 8, 49, 49);  attn_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_18 = self.getattr_getattr_L__mod___layers___1___blocks___1___attn_softmax(attn_17);  attn_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_19 = self.getattr_getattr_L__mod___layers___1___blocks___1___attn_attn_drop(attn_18);  attn_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_66 = attn_19 @ v_3;  attn_19 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_7 = x_66.transpose(1, 2);  x_66 = None
    x_67 = transpose_7.reshape(128, 49, -1);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_68 = self.getattr_getattr_L__mod___layers___1___blocks___1___attn_proj(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_6 = self.getattr_getattr_L__mod___layers___1___blocks___1___attn_proj_drop(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_7 = attn_windows_6.view(-1, 7, 7, 256);  attn_windows_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_70 = attn_windows_7.view(-1, 4, 4, 7, 7, 256);  attn_windows_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_17 = x_70.permute(0, 1, 3, 2, 4, 5);  x_70 = None
    contiguous_14 = permute_17.contiguous();  permute_17 = None
    shifted_x_14 = contiguous_14.view(-1, 28, 28, 256);  contiguous_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_19 = shifted_x_14[(slice(None, None, None), slice(None, 28, None), slice(None, 28, None), slice(None, None, None))];  shifted_x_14 = None
    shifted_x_15 = getitem_19.contiguous();  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_72 = torch.roll(shifted_x_15, shifts = (3, 3), dims = (1, 2));  shifted_x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_73 = x_64 + x_72;  x_64 = x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_74 = x_73.reshape(8, -1, 256);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___1___blocks___1___norm2 = self.getattr_getattr_L__mod___layers___1___blocks___1___norm2(x_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_75 = self.getattr_getattr_L__mod___layers___1___blocks___1___mlp_fc1(getattr_getattr_l__mod___layers___1___blocks___1___norm2);  getattr_getattr_l__mod___layers___1___blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_76 = self.getattr_getattr_L__mod___layers___1___blocks___1___mlp_act(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_77 = self.getattr_getattr_L__mod___layers___1___blocks___1___mlp_drop1(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_78 = self.getattr_getattr_L__mod___layers___1___blocks___1___mlp_norm(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_79 = self.getattr_getattr_L__mod___layers___1___blocks___1___mlp_fc2(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_80 = self.getattr_getattr_L__mod___layers___1___blocks___1___mlp_drop2(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_81 = x_74 + x_80;  x_74 = x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_83 = x_81.reshape(8, 28, 28, 256);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    reshape_17 = x_83.reshape(8, 14, 2, 14, 2, 256);  x_83 = None
    permute_18 = reshape_17.permute(0, 1, 3, 4, 2, 5);  reshape_17 = None
    x_84 = permute_18.flatten(3);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    x_85 = self.getattr_L__mod___layers___2___downsample_norm(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    x_87 = self.getattr_L__mod___layers___2___downsample_reduction(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_16 = self.getattr_getattr_L__mod___layers___2___blocks___0___norm1(x_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_17 = torch.nn.functional.pad(shifted_x_16, (0, 0, 0, 0, 0, 0));  shifted_x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_88 = shifted_x_17.view(8, 2, 7, 2, 7, 512);  shifted_x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_19 = x_88.permute(0, 1, 3, 2, 4, 5);  x_88 = None
    contiguous_16 = permute_19.contiguous();  permute_19 = None
    x_windows_8 = contiguous_16.view(-1, 7, 7, 512);  contiguous_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_9 = x_windows_8.view(-1, 49, 512);  x_windows_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___0___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___0___attn_qkv(x_windows_9);  x_windows_9 = None
    reshape_18 = getattr_getattr_l__mod___layers___2___blocks___0___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___0___attn_qkv = None
    qkv_4 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_8 = unbind_4[0]
    k_4 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_9 = q_8 * 0.1767766952966369;  q_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_8 = k_4.transpose(-2, -1);  k_4 = None
    attn_20 = q_9 @ transpose_8;  q_9 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___0___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___0___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___0___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___0___attn_relative_position_index
    view_39 = getattr_getattr_l__mod___layers___2___blocks___0___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___0___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_23 = getattr_getattr_l__mod___layers___2___blocks___0___attn_relative_position_bias_table[view_39];  getattr_getattr_l__mod___layers___2___blocks___0___attn_relative_position_bias_table = view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_8 = getitem_23.view(49, 49, -1);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_21 = relative_position_bias_8.permute(2, 0, 1);  relative_position_bias_8 = None
    relative_position_bias_9 = permute_21.contiguous();  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_8 = relative_position_bias_9.unsqueeze(0);  relative_position_bias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_21 = attn_20 + unsqueeze_8;  attn_20 = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_22 = self.getattr_getattr_L__mod___layers___2___blocks___0___attn_softmax(attn_21);  attn_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_23 = self.getattr_getattr_L__mod___layers___2___blocks___0___attn_attn_drop(attn_22);  attn_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_89 = attn_23 @ v_4;  attn_23 = v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_9 = x_89.transpose(1, 2);  x_89 = None
    x_90 = transpose_9.reshape(32, 49, -1);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_91 = self.getattr_getattr_L__mod___layers___2___blocks___0___attn_proj(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_8 = self.getattr_getattr_L__mod___layers___2___blocks___0___attn_proj_drop(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_9 = attn_windows_8.view(-1, 7, 7, 512);  attn_windows_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_93 = attn_windows_9.view(-1, 2, 2, 7, 7, 512);  attn_windows_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_22 = x_93.permute(0, 1, 3, 2, 4, 5);  x_93 = None
    contiguous_18 = permute_22.contiguous();  permute_22 = None
    shifted_x_18 = contiguous_18.view(-1, 14, 14, 512);  contiguous_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_24 = shifted_x_18[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_18 = None
    x_95 = getitem_24.contiguous();  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_96 = x_87 + x_95;  x_87 = x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_97 = x_96.reshape(8, -1, 512);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___0___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___0___norm2(x_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_98 = self.getattr_getattr_L__mod___layers___2___blocks___0___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___0___norm2);  getattr_getattr_l__mod___layers___2___blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_99 = self.getattr_getattr_L__mod___layers___2___blocks___0___mlp_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_100 = self.getattr_getattr_L__mod___layers___2___blocks___0___mlp_drop1(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_101 = self.getattr_getattr_L__mod___layers___2___blocks___0___mlp_norm(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_102 = self.getattr_getattr_L__mod___layers___2___blocks___0___mlp_fc2(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_103 = self.getattr_getattr_L__mod___layers___2___blocks___0___mlp_drop2(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_104 = x_97 + x_103;  x_97 = x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_105 = x_104.reshape(8, 14, 14, 512);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___1___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___1___norm1(x_105)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_20 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___1___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___1___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_21 = torch.nn.functional.pad(shifted_x_20, (0, 0, 0, 0, 0, 0));  shifted_x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_106 = shifted_x_21.view(8, 2, 7, 2, 7, 512);  shifted_x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_23 = x_106.permute(0, 1, 3, 2, 4, 5);  x_106 = None
    contiguous_20 = permute_23.contiguous();  permute_23 = None
    x_windows_10 = contiguous_20.view(-1, 7, 7, 512);  contiguous_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_11 = x_windows_10.view(-1, 49, 512);  x_windows_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___1___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___1___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___1___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___1___attn_qkv(x_windows_11);  x_windows_11 = None
    reshape_22 = getattr_getattr_l__mod___layers___2___blocks___1___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___1___attn_qkv = None
    qkv_5 = reshape_22.permute(2, 0, 3, 1, 4);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_10 = unbind_5[0]
    k_5 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_11 = q_10 * 0.1767766952966369;  q_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_10 = k_5.transpose(-2, -1);  k_5 = None
    attn_24 = q_11 @ transpose_10;  q_11 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___1___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___1___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___1___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___1___attn_relative_position_index
    view_47 = getattr_getattr_l__mod___layers___2___blocks___1___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___1___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_28 = getattr_getattr_l__mod___layers___2___blocks___1___attn_relative_position_bias_table[view_47];  getattr_getattr_l__mod___layers___2___blocks___1___attn_relative_position_bias_table = view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_10 = getitem_28.view(49, 49, -1);  getitem_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_25 = relative_position_bias_10.permute(2, 0, 1);  relative_position_bias_10 = None
    relative_position_bias_11 = permute_25.contiguous();  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_9 = relative_position_bias_11.unsqueeze(0);  relative_position_bias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_25 = attn_24 + unsqueeze_9;  attn_24 = unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_49 = attn_25.view(-1, 4, 16, 49, 49);  attn_25 = None
    unsqueeze_10 = getattr_getattr_l__mod___layers___2___blocks___1___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___1___attn_mask = None
    unsqueeze_11 = unsqueeze_10.unsqueeze(0);  unsqueeze_10 = None
    attn_26 = view_49 + unsqueeze_11;  view_49 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_27 = attn_26.view(-1, 16, 49, 49);  attn_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_28 = self.getattr_getattr_L__mod___layers___2___blocks___1___attn_softmax(attn_27);  attn_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_29 = self.getattr_getattr_L__mod___layers___2___blocks___1___attn_attn_drop(attn_28);  attn_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_107 = attn_29 @ v_5;  attn_29 = v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_11 = x_107.transpose(1, 2);  x_107 = None
    x_108 = transpose_11.reshape(32, 49, -1);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_109 = self.getattr_getattr_L__mod___layers___2___blocks___1___attn_proj(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_10 = self.getattr_getattr_L__mod___layers___2___blocks___1___attn_proj_drop(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_11 = attn_windows_10.view(-1, 7, 7, 512);  attn_windows_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_111 = attn_windows_11.view(-1, 2, 2, 7, 7, 512);  attn_windows_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_26 = x_111.permute(0, 1, 3, 2, 4, 5);  x_111 = None
    contiguous_22 = permute_26.contiguous();  permute_26 = None
    shifted_x_22 = contiguous_22.view(-1, 14, 14, 512);  contiguous_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_29 = shifted_x_22[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_22 = None
    shifted_x_23 = getitem_29.contiguous();  getitem_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_113 = torch.roll(shifted_x_23, shifts = (3, 3), dims = (1, 2));  shifted_x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_114 = x_105 + x_113;  x_105 = x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_115 = x_114.reshape(8, -1, 512);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___1___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___1___norm2(x_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_116 = self.getattr_getattr_L__mod___layers___2___blocks___1___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___1___norm2);  getattr_getattr_l__mod___layers___2___blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_117 = self.getattr_getattr_L__mod___layers___2___blocks___1___mlp_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_118 = self.getattr_getattr_L__mod___layers___2___blocks___1___mlp_drop1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_119 = self.getattr_getattr_L__mod___layers___2___blocks___1___mlp_norm(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_120 = self.getattr_getattr_L__mod___layers___2___blocks___1___mlp_fc2(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_121 = self.getattr_getattr_L__mod___layers___2___blocks___1___mlp_drop2(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_122 = x_115 + x_121;  x_115 = x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_123 = x_122.reshape(8, 14, 14, 512);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_24 = self.getattr_getattr_L__mod___layers___2___blocks___2___norm1(x_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_25 = torch.nn.functional.pad(shifted_x_24, (0, 0, 0, 0, 0, 0));  shifted_x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_124 = shifted_x_25.view(8, 2, 7, 2, 7, 512);  shifted_x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_27 = x_124.permute(0, 1, 3, 2, 4, 5);  x_124 = None
    contiguous_24 = permute_27.contiguous();  permute_27 = None
    x_windows_12 = contiguous_24.view(-1, 7, 7, 512);  contiguous_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_13 = x_windows_12.view(-1, 49, 512);  x_windows_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___2___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___2___attn_qkv(x_windows_13);  x_windows_13 = None
    reshape_26 = getattr_getattr_l__mod___layers___2___blocks___2___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___2___attn_qkv = None
    qkv_6 = reshape_26.permute(2, 0, 3, 1, 4);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_12 = unbind_6[0]
    k_6 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_13 = q_12 * 0.1767766952966369;  q_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_12 = k_6.transpose(-2, -1);  k_6 = None
    attn_30 = q_13 @ transpose_12;  q_13 = transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___2___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___2___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___2___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___2___attn_relative_position_index
    view_57 = getattr_getattr_l__mod___layers___2___blocks___2___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___2___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_33 = getattr_getattr_l__mod___layers___2___blocks___2___attn_relative_position_bias_table[view_57];  getattr_getattr_l__mod___layers___2___blocks___2___attn_relative_position_bias_table = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_12 = getitem_33.view(49, 49, -1);  getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_29 = relative_position_bias_12.permute(2, 0, 1);  relative_position_bias_12 = None
    relative_position_bias_13 = permute_29.contiguous();  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_12 = relative_position_bias_13.unsqueeze(0);  relative_position_bias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_31 = attn_30 + unsqueeze_12;  attn_30 = unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_32 = self.getattr_getattr_L__mod___layers___2___blocks___2___attn_softmax(attn_31);  attn_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_33 = self.getattr_getattr_L__mod___layers___2___blocks___2___attn_attn_drop(attn_32);  attn_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_125 = attn_33 @ v_6;  attn_33 = v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_13 = x_125.transpose(1, 2);  x_125 = None
    x_126 = transpose_13.reshape(32, 49, -1);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_127 = self.getattr_getattr_L__mod___layers___2___blocks___2___attn_proj(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_12 = self.getattr_getattr_L__mod___layers___2___blocks___2___attn_proj_drop(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_13 = attn_windows_12.view(-1, 7, 7, 512);  attn_windows_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_129 = attn_windows_13.view(-1, 2, 2, 7, 7, 512);  attn_windows_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_30 = x_129.permute(0, 1, 3, 2, 4, 5);  x_129 = None
    contiguous_26 = permute_30.contiguous();  permute_30 = None
    shifted_x_26 = contiguous_26.view(-1, 14, 14, 512);  contiguous_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_34 = shifted_x_26[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_26 = None
    x_131 = getitem_34.contiguous();  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_132 = x_123 + x_131;  x_123 = x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_133 = x_132.reshape(8, -1, 512);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___2___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___2___norm2(x_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_134 = self.getattr_getattr_L__mod___layers___2___blocks___2___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___2___norm2);  getattr_getattr_l__mod___layers___2___blocks___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_135 = self.getattr_getattr_L__mod___layers___2___blocks___2___mlp_act(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_136 = self.getattr_getattr_L__mod___layers___2___blocks___2___mlp_drop1(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_137 = self.getattr_getattr_L__mod___layers___2___blocks___2___mlp_norm(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_138 = self.getattr_getattr_L__mod___layers___2___blocks___2___mlp_fc2(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_139 = self.getattr_getattr_L__mod___layers___2___blocks___2___mlp_drop2(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_140 = x_133 + x_139;  x_133 = x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_141 = x_140.reshape(8, 14, 14, 512);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___3___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___3___norm1(x_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_28 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___3___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___3___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_29 = torch.nn.functional.pad(shifted_x_28, (0, 0, 0, 0, 0, 0));  shifted_x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_142 = shifted_x_29.view(8, 2, 7, 2, 7, 512);  shifted_x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_31 = x_142.permute(0, 1, 3, 2, 4, 5);  x_142 = None
    contiguous_28 = permute_31.contiguous();  permute_31 = None
    x_windows_14 = contiguous_28.view(-1, 7, 7, 512);  contiguous_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_15 = x_windows_14.view(-1, 49, 512);  x_windows_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___3___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___3___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___3___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___3___attn_qkv(x_windows_15);  x_windows_15 = None
    reshape_30 = getattr_getattr_l__mod___layers___2___blocks___3___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___3___attn_qkv = None
    qkv_7 = reshape_30.permute(2, 0, 3, 1, 4);  reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_14 = unbind_7[0]
    k_7 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_15 = q_14 * 0.1767766952966369;  q_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_14 = k_7.transpose(-2, -1);  k_7 = None
    attn_34 = q_15 @ transpose_14;  q_15 = transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___3___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___3___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___3___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___3___attn_relative_position_index
    view_65 = getattr_getattr_l__mod___layers___2___blocks___3___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___3___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_38 = getattr_getattr_l__mod___layers___2___blocks___3___attn_relative_position_bias_table[view_65];  getattr_getattr_l__mod___layers___2___blocks___3___attn_relative_position_bias_table = view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_14 = getitem_38.view(49, 49, -1);  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_33 = relative_position_bias_14.permute(2, 0, 1);  relative_position_bias_14 = None
    relative_position_bias_15 = permute_33.contiguous();  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_13 = relative_position_bias_15.unsqueeze(0);  relative_position_bias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_35 = attn_34 + unsqueeze_13;  attn_34 = unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_67 = attn_35.view(-1, 4, 16, 49, 49);  attn_35 = None
    unsqueeze_14 = getattr_getattr_l__mod___layers___2___blocks___3___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___3___attn_mask = None
    unsqueeze_15 = unsqueeze_14.unsqueeze(0);  unsqueeze_14 = None
    attn_36 = view_67 + unsqueeze_15;  view_67 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_37 = attn_36.view(-1, 16, 49, 49);  attn_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_38 = self.getattr_getattr_L__mod___layers___2___blocks___3___attn_softmax(attn_37);  attn_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_39 = self.getattr_getattr_L__mod___layers___2___blocks___3___attn_attn_drop(attn_38);  attn_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_143 = attn_39 @ v_7;  attn_39 = v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_15 = x_143.transpose(1, 2);  x_143 = None
    x_144 = transpose_15.reshape(32, 49, -1);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_145 = self.getattr_getattr_L__mod___layers___2___blocks___3___attn_proj(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_14 = self.getattr_getattr_L__mod___layers___2___blocks___3___attn_proj_drop(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_15 = attn_windows_14.view(-1, 7, 7, 512);  attn_windows_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_147 = attn_windows_15.view(-1, 2, 2, 7, 7, 512);  attn_windows_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_34 = x_147.permute(0, 1, 3, 2, 4, 5);  x_147 = None
    contiguous_30 = permute_34.contiguous();  permute_34 = None
    shifted_x_30 = contiguous_30.view(-1, 14, 14, 512);  contiguous_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_39 = shifted_x_30[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_30 = None
    shifted_x_31 = getitem_39.contiguous();  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_149 = torch.roll(shifted_x_31, shifts = (3, 3), dims = (1, 2));  shifted_x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_150 = x_141 + x_149;  x_141 = x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_151 = x_150.reshape(8, -1, 512);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___3___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___3___norm2(x_151)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_152 = self.getattr_getattr_L__mod___layers___2___blocks___3___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___3___norm2);  getattr_getattr_l__mod___layers___2___blocks___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_153 = self.getattr_getattr_L__mod___layers___2___blocks___3___mlp_act(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_154 = self.getattr_getattr_L__mod___layers___2___blocks___3___mlp_drop1(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_155 = self.getattr_getattr_L__mod___layers___2___blocks___3___mlp_norm(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_156 = self.getattr_getattr_L__mod___layers___2___blocks___3___mlp_fc2(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_157 = self.getattr_getattr_L__mod___layers___2___blocks___3___mlp_drop2(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_158 = x_151 + x_157;  x_151 = x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_159 = x_158.reshape(8, 14, 14, 512);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_32 = self.getattr_getattr_L__mod___layers___2___blocks___4___norm1(x_159)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_33 = torch.nn.functional.pad(shifted_x_32, (0, 0, 0, 0, 0, 0));  shifted_x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_160 = shifted_x_33.view(8, 2, 7, 2, 7, 512);  shifted_x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_35 = x_160.permute(0, 1, 3, 2, 4, 5);  x_160 = None
    contiguous_32 = permute_35.contiguous();  permute_35 = None
    x_windows_16 = contiguous_32.view(-1, 7, 7, 512);  contiguous_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_17 = x_windows_16.view(-1, 49, 512);  x_windows_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___4___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___4___attn_qkv(x_windows_17);  x_windows_17 = None
    reshape_34 = getattr_getattr_l__mod___layers___2___blocks___4___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___4___attn_qkv = None
    qkv_8 = reshape_34.permute(2, 0, 3, 1, 4);  reshape_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_8 = qkv_8.unbind(0);  qkv_8 = None
    q_16 = unbind_8[0]
    k_8 = unbind_8[1]
    v_8 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_17 = q_16 * 0.1767766952966369;  q_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_16 = k_8.transpose(-2, -1);  k_8 = None
    attn_40 = q_17 @ transpose_16;  q_17 = transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___4___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___4___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___4___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___4___attn_relative_position_index
    view_75 = getattr_getattr_l__mod___layers___2___blocks___4___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___4___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_43 = getattr_getattr_l__mod___layers___2___blocks___4___attn_relative_position_bias_table[view_75];  getattr_getattr_l__mod___layers___2___blocks___4___attn_relative_position_bias_table = view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_16 = getitem_43.view(49, 49, -1);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_37 = relative_position_bias_16.permute(2, 0, 1);  relative_position_bias_16 = None
    relative_position_bias_17 = permute_37.contiguous();  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_16 = relative_position_bias_17.unsqueeze(0);  relative_position_bias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_41 = attn_40 + unsqueeze_16;  attn_40 = unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_42 = self.getattr_getattr_L__mod___layers___2___blocks___4___attn_softmax(attn_41);  attn_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_43 = self.getattr_getattr_L__mod___layers___2___blocks___4___attn_attn_drop(attn_42);  attn_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_161 = attn_43 @ v_8;  attn_43 = v_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_17 = x_161.transpose(1, 2);  x_161 = None
    x_162 = transpose_17.reshape(32, 49, -1);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_163 = self.getattr_getattr_L__mod___layers___2___blocks___4___attn_proj(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_16 = self.getattr_getattr_L__mod___layers___2___blocks___4___attn_proj_drop(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_17 = attn_windows_16.view(-1, 7, 7, 512);  attn_windows_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_165 = attn_windows_17.view(-1, 2, 2, 7, 7, 512);  attn_windows_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_38 = x_165.permute(0, 1, 3, 2, 4, 5);  x_165 = None
    contiguous_34 = permute_38.contiguous();  permute_38 = None
    shifted_x_34 = contiguous_34.view(-1, 14, 14, 512);  contiguous_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_44 = shifted_x_34[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_34 = None
    x_167 = getitem_44.contiguous();  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_168 = x_159 + x_167;  x_159 = x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_169 = x_168.reshape(8, -1, 512);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___4___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___4___norm2(x_169)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_170 = self.getattr_getattr_L__mod___layers___2___blocks___4___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___4___norm2);  getattr_getattr_l__mod___layers___2___blocks___4___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_171 = self.getattr_getattr_L__mod___layers___2___blocks___4___mlp_act(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_172 = self.getattr_getattr_L__mod___layers___2___blocks___4___mlp_drop1(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_173 = self.getattr_getattr_L__mod___layers___2___blocks___4___mlp_norm(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_174 = self.getattr_getattr_L__mod___layers___2___blocks___4___mlp_fc2(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_175 = self.getattr_getattr_L__mod___layers___2___blocks___4___mlp_drop2(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_176 = x_169 + x_175;  x_169 = x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_177 = x_176.reshape(8, 14, 14, 512);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___5___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___5___norm1(x_177)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_36 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___5___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___5___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_37 = torch.nn.functional.pad(shifted_x_36, (0, 0, 0, 0, 0, 0));  shifted_x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_178 = shifted_x_37.view(8, 2, 7, 2, 7, 512);  shifted_x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_39 = x_178.permute(0, 1, 3, 2, 4, 5);  x_178 = None
    contiguous_36 = permute_39.contiguous();  permute_39 = None
    x_windows_18 = contiguous_36.view(-1, 7, 7, 512);  contiguous_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_19 = x_windows_18.view(-1, 49, 512);  x_windows_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___5___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___5___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___5___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___5___attn_qkv(x_windows_19);  x_windows_19 = None
    reshape_38 = getattr_getattr_l__mod___layers___2___blocks___5___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___5___attn_qkv = None
    qkv_9 = reshape_38.permute(2, 0, 3, 1, 4);  reshape_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_9 = qkv_9.unbind(0);  qkv_9 = None
    q_18 = unbind_9[0]
    k_9 = unbind_9[1]
    v_9 = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_19 = q_18 * 0.1767766952966369;  q_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_18 = k_9.transpose(-2, -1);  k_9 = None
    attn_44 = q_19 @ transpose_18;  q_19 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___5___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___5___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___5___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___5___attn_relative_position_index
    view_83 = getattr_getattr_l__mod___layers___2___blocks___5___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___5___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_48 = getattr_getattr_l__mod___layers___2___blocks___5___attn_relative_position_bias_table[view_83];  getattr_getattr_l__mod___layers___2___blocks___5___attn_relative_position_bias_table = view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_18 = getitem_48.view(49, 49, -1);  getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_41 = relative_position_bias_18.permute(2, 0, 1);  relative_position_bias_18 = None
    relative_position_bias_19 = permute_41.contiguous();  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_17 = relative_position_bias_19.unsqueeze(0);  relative_position_bias_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_45 = attn_44 + unsqueeze_17;  attn_44 = unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_85 = attn_45.view(-1, 4, 16, 49, 49);  attn_45 = None
    unsqueeze_18 = getattr_getattr_l__mod___layers___2___blocks___5___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___5___attn_mask = None
    unsqueeze_19 = unsqueeze_18.unsqueeze(0);  unsqueeze_18 = None
    attn_46 = view_85 + unsqueeze_19;  view_85 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_47 = attn_46.view(-1, 16, 49, 49);  attn_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_48 = self.getattr_getattr_L__mod___layers___2___blocks___5___attn_softmax(attn_47);  attn_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_49 = self.getattr_getattr_L__mod___layers___2___blocks___5___attn_attn_drop(attn_48);  attn_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_179 = attn_49 @ v_9;  attn_49 = v_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_19 = x_179.transpose(1, 2);  x_179 = None
    x_180 = transpose_19.reshape(32, 49, -1);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_181 = self.getattr_getattr_L__mod___layers___2___blocks___5___attn_proj(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_18 = self.getattr_getattr_L__mod___layers___2___blocks___5___attn_proj_drop(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_19 = attn_windows_18.view(-1, 7, 7, 512);  attn_windows_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_183 = attn_windows_19.view(-1, 2, 2, 7, 7, 512);  attn_windows_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_42 = x_183.permute(0, 1, 3, 2, 4, 5);  x_183 = None
    contiguous_38 = permute_42.contiguous();  permute_42 = None
    shifted_x_38 = contiguous_38.view(-1, 14, 14, 512);  contiguous_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_49 = shifted_x_38[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_38 = None
    shifted_x_39 = getitem_49.contiguous();  getitem_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_185 = torch.roll(shifted_x_39, shifts = (3, 3), dims = (1, 2));  shifted_x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_186 = x_177 + x_185;  x_177 = x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_187 = x_186.reshape(8, -1, 512);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___5___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___5___norm2(x_187)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_188 = self.getattr_getattr_L__mod___layers___2___blocks___5___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___5___norm2);  getattr_getattr_l__mod___layers___2___blocks___5___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_189 = self.getattr_getattr_L__mod___layers___2___blocks___5___mlp_act(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_190 = self.getattr_getattr_L__mod___layers___2___blocks___5___mlp_drop1(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_191 = self.getattr_getattr_L__mod___layers___2___blocks___5___mlp_norm(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_192 = self.getattr_getattr_L__mod___layers___2___blocks___5___mlp_fc2(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_193 = self.getattr_getattr_L__mod___layers___2___blocks___5___mlp_drop2(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_194 = x_187 + x_193;  x_187 = x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_195 = x_194.reshape(8, 14, 14, 512);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_40 = self.getattr_getattr_L__mod___layers___2___blocks___6___norm1(x_195)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_41 = torch.nn.functional.pad(shifted_x_40, (0, 0, 0, 0, 0, 0));  shifted_x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_196 = shifted_x_41.view(8, 2, 7, 2, 7, 512);  shifted_x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_43 = x_196.permute(0, 1, 3, 2, 4, 5);  x_196 = None
    contiguous_40 = permute_43.contiguous();  permute_43 = None
    x_windows_20 = contiguous_40.view(-1, 7, 7, 512);  contiguous_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_21 = x_windows_20.view(-1, 49, 512);  x_windows_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___6___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___6___attn_qkv(x_windows_21);  x_windows_21 = None
    reshape_42 = getattr_getattr_l__mod___layers___2___blocks___6___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___6___attn_qkv = None
    qkv_10 = reshape_42.permute(2, 0, 3, 1, 4);  reshape_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_10 = qkv_10.unbind(0);  qkv_10 = None
    q_20 = unbind_10[0]
    k_10 = unbind_10[1]
    v_10 = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_21 = q_20 * 0.1767766952966369;  q_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_20 = k_10.transpose(-2, -1);  k_10 = None
    attn_50 = q_21 @ transpose_20;  q_21 = transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___6___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___6___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___6___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___6___attn_relative_position_index
    view_93 = getattr_getattr_l__mod___layers___2___blocks___6___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___6___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_53 = getattr_getattr_l__mod___layers___2___blocks___6___attn_relative_position_bias_table[view_93];  getattr_getattr_l__mod___layers___2___blocks___6___attn_relative_position_bias_table = view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_20 = getitem_53.view(49, 49, -1);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_45 = relative_position_bias_20.permute(2, 0, 1);  relative_position_bias_20 = None
    relative_position_bias_21 = permute_45.contiguous();  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_20 = relative_position_bias_21.unsqueeze(0);  relative_position_bias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_51 = attn_50 + unsqueeze_20;  attn_50 = unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_52 = self.getattr_getattr_L__mod___layers___2___blocks___6___attn_softmax(attn_51);  attn_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_53 = self.getattr_getattr_L__mod___layers___2___blocks___6___attn_attn_drop(attn_52);  attn_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_197 = attn_53 @ v_10;  attn_53 = v_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_21 = x_197.transpose(1, 2);  x_197 = None
    x_198 = transpose_21.reshape(32, 49, -1);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_199 = self.getattr_getattr_L__mod___layers___2___blocks___6___attn_proj(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_20 = self.getattr_getattr_L__mod___layers___2___blocks___6___attn_proj_drop(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_21 = attn_windows_20.view(-1, 7, 7, 512);  attn_windows_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_201 = attn_windows_21.view(-1, 2, 2, 7, 7, 512);  attn_windows_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_46 = x_201.permute(0, 1, 3, 2, 4, 5);  x_201 = None
    contiguous_42 = permute_46.contiguous();  permute_46 = None
    shifted_x_42 = contiguous_42.view(-1, 14, 14, 512);  contiguous_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_54 = shifted_x_42[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_42 = None
    x_203 = getitem_54.contiguous();  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_204 = x_195 + x_203;  x_195 = x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_205 = x_204.reshape(8, -1, 512);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___6___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___6___norm2(x_205)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_206 = self.getattr_getattr_L__mod___layers___2___blocks___6___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___6___norm2);  getattr_getattr_l__mod___layers___2___blocks___6___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_207 = self.getattr_getattr_L__mod___layers___2___blocks___6___mlp_act(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_208 = self.getattr_getattr_L__mod___layers___2___blocks___6___mlp_drop1(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_209 = self.getattr_getattr_L__mod___layers___2___blocks___6___mlp_norm(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_210 = self.getattr_getattr_L__mod___layers___2___blocks___6___mlp_fc2(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_211 = self.getattr_getattr_L__mod___layers___2___blocks___6___mlp_drop2(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_212 = x_205 + x_211;  x_205 = x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_213 = x_212.reshape(8, 14, 14, 512);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___7___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___7___norm1(x_213)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_44 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___7___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___7___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_45 = torch.nn.functional.pad(shifted_x_44, (0, 0, 0, 0, 0, 0));  shifted_x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_214 = shifted_x_45.view(8, 2, 7, 2, 7, 512);  shifted_x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_47 = x_214.permute(0, 1, 3, 2, 4, 5);  x_214 = None
    contiguous_44 = permute_47.contiguous();  permute_47 = None
    x_windows_22 = contiguous_44.view(-1, 7, 7, 512);  contiguous_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_23 = x_windows_22.view(-1, 49, 512);  x_windows_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___7___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___7___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___7___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___7___attn_qkv(x_windows_23);  x_windows_23 = None
    reshape_46 = getattr_getattr_l__mod___layers___2___blocks___7___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___7___attn_qkv = None
    qkv_11 = reshape_46.permute(2, 0, 3, 1, 4);  reshape_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_11 = qkv_11.unbind(0);  qkv_11 = None
    q_22 = unbind_11[0]
    k_11 = unbind_11[1]
    v_11 = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_23 = q_22 * 0.1767766952966369;  q_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_22 = k_11.transpose(-2, -1);  k_11 = None
    attn_54 = q_23 @ transpose_22;  q_23 = transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___7___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___7___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___7___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___7___attn_relative_position_index
    view_101 = getattr_getattr_l__mod___layers___2___blocks___7___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___7___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_58 = getattr_getattr_l__mod___layers___2___blocks___7___attn_relative_position_bias_table[view_101];  getattr_getattr_l__mod___layers___2___blocks___7___attn_relative_position_bias_table = view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_22 = getitem_58.view(49, 49, -1);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_49 = relative_position_bias_22.permute(2, 0, 1);  relative_position_bias_22 = None
    relative_position_bias_23 = permute_49.contiguous();  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_21 = relative_position_bias_23.unsqueeze(0);  relative_position_bias_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_55 = attn_54 + unsqueeze_21;  attn_54 = unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_103 = attn_55.view(-1, 4, 16, 49, 49);  attn_55 = None
    unsqueeze_22 = getattr_getattr_l__mod___layers___2___blocks___7___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___7___attn_mask = None
    unsqueeze_23 = unsqueeze_22.unsqueeze(0);  unsqueeze_22 = None
    attn_56 = view_103 + unsqueeze_23;  view_103 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_57 = attn_56.view(-1, 16, 49, 49);  attn_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_58 = self.getattr_getattr_L__mod___layers___2___blocks___7___attn_softmax(attn_57);  attn_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_59 = self.getattr_getattr_L__mod___layers___2___blocks___7___attn_attn_drop(attn_58);  attn_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_215 = attn_59 @ v_11;  attn_59 = v_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_23 = x_215.transpose(1, 2);  x_215 = None
    x_216 = transpose_23.reshape(32, 49, -1);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_217 = self.getattr_getattr_L__mod___layers___2___blocks___7___attn_proj(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_22 = self.getattr_getattr_L__mod___layers___2___blocks___7___attn_proj_drop(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_23 = attn_windows_22.view(-1, 7, 7, 512);  attn_windows_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_219 = attn_windows_23.view(-1, 2, 2, 7, 7, 512);  attn_windows_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_50 = x_219.permute(0, 1, 3, 2, 4, 5);  x_219 = None
    contiguous_46 = permute_50.contiguous();  permute_50 = None
    shifted_x_46 = contiguous_46.view(-1, 14, 14, 512);  contiguous_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_59 = shifted_x_46[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_46 = None
    shifted_x_47 = getitem_59.contiguous();  getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_221 = torch.roll(shifted_x_47, shifts = (3, 3), dims = (1, 2));  shifted_x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_222 = x_213 + x_221;  x_213 = x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_223 = x_222.reshape(8, -1, 512);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___7___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___7___norm2(x_223)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_224 = self.getattr_getattr_L__mod___layers___2___blocks___7___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___7___norm2);  getattr_getattr_l__mod___layers___2___blocks___7___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_225 = self.getattr_getattr_L__mod___layers___2___blocks___7___mlp_act(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_226 = self.getattr_getattr_L__mod___layers___2___blocks___7___mlp_drop1(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_227 = self.getattr_getattr_L__mod___layers___2___blocks___7___mlp_norm(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_228 = self.getattr_getattr_L__mod___layers___2___blocks___7___mlp_fc2(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_229 = self.getattr_getattr_L__mod___layers___2___blocks___7___mlp_drop2(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_230 = x_223 + x_229;  x_223 = x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_231 = x_230.reshape(8, 14, 14, 512);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_48 = self.getattr_getattr_L__mod___layers___2___blocks___8___norm1(x_231)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_49 = torch.nn.functional.pad(shifted_x_48, (0, 0, 0, 0, 0, 0));  shifted_x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_232 = shifted_x_49.view(8, 2, 7, 2, 7, 512);  shifted_x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_51 = x_232.permute(0, 1, 3, 2, 4, 5);  x_232 = None
    contiguous_48 = permute_51.contiguous();  permute_51 = None
    x_windows_24 = contiguous_48.view(-1, 7, 7, 512);  contiguous_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_25 = x_windows_24.view(-1, 49, 512);  x_windows_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___8___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___8___attn_qkv(x_windows_25);  x_windows_25 = None
    reshape_50 = getattr_getattr_l__mod___layers___2___blocks___8___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___8___attn_qkv = None
    qkv_12 = reshape_50.permute(2, 0, 3, 1, 4);  reshape_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_12 = qkv_12.unbind(0);  qkv_12 = None
    q_24 = unbind_12[0]
    k_12 = unbind_12[1]
    v_12 = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_25 = q_24 * 0.1767766952966369;  q_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_24 = k_12.transpose(-2, -1);  k_12 = None
    attn_60 = q_25 @ transpose_24;  q_25 = transpose_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___8___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___8___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___8___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___8___attn_relative_position_index
    view_111 = getattr_getattr_l__mod___layers___2___blocks___8___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___8___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_63 = getattr_getattr_l__mod___layers___2___blocks___8___attn_relative_position_bias_table[view_111];  getattr_getattr_l__mod___layers___2___blocks___8___attn_relative_position_bias_table = view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_24 = getitem_63.view(49, 49, -1);  getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_53 = relative_position_bias_24.permute(2, 0, 1);  relative_position_bias_24 = None
    relative_position_bias_25 = permute_53.contiguous();  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_24 = relative_position_bias_25.unsqueeze(0);  relative_position_bias_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_61 = attn_60 + unsqueeze_24;  attn_60 = unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_62 = self.getattr_getattr_L__mod___layers___2___blocks___8___attn_softmax(attn_61);  attn_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_63 = self.getattr_getattr_L__mod___layers___2___blocks___8___attn_attn_drop(attn_62);  attn_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_233 = attn_63 @ v_12;  attn_63 = v_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_25 = x_233.transpose(1, 2);  x_233 = None
    x_234 = transpose_25.reshape(32, 49, -1);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_235 = self.getattr_getattr_L__mod___layers___2___blocks___8___attn_proj(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_24 = self.getattr_getattr_L__mod___layers___2___blocks___8___attn_proj_drop(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_25 = attn_windows_24.view(-1, 7, 7, 512);  attn_windows_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_237 = attn_windows_25.view(-1, 2, 2, 7, 7, 512);  attn_windows_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_54 = x_237.permute(0, 1, 3, 2, 4, 5);  x_237 = None
    contiguous_50 = permute_54.contiguous();  permute_54 = None
    shifted_x_50 = contiguous_50.view(-1, 14, 14, 512);  contiguous_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_64 = shifted_x_50[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_50 = None
    x_239 = getitem_64.contiguous();  getitem_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_240 = x_231 + x_239;  x_231 = x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_241 = x_240.reshape(8, -1, 512);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___8___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___8___norm2(x_241)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_242 = self.getattr_getattr_L__mod___layers___2___blocks___8___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___8___norm2);  getattr_getattr_l__mod___layers___2___blocks___8___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_243 = self.getattr_getattr_L__mod___layers___2___blocks___8___mlp_act(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_244 = self.getattr_getattr_L__mod___layers___2___blocks___8___mlp_drop1(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_245 = self.getattr_getattr_L__mod___layers___2___blocks___8___mlp_norm(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_246 = self.getattr_getattr_L__mod___layers___2___blocks___8___mlp_fc2(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_247 = self.getattr_getattr_L__mod___layers___2___blocks___8___mlp_drop2(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_248 = x_241 + x_247;  x_241 = x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_249 = x_248.reshape(8, 14, 14, 512);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___9___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___9___norm1(x_249)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_52 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___9___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___9___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_53 = torch.nn.functional.pad(shifted_x_52, (0, 0, 0, 0, 0, 0));  shifted_x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_250 = shifted_x_53.view(8, 2, 7, 2, 7, 512);  shifted_x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_55 = x_250.permute(0, 1, 3, 2, 4, 5);  x_250 = None
    contiguous_52 = permute_55.contiguous();  permute_55 = None
    x_windows_26 = contiguous_52.view(-1, 7, 7, 512);  contiguous_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_27 = x_windows_26.view(-1, 49, 512);  x_windows_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___9___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___9___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___9___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___9___attn_qkv(x_windows_27);  x_windows_27 = None
    reshape_54 = getattr_getattr_l__mod___layers___2___blocks___9___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___9___attn_qkv = None
    qkv_13 = reshape_54.permute(2, 0, 3, 1, 4);  reshape_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_13 = qkv_13.unbind(0);  qkv_13 = None
    q_26 = unbind_13[0]
    k_13 = unbind_13[1]
    v_13 = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_27 = q_26 * 0.1767766952966369;  q_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_26 = k_13.transpose(-2, -1);  k_13 = None
    attn_64 = q_27 @ transpose_26;  q_27 = transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___9___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___9___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___9___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___9___attn_relative_position_index
    view_119 = getattr_getattr_l__mod___layers___2___blocks___9___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___9___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_68 = getattr_getattr_l__mod___layers___2___blocks___9___attn_relative_position_bias_table[view_119];  getattr_getattr_l__mod___layers___2___blocks___9___attn_relative_position_bias_table = view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_26 = getitem_68.view(49, 49, -1);  getitem_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_57 = relative_position_bias_26.permute(2, 0, 1);  relative_position_bias_26 = None
    relative_position_bias_27 = permute_57.contiguous();  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_25 = relative_position_bias_27.unsqueeze(0);  relative_position_bias_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_65 = attn_64 + unsqueeze_25;  attn_64 = unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_121 = attn_65.view(-1, 4, 16, 49, 49);  attn_65 = None
    unsqueeze_26 = getattr_getattr_l__mod___layers___2___blocks___9___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___9___attn_mask = None
    unsqueeze_27 = unsqueeze_26.unsqueeze(0);  unsqueeze_26 = None
    attn_66 = view_121 + unsqueeze_27;  view_121 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_67 = attn_66.view(-1, 16, 49, 49);  attn_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_68 = self.getattr_getattr_L__mod___layers___2___blocks___9___attn_softmax(attn_67);  attn_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_69 = self.getattr_getattr_L__mod___layers___2___blocks___9___attn_attn_drop(attn_68);  attn_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_251 = attn_69 @ v_13;  attn_69 = v_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_27 = x_251.transpose(1, 2);  x_251 = None
    x_252 = transpose_27.reshape(32, 49, -1);  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_253 = self.getattr_getattr_L__mod___layers___2___blocks___9___attn_proj(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_26 = self.getattr_getattr_L__mod___layers___2___blocks___9___attn_proj_drop(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_27 = attn_windows_26.view(-1, 7, 7, 512);  attn_windows_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_255 = attn_windows_27.view(-1, 2, 2, 7, 7, 512);  attn_windows_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_58 = x_255.permute(0, 1, 3, 2, 4, 5);  x_255 = None
    contiguous_54 = permute_58.contiguous();  permute_58 = None
    shifted_x_54 = contiguous_54.view(-1, 14, 14, 512);  contiguous_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_69 = shifted_x_54[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_54 = None
    shifted_x_55 = getitem_69.contiguous();  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_257 = torch.roll(shifted_x_55, shifts = (3, 3), dims = (1, 2));  shifted_x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_258 = x_249 + x_257;  x_249 = x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_259 = x_258.reshape(8, -1, 512);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___9___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___9___norm2(x_259)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_260 = self.getattr_getattr_L__mod___layers___2___blocks___9___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___9___norm2);  getattr_getattr_l__mod___layers___2___blocks___9___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_261 = self.getattr_getattr_L__mod___layers___2___blocks___9___mlp_act(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_262 = self.getattr_getattr_L__mod___layers___2___blocks___9___mlp_drop1(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_263 = self.getattr_getattr_L__mod___layers___2___blocks___9___mlp_norm(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_264 = self.getattr_getattr_L__mod___layers___2___blocks___9___mlp_fc2(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_265 = self.getattr_getattr_L__mod___layers___2___blocks___9___mlp_drop2(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_266 = x_259 + x_265;  x_259 = x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_267 = x_266.reshape(8, 14, 14, 512);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_56 = self.getattr_getattr_L__mod___layers___2___blocks___10___norm1(x_267)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_57 = torch.nn.functional.pad(shifted_x_56, (0, 0, 0, 0, 0, 0));  shifted_x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_268 = shifted_x_57.view(8, 2, 7, 2, 7, 512);  shifted_x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_59 = x_268.permute(0, 1, 3, 2, 4, 5);  x_268 = None
    contiguous_56 = permute_59.contiguous();  permute_59 = None
    x_windows_28 = contiguous_56.view(-1, 7, 7, 512);  contiguous_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_29 = x_windows_28.view(-1, 49, 512);  x_windows_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___10___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___10___attn_qkv(x_windows_29);  x_windows_29 = None
    reshape_58 = getattr_getattr_l__mod___layers___2___blocks___10___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___10___attn_qkv = None
    qkv_14 = reshape_58.permute(2, 0, 3, 1, 4);  reshape_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_14 = qkv_14.unbind(0);  qkv_14 = None
    q_28 = unbind_14[0]
    k_14 = unbind_14[1]
    v_14 = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_29 = q_28 * 0.1767766952966369;  q_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_28 = k_14.transpose(-2, -1);  k_14 = None
    attn_70 = q_29 @ transpose_28;  q_29 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___10___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___10___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___10___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___10___attn_relative_position_index
    view_129 = getattr_getattr_l__mod___layers___2___blocks___10___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___10___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_73 = getattr_getattr_l__mod___layers___2___blocks___10___attn_relative_position_bias_table[view_129];  getattr_getattr_l__mod___layers___2___blocks___10___attn_relative_position_bias_table = view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_28 = getitem_73.view(49, 49, -1);  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_61 = relative_position_bias_28.permute(2, 0, 1);  relative_position_bias_28 = None
    relative_position_bias_29 = permute_61.contiguous();  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_28 = relative_position_bias_29.unsqueeze(0);  relative_position_bias_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_71 = attn_70 + unsqueeze_28;  attn_70 = unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_72 = self.getattr_getattr_L__mod___layers___2___blocks___10___attn_softmax(attn_71);  attn_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_73 = self.getattr_getattr_L__mod___layers___2___blocks___10___attn_attn_drop(attn_72);  attn_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_269 = attn_73 @ v_14;  attn_73 = v_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_29 = x_269.transpose(1, 2);  x_269 = None
    x_270 = transpose_29.reshape(32, 49, -1);  transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_271 = self.getattr_getattr_L__mod___layers___2___blocks___10___attn_proj(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_28 = self.getattr_getattr_L__mod___layers___2___blocks___10___attn_proj_drop(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_29 = attn_windows_28.view(-1, 7, 7, 512);  attn_windows_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_273 = attn_windows_29.view(-1, 2, 2, 7, 7, 512);  attn_windows_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_62 = x_273.permute(0, 1, 3, 2, 4, 5);  x_273 = None
    contiguous_58 = permute_62.contiguous();  permute_62 = None
    shifted_x_58 = contiguous_58.view(-1, 14, 14, 512);  contiguous_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_74 = shifted_x_58[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_58 = None
    x_275 = getitem_74.contiguous();  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_276 = x_267 + x_275;  x_267 = x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_277 = x_276.reshape(8, -1, 512);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___10___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___10___norm2(x_277)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_278 = self.getattr_getattr_L__mod___layers___2___blocks___10___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___10___norm2);  getattr_getattr_l__mod___layers___2___blocks___10___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_279 = self.getattr_getattr_L__mod___layers___2___blocks___10___mlp_act(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_280 = self.getattr_getattr_L__mod___layers___2___blocks___10___mlp_drop1(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_281 = self.getattr_getattr_L__mod___layers___2___blocks___10___mlp_norm(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_282 = self.getattr_getattr_L__mod___layers___2___blocks___10___mlp_fc2(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_283 = self.getattr_getattr_L__mod___layers___2___blocks___10___mlp_drop2(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_284 = x_277 + x_283;  x_277 = x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_285 = x_284.reshape(8, 14, 14, 512);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___11___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___11___norm1(x_285)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_60 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___11___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___11___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_61 = torch.nn.functional.pad(shifted_x_60, (0, 0, 0, 0, 0, 0));  shifted_x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_286 = shifted_x_61.view(8, 2, 7, 2, 7, 512);  shifted_x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_63 = x_286.permute(0, 1, 3, 2, 4, 5);  x_286 = None
    contiguous_60 = permute_63.contiguous();  permute_63 = None
    x_windows_30 = contiguous_60.view(-1, 7, 7, 512);  contiguous_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_31 = x_windows_30.view(-1, 49, 512);  x_windows_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___11___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___11___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___11___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___11___attn_qkv(x_windows_31);  x_windows_31 = None
    reshape_62 = getattr_getattr_l__mod___layers___2___blocks___11___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___11___attn_qkv = None
    qkv_15 = reshape_62.permute(2, 0, 3, 1, 4);  reshape_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_15 = qkv_15.unbind(0);  qkv_15 = None
    q_30 = unbind_15[0]
    k_15 = unbind_15[1]
    v_15 = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_31 = q_30 * 0.1767766952966369;  q_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_30 = k_15.transpose(-2, -1);  k_15 = None
    attn_74 = q_31 @ transpose_30;  q_31 = transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___11___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___11___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___11___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___11___attn_relative_position_index
    view_137 = getattr_getattr_l__mod___layers___2___blocks___11___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___11___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_78 = getattr_getattr_l__mod___layers___2___blocks___11___attn_relative_position_bias_table[view_137];  getattr_getattr_l__mod___layers___2___blocks___11___attn_relative_position_bias_table = view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_30 = getitem_78.view(49, 49, -1);  getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_65 = relative_position_bias_30.permute(2, 0, 1);  relative_position_bias_30 = None
    relative_position_bias_31 = permute_65.contiguous();  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_29 = relative_position_bias_31.unsqueeze(0);  relative_position_bias_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_75 = attn_74 + unsqueeze_29;  attn_74 = unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_139 = attn_75.view(-1, 4, 16, 49, 49);  attn_75 = None
    unsqueeze_30 = getattr_getattr_l__mod___layers___2___blocks___11___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___11___attn_mask = None
    unsqueeze_31 = unsqueeze_30.unsqueeze(0);  unsqueeze_30 = None
    attn_76 = view_139 + unsqueeze_31;  view_139 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_77 = attn_76.view(-1, 16, 49, 49);  attn_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_78 = self.getattr_getattr_L__mod___layers___2___blocks___11___attn_softmax(attn_77);  attn_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_79 = self.getattr_getattr_L__mod___layers___2___blocks___11___attn_attn_drop(attn_78);  attn_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_287 = attn_79 @ v_15;  attn_79 = v_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_31 = x_287.transpose(1, 2);  x_287 = None
    x_288 = transpose_31.reshape(32, 49, -1);  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_289 = self.getattr_getattr_L__mod___layers___2___blocks___11___attn_proj(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_30 = self.getattr_getattr_L__mod___layers___2___blocks___11___attn_proj_drop(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_31 = attn_windows_30.view(-1, 7, 7, 512);  attn_windows_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_291 = attn_windows_31.view(-1, 2, 2, 7, 7, 512);  attn_windows_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_66 = x_291.permute(0, 1, 3, 2, 4, 5);  x_291 = None
    contiguous_62 = permute_66.contiguous();  permute_66 = None
    shifted_x_62 = contiguous_62.view(-1, 14, 14, 512);  contiguous_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_79 = shifted_x_62[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_62 = None
    shifted_x_63 = getitem_79.contiguous();  getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_293 = torch.roll(shifted_x_63, shifts = (3, 3), dims = (1, 2));  shifted_x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_294 = x_285 + x_293;  x_285 = x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_295 = x_294.reshape(8, -1, 512);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___11___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___11___norm2(x_295)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_296 = self.getattr_getattr_L__mod___layers___2___blocks___11___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___11___norm2);  getattr_getattr_l__mod___layers___2___blocks___11___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_297 = self.getattr_getattr_L__mod___layers___2___blocks___11___mlp_act(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_298 = self.getattr_getattr_L__mod___layers___2___blocks___11___mlp_drop1(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_299 = self.getattr_getattr_L__mod___layers___2___blocks___11___mlp_norm(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_300 = self.getattr_getattr_L__mod___layers___2___blocks___11___mlp_fc2(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_301 = self.getattr_getattr_L__mod___layers___2___blocks___11___mlp_drop2(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_302 = x_295 + x_301;  x_295 = x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_303 = x_302.reshape(8, 14, 14, 512);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_64 = self.getattr_getattr_L__mod___layers___2___blocks___12___norm1(x_303)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_65 = torch.nn.functional.pad(shifted_x_64, (0, 0, 0, 0, 0, 0));  shifted_x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_304 = shifted_x_65.view(8, 2, 7, 2, 7, 512);  shifted_x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_67 = x_304.permute(0, 1, 3, 2, 4, 5);  x_304 = None
    contiguous_64 = permute_67.contiguous();  permute_67 = None
    x_windows_32 = contiguous_64.view(-1, 7, 7, 512);  contiguous_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_33 = x_windows_32.view(-1, 49, 512);  x_windows_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___12___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___12___attn_qkv(x_windows_33);  x_windows_33 = None
    reshape_66 = getattr_getattr_l__mod___layers___2___blocks___12___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___12___attn_qkv = None
    qkv_16 = reshape_66.permute(2, 0, 3, 1, 4);  reshape_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_16 = qkv_16.unbind(0);  qkv_16 = None
    q_32 = unbind_16[0]
    k_16 = unbind_16[1]
    v_16 = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_33 = q_32 * 0.1767766952966369;  q_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_32 = k_16.transpose(-2, -1);  k_16 = None
    attn_80 = q_33 @ transpose_32;  q_33 = transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___12___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___12___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___12___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___12___attn_relative_position_index
    view_147 = getattr_getattr_l__mod___layers___2___blocks___12___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___12___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_83 = getattr_getattr_l__mod___layers___2___blocks___12___attn_relative_position_bias_table[view_147];  getattr_getattr_l__mod___layers___2___blocks___12___attn_relative_position_bias_table = view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_32 = getitem_83.view(49, 49, -1);  getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_69 = relative_position_bias_32.permute(2, 0, 1);  relative_position_bias_32 = None
    relative_position_bias_33 = permute_69.contiguous();  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_32 = relative_position_bias_33.unsqueeze(0);  relative_position_bias_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_81 = attn_80 + unsqueeze_32;  attn_80 = unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_82 = self.getattr_getattr_L__mod___layers___2___blocks___12___attn_softmax(attn_81);  attn_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_83 = self.getattr_getattr_L__mod___layers___2___blocks___12___attn_attn_drop(attn_82);  attn_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_305 = attn_83 @ v_16;  attn_83 = v_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_33 = x_305.transpose(1, 2);  x_305 = None
    x_306 = transpose_33.reshape(32, 49, -1);  transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_307 = self.getattr_getattr_L__mod___layers___2___blocks___12___attn_proj(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_32 = self.getattr_getattr_L__mod___layers___2___blocks___12___attn_proj_drop(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_33 = attn_windows_32.view(-1, 7, 7, 512);  attn_windows_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_309 = attn_windows_33.view(-1, 2, 2, 7, 7, 512);  attn_windows_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_70 = x_309.permute(0, 1, 3, 2, 4, 5);  x_309 = None
    contiguous_66 = permute_70.contiguous();  permute_70 = None
    shifted_x_66 = contiguous_66.view(-1, 14, 14, 512);  contiguous_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_84 = shifted_x_66[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_66 = None
    x_311 = getitem_84.contiguous();  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_312 = x_303 + x_311;  x_303 = x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_313 = x_312.reshape(8, -1, 512);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___12___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___12___norm2(x_313)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_314 = self.getattr_getattr_L__mod___layers___2___blocks___12___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___12___norm2);  getattr_getattr_l__mod___layers___2___blocks___12___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_315 = self.getattr_getattr_L__mod___layers___2___blocks___12___mlp_act(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_316 = self.getattr_getattr_L__mod___layers___2___blocks___12___mlp_drop1(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_317 = self.getattr_getattr_L__mod___layers___2___blocks___12___mlp_norm(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_318 = self.getattr_getattr_L__mod___layers___2___blocks___12___mlp_fc2(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_319 = self.getattr_getattr_L__mod___layers___2___blocks___12___mlp_drop2(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_320 = x_313 + x_319;  x_313 = x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_321 = x_320.reshape(8, 14, 14, 512);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___13___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___13___norm1(x_321)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_68 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___13___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___13___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_69 = torch.nn.functional.pad(shifted_x_68, (0, 0, 0, 0, 0, 0));  shifted_x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_322 = shifted_x_69.view(8, 2, 7, 2, 7, 512);  shifted_x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_71 = x_322.permute(0, 1, 3, 2, 4, 5);  x_322 = None
    contiguous_68 = permute_71.contiguous();  permute_71 = None
    x_windows_34 = contiguous_68.view(-1, 7, 7, 512);  contiguous_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_35 = x_windows_34.view(-1, 49, 512);  x_windows_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___13___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___13___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___13___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___13___attn_qkv(x_windows_35);  x_windows_35 = None
    reshape_70 = getattr_getattr_l__mod___layers___2___blocks___13___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___13___attn_qkv = None
    qkv_17 = reshape_70.permute(2, 0, 3, 1, 4);  reshape_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_17 = qkv_17.unbind(0);  qkv_17 = None
    q_34 = unbind_17[0]
    k_17 = unbind_17[1]
    v_17 = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_35 = q_34 * 0.1767766952966369;  q_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_34 = k_17.transpose(-2, -1);  k_17 = None
    attn_84 = q_35 @ transpose_34;  q_35 = transpose_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___13___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___13___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___13___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___13___attn_relative_position_index
    view_155 = getattr_getattr_l__mod___layers___2___blocks___13___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___13___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_88 = getattr_getattr_l__mod___layers___2___blocks___13___attn_relative_position_bias_table[view_155];  getattr_getattr_l__mod___layers___2___blocks___13___attn_relative_position_bias_table = view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_34 = getitem_88.view(49, 49, -1);  getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_73 = relative_position_bias_34.permute(2, 0, 1);  relative_position_bias_34 = None
    relative_position_bias_35 = permute_73.contiguous();  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_33 = relative_position_bias_35.unsqueeze(0);  relative_position_bias_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_85 = attn_84 + unsqueeze_33;  attn_84 = unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_157 = attn_85.view(-1, 4, 16, 49, 49);  attn_85 = None
    unsqueeze_34 = getattr_getattr_l__mod___layers___2___blocks___13___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___13___attn_mask = None
    unsqueeze_35 = unsqueeze_34.unsqueeze(0);  unsqueeze_34 = None
    attn_86 = view_157 + unsqueeze_35;  view_157 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_87 = attn_86.view(-1, 16, 49, 49);  attn_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_88 = self.getattr_getattr_L__mod___layers___2___blocks___13___attn_softmax(attn_87);  attn_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_89 = self.getattr_getattr_L__mod___layers___2___blocks___13___attn_attn_drop(attn_88);  attn_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_323 = attn_89 @ v_17;  attn_89 = v_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_35 = x_323.transpose(1, 2);  x_323 = None
    x_324 = transpose_35.reshape(32, 49, -1);  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_325 = self.getattr_getattr_L__mod___layers___2___blocks___13___attn_proj(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_34 = self.getattr_getattr_L__mod___layers___2___blocks___13___attn_proj_drop(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_35 = attn_windows_34.view(-1, 7, 7, 512);  attn_windows_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_327 = attn_windows_35.view(-1, 2, 2, 7, 7, 512);  attn_windows_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_74 = x_327.permute(0, 1, 3, 2, 4, 5);  x_327 = None
    contiguous_70 = permute_74.contiguous();  permute_74 = None
    shifted_x_70 = contiguous_70.view(-1, 14, 14, 512);  contiguous_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_89 = shifted_x_70[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_70 = None
    shifted_x_71 = getitem_89.contiguous();  getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_329 = torch.roll(shifted_x_71, shifts = (3, 3), dims = (1, 2));  shifted_x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_330 = x_321 + x_329;  x_321 = x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_331 = x_330.reshape(8, -1, 512);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___13___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___13___norm2(x_331)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_332 = self.getattr_getattr_L__mod___layers___2___blocks___13___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___13___norm2);  getattr_getattr_l__mod___layers___2___blocks___13___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_333 = self.getattr_getattr_L__mod___layers___2___blocks___13___mlp_act(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_334 = self.getattr_getattr_L__mod___layers___2___blocks___13___mlp_drop1(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_335 = self.getattr_getattr_L__mod___layers___2___blocks___13___mlp_norm(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_336 = self.getattr_getattr_L__mod___layers___2___blocks___13___mlp_fc2(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_337 = self.getattr_getattr_L__mod___layers___2___blocks___13___mlp_drop2(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_338 = x_331 + x_337;  x_331 = x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_339 = x_338.reshape(8, 14, 14, 512);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_72 = self.getattr_getattr_L__mod___layers___2___blocks___14___norm1(x_339)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_73 = torch.nn.functional.pad(shifted_x_72, (0, 0, 0, 0, 0, 0));  shifted_x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_340 = shifted_x_73.view(8, 2, 7, 2, 7, 512);  shifted_x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_75 = x_340.permute(0, 1, 3, 2, 4, 5);  x_340 = None
    contiguous_72 = permute_75.contiguous();  permute_75 = None
    x_windows_36 = contiguous_72.view(-1, 7, 7, 512);  contiguous_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_37 = x_windows_36.view(-1, 49, 512);  x_windows_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___14___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___14___attn_qkv(x_windows_37);  x_windows_37 = None
    reshape_74 = getattr_getattr_l__mod___layers___2___blocks___14___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___14___attn_qkv = None
    qkv_18 = reshape_74.permute(2, 0, 3, 1, 4);  reshape_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_18 = qkv_18.unbind(0);  qkv_18 = None
    q_36 = unbind_18[0]
    k_18 = unbind_18[1]
    v_18 = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_37 = q_36 * 0.1767766952966369;  q_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_36 = k_18.transpose(-2, -1);  k_18 = None
    attn_90 = q_37 @ transpose_36;  q_37 = transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___14___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___14___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___14___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___14___attn_relative_position_index
    view_165 = getattr_getattr_l__mod___layers___2___blocks___14___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___14___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_93 = getattr_getattr_l__mod___layers___2___blocks___14___attn_relative_position_bias_table[view_165];  getattr_getattr_l__mod___layers___2___blocks___14___attn_relative_position_bias_table = view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_36 = getitem_93.view(49, 49, -1);  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_77 = relative_position_bias_36.permute(2, 0, 1);  relative_position_bias_36 = None
    relative_position_bias_37 = permute_77.contiguous();  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_36 = relative_position_bias_37.unsqueeze(0);  relative_position_bias_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_91 = attn_90 + unsqueeze_36;  attn_90 = unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_92 = self.getattr_getattr_L__mod___layers___2___blocks___14___attn_softmax(attn_91);  attn_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_93 = self.getattr_getattr_L__mod___layers___2___blocks___14___attn_attn_drop(attn_92);  attn_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_341 = attn_93 @ v_18;  attn_93 = v_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_37 = x_341.transpose(1, 2);  x_341 = None
    x_342 = transpose_37.reshape(32, 49, -1);  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_343 = self.getattr_getattr_L__mod___layers___2___blocks___14___attn_proj(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_36 = self.getattr_getattr_L__mod___layers___2___blocks___14___attn_proj_drop(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_37 = attn_windows_36.view(-1, 7, 7, 512);  attn_windows_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_345 = attn_windows_37.view(-1, 2, 2, 7, 7, 512);  attn_windows_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_78 = x_345.permute(0, 1, 3, 2, 4, 5);  x_345 = None
    contiguous_74 = permute_78.contiguous();  permute_78 = None
    shifted_x_74 = contiguous_74.view(-1, 14, 14, 512);  contiguous_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_94 = shifted_x_74[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_74 = None
    x_347 = getitem_94.contiguous();  getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_348 = x_339 + x_347;  x_339 = x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_349 = x_348.reshape(8, -1, 512);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___14___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___14___norm2(x_349)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_350 = self.getattr_getattr_L__mod___layers___2___blocks___14___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___14___norm2);  getattr_getattr_l__mod___layers___2___blocks___14___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_351 = self.getattr_getattr_L__mod___layers___2___blocks___14___mlp_act(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_352 = self.getattr_getattr_L__mod___layers___2___blocks___14___mlp_drop1(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_353 = self.getattr_getattr_L__mod___layers___2___blocks___14___mlp_norm(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_354 = self.getattr_getattr_L__mod___layers___2___blocks___14___mlp_fc2(x_353);  x_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_355 = self.getattr_getattr_L__mod___layers___2___blocks___14___mlp_drop2(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_356 = x_349 + x_355;  x_349 = x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_357 = x_356.reshape(8, 14, 14, 512);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___15___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___15___norm1(x_357)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_76 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___15___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___15___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_77 = torch.nn.functional.pad(shifted_x_76, (0, 0, 0, 0, 0, 0));  shifted_x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_358 = shifted_x_77.view(8, 2, 7, 2, 7, 512);  shifted_x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_79 = x_358.permute(0, 1, 3, 2, 4, 5);  x_358 = None
    contiguous_76 = permute_79.contiguous();  permute_79 = None
    x_windows_38 = contiguous_76.view(-1, 7, 7, 512);  contiguous_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_39 = x_windows_38.view(-1, 49, 512);  x_windows_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___15___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___15___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___15___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___15___attn_qkv(x_windows_39);  x_windows_39 = None
    reshape_78 = getattr_getattr_l__mod___layers___2___blocks___15___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___15___attn_qkv = None
    qkv_19 = reshape_78.permute(2, 0, 3, 1, 4);  reshape_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_19 = qkv_19.unbind(0);  qkv_19 = None
    q_38 = unbind_19[0]
    k_19 = unbind_19[1]
    v_19 = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_39 = q_38 * 0.1767766952966369;  q_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_38 = k_19.transpose(-2, -1);  k_19 = None
    attn_94 = q_39 @ transpose_38;  q_39 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___15___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___15___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___15___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___15___attn_relative_position_index
    view_173 = getattr_getattr_l__mod___layers___2___blocks___15___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___15___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_98 = getattr_getattr_l__mod___layers___2___blocks___15___attn_relative_position_bias_table[view_173];  getattr_getattr_l__mod___layers___2___blocks___15___attn_relative_position_bias_table = view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_38 = getitem_98.view(49, 49, -1);  getitem_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_81 = relative_position_bias_38.permute(2, 0, 1);  relative_position_bias_38 = None
    relative_position_bias_39 = permute_81.contiguous();  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_37 = relative_position_bias_39.unsqueeze(0);  relative_position_bias_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_95 = attn_94 + unsqueeze_37;  attn_94 = unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_175 = attn_95.view(-1, 4, 16, 49, 49);  attn_95 = None
    unsqueeze_38 = getattr_getattr_l__mod___layers___2___blocks___15___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___15___attn_mask = None
    unsqueeze_39 = unsqueeze_38.unsqueeze(0);  unsqueeze_38 = None
    attn_96 = view_175 + unsqueeze_39;  view_175 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_97 = attn_96.view(-1, 16, 49, 49);  attn_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_98 = self.getattr_getattr_L__mod___layers___2___blocks___15___attn_softmax(attn_97);  attn_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_99 = self.getattr_getattr_L__mod___layers___2___blocks___15___attn_attn_drop(attn_98);  attn_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_359 = attn_99 @ v_19;  attn_99 = v_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_39 = x_359.transpose(1, 2);  x_359 = None
    x_360 = transpose_39.reshape(32, 49, -1);  transpose_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_361 = self.getattr_getattr_L__mod___layers___2___blocks___15___attn_proj(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_38 = self.getattr_getattr_L__mod___layers___2___blocks___15___attn_proj_drop(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_39 = attn_windows_38.view(-1, 7, 7, 512);  attn_windows_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_363 = attn_windows_39.view(-1, 2, 2, 7, 7, 512);  attn_windows_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_82 = x_363.permute(0, 1, 3, 2, 4, 5);  x_363 = None
    contiguous_78 = permute_82.contiguous();  permute_82 = None
    shifted_x_78 = contiguous_78.view(-1, 14, 14, 512);  contiguous_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_99 = shifted_x_78[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_78 = None
    shifted_x_79 = getitem_99.contiguous();  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_365 = torch.roll(shifted_x_79, shifts = (3, 3), dims = (1, 2));  shifted_x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_366 = x_357 + x_365;  x_357 = x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_367 = x_366.reshape(8, -1, 512);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___15___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___15___norm2(x_367)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_368 = self.getattr_getattr_L__mod___layers___2___blocks___15___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___15___norm2);  getattr_getattr_l__mod___layers___2___blocks___15___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_369 = self.getattr_getattr_L__mod___layers___2___blocks___15___mlp_act(x_368);  x_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_370 = self.getattr_getattr_L__mod___layers___2___blocks___15___mlp_drop1(x_369);  x_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_371 = self.getattr_getattr_L__mod___layers___2___blocks___15___mlp_norm(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_372 = self.getattr_getattr_L__mod___layers___2___blocks___15___mlp_fc2(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_373 = self.getattr_getattr_L__mod___layers___2___blocks___15___mlp_drop2(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_374 = x_367 + x_373;  x_367 = x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_375 = x_374.reshape(8, 14, 14, 512);  x_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_80 = self.getattr_getattr_L__mod___layers___2___blocks___16___norm1(x_375)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_81 = torch.nn.functional.pad(shifted_x_80, (0, 0, 0, 0, 0, 0));  shifted_x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_376 = shifted_x_81.view(8, 2, 7, 2, 7, 512);  shifted_x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_83 = x_376.permute(0, 1, 3, 2, 4, 5);  x_376 = None
    contiguous_80 = permute_83.contiguous();  permute_83 = None
    x_windows_40 = contiguous_80.view(-1, 7, 7, 512);  contiguous_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_41 = x_windows_40.view(-1, 49, 512);  x_windows_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___16___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___16___attn_qkv(x_windows_41);  x_windows_41 = None
    reshape_82 = getattr_getattr_l__mod___layers___2___blocks___16___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___16___attn_qkv = None
    qkv_20 = reshape_82.permute(2, 0, 3, 1, 4);  reshape_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_20 = qkv_20.unbind(0);  qkv_20 = None
    q_40 = unbind_20[0]
    k_20 = unbind_20[1]
    v_20 = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_41 = q_40 * 0.1767766952966369;  q_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_40 = k_20.transpose(-2, -1);  k_20 = None
    attn_100 = q_41 @ transpose_40;  q_41 = transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___16___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___16___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___16___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___16___attn_relative_position_index
    view_183 = getattr_getattr_l__mod___layers___2___blocks___16___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___16___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_103 = getattr_getattr_l__mod___layers___2___blocks___16___attn_relative_position_bias_table[view_183];  getattr_getattr_l__mod___layers___2___blocks___16___attn_relative_position_bias_table = view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_40 = getitem_103.view(49, 49, -1);  getitem_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_85 = relative_position_bias_40.permute(2, 0, 1);  relative_position_bias_40 = None
    relative_position_bias_41 = permute_85.contiguous();  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_40 = relative_position_bias_41.unsqueeze(0);  relative_position_bias_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_101 = attn_100 + unsqueeze_40;  attn_100 = unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_102 = self.getattr_getattr_L__mod___layers___2___blocks___16___attn_softmax(attn_101);  attn_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_103 = self.getattr_getattr_L__mod___layers___2___blocks___16___attn_attn_drop(attn_102);  attn_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_377 = attn_103 @ v_20;  attn_103 = v_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_41 = x_377.transpose(1, 2);  x_377 = None
    x_378 = transpose_41.reshape(32, 49, -1);  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_379 = self.getattr_getattr_L__mod___layers___2___blocks___16___attn_proj(x_378);  x_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_40 = self.getattr_getattr_L__mod___layers___2___blocks___16___attn_proj_drop(x_379);  x_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_41 = attn_windows_40.view(-1, 7, 7, 512);  attn_windows_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_381 = attn_windows_41.view(-1, 2, 2, 7, 7, 512);  attn_windows_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_86 = x_381.permute(0, 1, 3, 2, 4, 5);  x_381 = None
    contiguous_82 = permute_86.contiguous();  permute_86 = None
    shifted_x_82 = contiguous_82.view(-1, 14, 14, 512);  contiguous_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_104 = shifted_x_82[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_82 = None
    x_383 = getitem_104.contiguous();  getitem_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_384 = x_375 + x_383;  x_375 = x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_385 = x_384.reshape(8, -1, 512);  x_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___16___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___16___norm2(x_385)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_386 = self.getattr_getattr_L__mod___layers___2___blocks___16___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___16___norm2);  getattr_getattr_l__mod___layers___2___blocks___16___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_387 = self.getattr_getattr_L__mod___layers___2___blocks___16___mlp_act(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_388 = self.getattr_getattr_L__mod___layers___2___blocks___16___mlp_drop1(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_389 = self.getattr_getattr_L__mod___layers___2___blocks___16___mlp_norm(x_388);  x_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_390 = self.getattr_getattr_L__mod___layers___2___blocks___16___mlp_fc2(x_389);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_391 = self.getattr_getattr_L__mod___layers___2___blocks___16___mlp_drop2(x_390);  x_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_392 = x_385 + x_391;  x_385 = x_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_393 = x_392.reshape(8, 14, 14, 512);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    getattr_getattr_l__mod___layers___2___blocks___17___norm1 = self.getattr_getattr_L__mod___layers___2___blocks___17___norm1(x_393)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    shifted_x_84 = torch.roll(getattr_getattr_l__mod___layers___2___blocks___17___norm1, shifts = (-3, -3), dims = (1, 2));  getattr_getattr_l__mod___layers___2___blocks___17___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_85 = torch.nn.functional.pad(shifted_x_84, (0, 0, 0, 0, 0, 0));  shifted_x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_394 = shifted_x_85.view(8, 2, 7, 2, 7, 512);  shifted_x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_87 = x_394.permute(0, 1, 3, 2, 4, 5);  x_394 = None
    contiguous_84 = permute_87.contiguous();  permute_87 = None
    x_windows_42 = contiguous_84.view(-1, 7, 7, 512);  contiguous_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_43 = x_windows_42.view(-1, 49, 512);  x_windows_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:307, code: attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    getattr_getattr_l__mod___layers___2___blocks___17___attn_mask = self.getattr_getattr_L__mod___layers___2___blocks___17___attn_mask
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___2___blocks___17___attn_qkv = self.getattr_getattr_L__mod___layers___2___blocks___17___attn_qkv(x_windows_43);  x_windows_43 = None
    reshape_86 = getattr_getattr_l__mod___layers___2___blocks___17___attn_qkv.reshape(32, 49, 3, 16, -1);  getattr_getattr_l__mod___layers___2___blocks___17___attn_qkv = None
    qkv_21 = reshape_86.permute(2, 0, 3, 1, 4);  reshape_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_21 = qkv_21.unbind(0);  qkv_21 = None
    q_42 = unbind_21[0]
    k_21 = unbind_21[1]
    v_21 = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_43 = q_42 * 0.1767766952966369;  q_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_42 = k_21.transpose(-2, -1);  k_21 = None
    attn_104 = q_43 @ transpose_42;  q_43 = transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___2___blocks___17___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___2___blocks___17___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___2___blocks___17___attn_relative_position_index = self.getattr_getattr_L__mod___layers___2___blocks___17___attn_relative_position_index
    view_191 = getattr_getattr_l__mod___layers___2___blocks___17___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___2___blocks___17___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_108 = getattr_getattr_l__mod___layers___2___blocks___17___attn_relative_position_bias_table[view_191];  getattr_getattr_l__mod___layers___2___blocks___17___attn_relative_position_bias_table = view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_42 = getitem_108.view(49, 49, -1);  getitem_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_89 = relative_position_bias_42.permute(2, 0, 1);  relative_position_bias_42 = None
    relative_position_bias_43 = permute_89.contiguous();  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_41 = relative_position_bias_43.unsqueeze(0);  relative_position_bias_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_105 = attn_104 + unsqueeze_41;  attn_104 = unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_193 = attn_105.view(-1, 4, 16, 49, 49);  attn_105 = None
    unsqueeze_42 = getattr_getattr_l__mod___layers___2___blocks___17___attn_mask.unsqueeze(1);  getattr_getattr_l__mod___layers___2___blocks___17___attn_mask = None
    unsqueeze_43 = unsqueeze_42.unsqueeze(0);  unsqueeze_42 = None
    attn_106 = view_193 + unsqueeze_43;  view_193 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    attn_107 = attn_106.view(-1, 16, 49, 49);  attn_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_108 = self.getattr_getattr_L__mod___layers___2___blocks___17___attn_softmax(attn_107);  attn_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_109 = self.getattr_getattr_L__mod___layers___2___blocks___17___attn_attn_drop(attn_108);  attn_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_395 = attn_109 @ v_21;  attn_109 = v_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_43 = x_395.transpose(1, 2);  x_395 = None
    x_396 = transpose_43.reshape(32, 49, -1);  transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_397 = self.getattr_getattr_L__mod___layers___2___blocks___17___attn_proj(x_396);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_42 = self.getattr_getattr_L__mod___layers___2___blocks___17___attn_proj_drop(x_397);  x_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_43 = attn_windows_42.view(-1, 7, 7, 512);  attn_windows_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_399 = attn_windows_43.view(-1, 2, 2, 7, 7, 512);  attn_windows_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_90 = x_399.permute(0, 1, 3, 2, 4, 5);  x_399 = None
    contiguous_86 = permute_90.contiguous();  permute_90 = None
    shifted_x_86 = contiguous_86.view(-1, 14, 14, 512);  contiguous_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_109 = shifted_x_86[(slice(None, None, None), slice(None, 14, None), slice(None, 14, None), slice(None, None, None))];  shifted_x_86 = None
    shifted_x_87 = getitem_109.contiguous();  getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    x_401 = torch.roll(shifted_x_87, shifts = (3, 3), dims = (1, 2));  shifted_x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_402 = x_393 + x_401;  x_393 = x_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_403 = x_402.reshape(8, -1, 512);  x_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___2___blocks___17___norm2 = self.getattr_getattr_L__mod___layers___2___blocks___17___norm2(x_403)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_404 = self.getattr_getattr_L__mod___layers___2___blocks___17___mlp_fc1(getattr_getattr_l__mod___layers___2___blocks___17___norm2);  getattr_getattr_l__mod___layers___2___blocks___17___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_405 = self.getattr_getattr_L__mod___layers___2___blocks___17___mlp_act(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_406 = self.getattr_getattr_L__mod___layers___2___blocks___17___mlp_drop1(x_405);  x_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_407 = self.getattr_getattr_L__mod___layers___2___blocks___17___mlp_norm(x_406);  x_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_408 = self.getattr_getattr_L__mod___layers___2___blocks___17___mlp_fc2(x_407);  x_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_409 = self.getattr_getattr_L__mod___layers___2___blocks___17___mlp_drop2(x_408);  x_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_410 = x_403 + x_409;  x_403 = x_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_412 = x_410.reshape(8, 14, 14, 512);  x_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    reshape_90 = x_412.reshape(8, 7, 2, 7, 2, 512);  x_412 = None
    permute_91 = reshape_90.permute(0, 1, 3, 4, 2, 5);  reshape_90 = None
    x_413 = permute_91.flatten(3);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    x_414 = self.getattr_L__mod___layers___3___downsample_norm(x_413);  x_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    x_416 = self.getattr_L__mod___layers___3___downsample_reduction(x_414);  x_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_88 = self.getattr_getattr_L__mod___layers___3___blocks___0___norm1(x_416)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_89 = torch.nn.functional.pad(shifted_x_88, (0, 0, 0, 0, 0, 0));  shifted_x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_417 = shifted_x_89.view(8, 1, 7, 1, 7, 1024);  shifted_x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_92 = x_417.permute(0, 1, 3, 2, 4, 5);  x_417 = None
    contiguous_88 = permute_92.contiguous();  permute_92 = None
    x_windows_44 = contiguous_88.view(-1, 7, 7, 1024);  contiguous_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_45 = x_windows_44.view(-1, 49, 1024);  x_windows_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___3___blocks___0___attn_qkv = self.getattr_getattr_L__mod___layers___3___blocks___0___attn_qkv(x_windows_45);  x_windows_45 = None
    reshape_91 = getattr_getattr_l__mod___layers___3___blocks___0___attn_qkv.reshape(8, 49, 3, 32, -1);  getattr_getattr_l__mod___layers___3___blocks___0___attn_qkv = None
    qkv_22 = reshape_91.permute(2, 0, 3, 1, 4);  reshape_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_22 = qkv_22.unbind(0);  qkv_22 = None
    q_44 = unbind_22[0]
    k_22 = unbind_22[1]
    v_22 = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_45 = q_44 * 0.1767766952966369;  q_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_44 = k_22.transpose(-2, -1);  k_22 = None
    attn_110 = q_45 @ transpose_44;  q_45 = transpose_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___3___blocks___0___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___3___blocks___0___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___3___blocks___0___attn_relative_position_index = self.getattr_getattr_L__mod___layers___3___blocks___0___attn_relative_position_index
    view_201 = getattr_getattr_l__mod___layers___3___blocks___0___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___3___blocks___0___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_113 = getattr_getattr_l__mod___layers___3___blocks___0___attn_relative_position_bias_table[view_201];  getattr_getattr_l__mod___layers___3___blocks___0___attn_relative_position_bias_table = view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_44 = getitem_113.view(49, 49, -1);  getitem_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_94 = relative_position_bias_44.permute(2, 0, 1);  relative_position_bias_44 = None
    relative_position_bias_45 = permute_94.contiguous();  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_44 = relative_position_bias_45.unsqueeze(0);  relative_position_bias_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_111 = attn_110 + unsqueeze_44;  attn_110 = unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_112 = self.getattr_getattr_L__mod___layers___3___blocks___0___attn_softmax(attn_111);  attn_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_113 = self.getattr_getattr_L__mod___layers___3___blocks___0___attn_attn_drop(attn_112);  attn_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_418 = attn_113 @ v_22;  attn_113 = v_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_45 = x_418.transpose(1, 2);  x_418 = None
    x_419 = transpose_45.reshape(8, 49, -1);  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_420 = self.getattr_getattr_L__mod___layers___3___blocks___0___attn_proj(x_419);  x_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_44 = self.getattr_getattr_L__mod___layers___3___blocks___0___attn_proj_drop(x_420);  x_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_45 = attn_windows_44.view(-1, 7, 7, 1024);  attn_windows_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_422 = attn_windows_45.view(-1, 1, 1, 7, 7, 1024);  attn_windows_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_95 = x_422.permute(0, 1, 3, 2, 4, 5);  x_422 = None
    contiguous_90 = permute_95.contiguous();  permute_95 = None
    shifted_x_90 = contiguous_90.view(-1, 7, 7, 1024);  contiguous_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_114 = shifted_x_90[(slice(None, None, None), slice(None, 7, None), slice(None, 7, None), slice(None, None, None))];  shifted_x_90 = None
    x_424 = getitem_114.contiguous();  getitem_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_425 = x_416 + x_424;  x_416 = x_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_426 = x_425.reshape(8, -1, 1024);  x_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___3___blocks___0___norm2 = self.getattr_getattr_L__mod___layers___3___blocks___0___norm2(x_426)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_427 = self.getattr_getattr_L__mod___layers___3___blocks___0___mlp_fc1(getattr_getattr_l__mod___layers___3___blocks___0___norm2);  getattr_getattr_l__mod___layers___3___blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_428 = self.getattr_getattr_L__mod___layers___3___blocks___0___mlp_act(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_429 = self.getattr_getattr_L__mod___layers___3___blocks___0___mlp_drop1(x_428);  x_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_430 = self.getattr_getattr_L__mod___layers___3___blocks___0___mlp_norm(x_429);  x_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_431 = self.getattr_getattr_L__mod___layers___3___blocks___0___mlp_fc2(x_430);  x_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_432 = self.getattr_getattr_L__mod___layers___3___blocks___0___mlp_drop2(x_431);  x_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_433 = x_426 + x_432;  x_426 = x_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_434 = x_433.reshape(8, 7, 7, 1024);  x_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    shifted_x_92 = self.getattr_getattr_L__mod___layers___3___blocks___1___norm1(x_434)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:299, code: shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x_93 = torch.nn.functional.pad(shifted_x_92, (0, 0, 0, 0, 0, 0));  shifted_x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    x_435 = shifted_x_93.view(8, 1, 7, 1, 7, 1024);  shifted_x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    permute_96 = x_435.permute(0, 1, 3, 2, 4, 5);  x_435 = None
    contiguous_92 = permute_96.contiguous();  permute_96 = None
    x_windows_46 = contiguous_92.view(-1, 7, 7, 1024);  contiguous_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    x_windows_47 = x_windows_46.view(-1, 49, 1024);  x_windows_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    getattr_getattr_l__mod___layers___3___blocks___1___attn_qkv = self.getattr_getattr_L__mod___layers___3___blocks___1___attn_qkv(x_windows_47);  x_windows_47 = None
    reshape_95 = getattr_getattr_l__mod___layers___3___blocks___1___attn_qkv.reshape(8, 49, 3, 32, -1);  getattr_getattr_l__mod___layers___3___blocks___1___attn_qkv = None
    qkv_23 = reshape_95.permute(2, 0, 3, 1, 4);  reshape_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    unbind_23 = qkv_23.unbind(0);  qkv_23 = None
    q_46 = unbind_23[0]
    k_23 = unbind_23[1]
    v_23 = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    q_47 = q_46 * 0.1767766952966369;  q_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    transpose_46 = k_23.transpose(-2, -1);  k_23 = None
    attn_114 = q_47 @ transpose_46;  q_47 = transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getattr_getattr_l__mod___layers___3___blocks___1___attn_relative_position_bias_table = self.getattr_getattr_L__mod___layers___3___blocks___1___attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    getattr_getattr_l__mod___layers___3___blocks___1___attn_relative_position_index = self.getattr_getattr_L__mod___layers___3___blocks___1___attn_relative_position_index
    view_209 = getattr_getattr_l__mod___layers___3___blocks___1___attn_relative_position_index.view(-1);  getattr_getattr_l__mod___layers___3___blocks___1___attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    getitem_118 = getattr_getattr_l__mod___layers___3___blocks___1___attn_relative_position_bias_table[view_209];  getattr_getattr_l__mod___layers___3___blocks___1___attn_relative_position_bias_table = view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias_46 = getitem_118.view(49, 49, -1);  getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_98 = relative_position_bias_46.permute(2, 0, 1);  relative_position_bias_46 = None
    relative_position_bias_47 = permute_98.contiguous();  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_45 = relative_position_bias_47.unsqueeze(0);  relative_position_bias_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    attn_115 = attn_114 + unsqueeze_45;  attn_114 = unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    attn_116 = self.getattr_getattr_L__mod___layers___3___blocks___1___attn_softmax(attn_115);  attn_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:178, code: attn = self.attn_drop(attn)
    attn_117 = self.getattr_getattr_L__mod___layers___3___blocks___1___attn_attn_drop(attn_116);  attn_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    x_436 = attn_117 @ v_23;  attn_117 = v_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    transpose_47 = x_436.transpose(1, 2);  x_436 = None
    x_437 = transpose_47.reshape(8, 49, -1);  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    x_438 = self.getattr_getattr_L__mod___layers___3___blocks___1___attn_proj(x_437);  x_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:183, code: x = self.proj_drop(x)
    attn_windows_46 = self.getattr_getattr_L__mod___layers___3___blocks___1___attn_proj_drop(x_438);  x_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    attn_windows_47 = attn_windows_46.view(-1, 7, 7, 1024);  attn_windows_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x_440 = attn_windows_47.view(-1, 1, 1, 7, 7, 1024);  attn_windows_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    permute_99 = x_440.permute(0, 1, 3, 2, 4, 5);  x_440 = None
    contiguous_94 = permute_99.contiguous();  permute_99 = None
    shifted_x_94 = contiguous_94.view(-1, 7, 7, 1024);  contiguous_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    getitem_119 = shifted_x_94[(slice(None, None, None), slice(None, 7, None), slice(None, 7, None), slice(None, None, None))];  shifted_x_94 = None
    x_442 = getitem_119.contiguous();  getitem_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    x_443 = x_434 + x_442;  x_434 = x_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    x_444 = x_443.reshape(8, -1, 1024);  x_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    getattr_getattr_l__mod___layers___3___blocks___1___norm2 = self.getattr_getattr_L__mod___layers___3___blocks___1___norm2(x_444)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_445 = self.getattr_getattr_L__mod___layers___3___blocks___1___mlp_fc1(getattr_getattr_l__mod___layers___3___blocks___1___norm2);  getattr_getattr_l__mod___layers___3___blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_446 = self.getattr_getattr_L__mod___layers___3___blocks___1___mlp_act(x_445);  x_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_447 = self.getattr_getattr_L__mod___layers___3___blocks___1___mlp_drop1(x_446);  x_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_448 = self.getattr_getattr_L__mod___layers___3___blocks___1___mlp_norm(x_447);  x_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_449 = self.getattr_getattr_L__mod___layers___3___blocks___1___mlp_fc2(x_448);  x_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_450 = self.getattr_getattr_L__mod___layers___3___blocks___1___mlp_drop2(x_449);  x_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    x_451 = x_444 + x_450;  x_444 = x_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    x_454 = x_451.reshape(8, 7, 7, 1024);  x_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:610, code: x = self.norm(x)
    x_456 = self.L__mod___norm(x_454);  x_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:65, code: return x.mean(self.dim, keepdim=not self.flatten)
    x_457 = x_456.mean((1, 2), keepdim = False);  x_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_459 = self.L__mod___head_global_pool_flatten(x_457);  x_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_460 = self.L__mod___head_drop(x_459);  x_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_461 = self.L__mod___head_fc(x_460);  x_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_462 = self.L__mod___head_flatten(x_461);  x_461 = None
    return (x_462,)
    