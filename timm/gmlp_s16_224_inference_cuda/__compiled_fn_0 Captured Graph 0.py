from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x = self.L__mod___stem_proj(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten = x.flatten(2);  x = None
    x_1 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_3 = self.L__mod___stem_norm(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___0___norm = self.getattr_L__mod___blocks___0___norm(x_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_4 = self.getattr_L__mod___blocks___0___mlp_channels_fc1(getattr_l__mod___blocks___0___norm);  getattr_l__mod___blocks___0___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_5 = self.getattr_L__mod___blocks___0___mlp_channels_act(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_6 = self.getattr_L__mod___blocks___0___mlp_channels_drop1(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk = x_6.chunk(2, dim = -1);  x_6 = None
    u = chunk[0]
    v = chunk[1];  chunk = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_1 = self.getattr_L__mod___blocks___0___mlp_channels_gate_norm(v);  v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_1 = v_1.transpose(-1, -2);  v_1 = None
    v_2 = self.getattr_L__mod___blocks___0___mlp_channels_gate_proj(transpose_1);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_2 = v_2.transpose(-1, -2);  v_2 = None
    x_7 = u * transpose_2;  u = transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_8 = self.getattr_L__mod___blocks___0___mlp_channels_norm(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_9 = self.getattr_L__mod___blocks___0___mlp_channels_fc2(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_10 = self.getattr_L__mod___blocks___0___mlp_channels_drop2(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___0___drop_path = self.getattr_L__mod___blocks___0___drop_path(x_10);  x_10 = None
    x_11 = x_3 + getattr_l__mod___blocks___0___drop_path;  x_3 = getattr_l__mod___blocks___0___drop_path = None
    getattr_l__mod___blocks___1___norm = self.getattr_L__mod___blocks___1___norm(x_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_12 = self.getattr_L__mod___blocks___1___mlp_channels_fc1(getattr_l__mod___blocks___1___norm);  getattr_l__mod___blocks___1___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_13 = self.getattr_L__mod___blocks___1___mlp_channels_act(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_14 = self.getattr_L__mod___blocks___1___mlp_channels_drop1(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_1 = x_14.chunk(2, dim = -1);  x_14 = None
    u_1 = chunk_1[0]
    v_3 = chunk_1[1];  chunk_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_4 = self.getattr_L__mod___blocks___1___mlp_channels_gate_norm(v_3);  v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_3 = v_4.transpose(-1, -2);  v_4 = None
    v_5 = self.getattr_L__mod___blocks___1___mlp_channels_gate_proj(transpose_3);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_4 = v_5.transpose(-1, -2);  v_5 = None
    x_15 = u_1 * transpose_4;  u_1 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_16 = self.getattr_L__mod___blocks___1___mlp_channels_norm(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_17 = self.getattr_L__mod___blocks___1___mlp_channels_fc2(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_18 = self.getattr_L__mod___blocks___1___mlp_channels_drop2(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___1___drop_path = self.getattr_L__mod___blocks___1___drop_path(x_18);  x_18 = None
    x_19 = x_11 + getattr_l__mod___blocks___1___drop_path;  x_11 = getattr_l__mod___blocks___1___drop_path = None
    getattr_l__mod___blocks___2___norm = self.getattr_L__mod___blocks___2___norm(x_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_20 = self.getattr_L__mod___blocks___2___mlp_channels_fc1(getattr_l__mod___blocks___2___norm);  getattr_l__mod___blocks___2___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_21 = self.getattr_L__mod___blocks___2___mlp_channels_act(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_22 = self.getattr_L__mod___blocks___2___mlp_channels_drop1(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_2 = x_22.chunk(2, dim = -1);  x_22 = None
    u_2 = chunk_2[0]
    v_6 = chunk_2[1];  chunk_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_7 = self.getattr_L__mod___blocks___2___mlp_channels_gate_norm(v_6);  v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_5 = v_7.transpose(-1, -2);  v_7 = None
    v_8 = self.getattr_L__mod___blocks___2___mlp_channels_gate_proj(transpose_5);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_6 = v_8.transpose(-1, -2);  v_8 = None
    x_23 = u_2 * transpose_6;  u_2 = transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_24 = self.getattr_L__mod___blocks___2___mlp_channels_norm(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_25 = self.getattr_L__mod___blocks___2___mlp_channels_fc2(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_26 = self.getattr_L__mod___blocks___2___mlp_channels_drop2(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___2___drop_path = self.getattr_L__mod___blocks___2___drop_path(x_26);  x_26 = None
    x_27 = x_19 + getattr_l__mod___blocks___2___drop_path;  x_19 = getattr_l__mod___blocks___2___drop_path = None
    getattr_l__mod___blocks___3___norm = self.getattr_L__mod___blocks___3___norm(x_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_28 = self.getattr_L__mod___blocks___3___mlp_channels_fc1(getattr_l__mod___blocks___3___norm);  getattr_l__mod___blocks___3___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_29 = self.getattr_L__mod___blocks___3___mlp_channels_act(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_30 = self.getattr_L__mod___blocks___3___mlp_channels_drop1(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_3 = x_30.chunk(2, dim = -1);  x_30 = None
    u_3 = chunk_3[0]
    v_9 = chunk_3[1];  chunk_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_10 = self.getattr_L__mod___blocks___3___mlp_channels_gate_norm(v_9);  v_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_7 = v_10.transpose(-1, -2);  v_10 = None
    v_11 = self.getattr_L__mod___blocks___3___mlp_channels_gate_proj(transpose_7);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_8 = v_11.transpose(-1, -2);  v_11 = None
    x_31 = u_3 * transpose_8;  u_3 = transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_32 = self.getattr_L__mod___blocks___3___mlp_channels_norm(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_33 = self.getattr_L__mod___blocks___3___mlp_channels_fc2(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_34 = self.getattr_L__mod___blocks___3___mlp_channels_drop2(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___3___drop_path = self.getattr_L__mod___blocks___3___drop_path(x_34);  x_34 = None
    x_35 = x_27 + getattr_l__mod___blocks___3___drop_path;  x_27 = getattr_l__mod___blocks___3___drop_path = None
    getattr_l__mod___blocks___4___norm = self.getattr_L__mod___blocks___4___norm(x_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_36 = self.getattr_L__mod___blocks___4___mlp_channels_fc1(getattr_l__mod___blocks___4___norm);  getattr_l__mod___blocks___4___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_37 = self.getattr_L__mod___blocks___4___mlp_channels_act(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_38 = self.getattr_L__mod___blocks___4___mlp_channels_drop1(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_4 = x_38.chunk(2, dim = -1);  x_38 = None
    u_4 = chunk_4[0]
    v_12 = chunk_4[1];  chunk_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_13 = self.getattr_L__mod___blocks___4___mlp_channels_gate_norm(v_12);  v_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_9 = v_13.transpose(-1, -2);  v_13 = None
    v_14 = self.getattr_L__mod___blocks___4___mlp_channels_gate_proj(transpose_9);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_10 = v_14.transpose(-1, -2);  v_14 = None
    x_39 = u_4 * transpose_10;  u_4 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_40 = self.getattr_L__mod___blocks___4___mlp_channels_norm(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_41 = self.getattr_L__mod___blocks___4___mlp_channels_fc2(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_42 = self.getattr_L__mod___blocks___4___mlp_channels_drop2(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___4___drop_path = self.getattr_L__mod___blocks___4___drop_path(x_42);  x_42 = None
    x_43 = x_35 + getattr_l__mod___blocks___4___drop_path;  x_35 = getattr_l__mod___blocks___4___drop_path = None
    getattr_l__mod___blocks___5___norm = self.getattr_L__mod___blocks___5___norm(x_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_44 = self.getattr_L__mod___blocks___5___mlp_channels_fc1(getattr_l__mod___blocks___5___norm);  getattr_l__mod___blocks___5___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_45 = self.getattr_L__mod___blocks___5___mlp_channels_act(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_46 = self.getattr_L__mod___blocks___5___mlp_channels_drop1(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_5 = x_46.chunk(2, dim = -1);  x_46 = None
    u_5 = chunk_5[0]
    v_15 = chunk_5[1];  chunk_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_16 = self.getattr_L__mod___blocks___5___mlp_channels_gate_norm(v_15);  v_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_11 = v_16.transpose(-1, -2);  v_16 = None
    v_17 = self.getattr_L__mod___blocks___5___mlp_channels_gate_proj(transpose_11);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_12 = v_17.transpose(-1, -2);  v_17 = None
    x_47 = u_5 * transpose_12;  u_5 = transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_48 = self.getattr_L__mod___blocks___5___mlp_channels_norm(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_49 = self.getattr_L__mod___blocks___5___mlp_channels_fc2(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_50 = self.getattr_L__mod___blocks___5___mlp_channels_drop2(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___5___drop_path = self.getattr_L__mod___blocks___5___drop_path(x_50);  x_50 = None
    x_51 = x_43 + getattr_l__mod___blocks___5___drop_path;  x_43 = getattr_l__mod___blocks___5___drop_path = None
    getattr_l__mod___blocks___6___norm = self.getattr_L__mod___blocks___6___norm(x_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_52 = self.getattr_L__mod___blocks___6___mlp_channels_fc1(getattr_l__mod___blocks___6___norm);  getattr_l__mod___blocks___6___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_53 = self.getattr_L__mod___blocks___6___mlp_channels_act(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_54 = self.getattr_L__mod___blocks___6___mlp_channels_drop1(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_6 = x_54.chunk(2, dim = -1);  x_54 = None
    u_6 = chunk_6[0]
    v_18 = chunk_6[1];  chunk_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_19 = self.getattr_L__mod___blocks___6___mlp_channels_gate_norm(v_18);  v_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_13 = v_19.transpose(-1, -2);  v_19 = None
    v_20 = self.getattr_L__mod___blocks___6___mlp_channels_gate_proj(transpose_13);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_14 = v_20.transpose(-1, -2);  v_20 = None
    x_55 = u_6 * transpose_14;  u_6 = transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_56 = self.getattr_L__mod___blocks___6___mlp_channels_norm(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_57 = self.getattr_L__mod___blocks___6___mlp_channels_fc2(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_58 = self.getattr_L__mod___blocks___6___mlp_channels_drop2(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___6___drop_path = self.getattr_L__mod___blocks___6___drop_path(x_58);  x_58 = None
    x_59 = x_51 + getattr_l__mod___blocks___6___drop_path;  x_51 = getattr_l__mod___blocks___6___drop_path = None
    getattr_l__mod___blocks___7___norm = self.getattr_L__mod___blocks___7___norm(x_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_60 = self.getattr_L__mod___blocks___7___mlp_channels_fc1(getattr_l__mod___blocks___7___norm);  getattr_l__mod___blocks___7___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_61 = self.getattr_L__mod___blocks___7___mlp_channels_act(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_62 = self.getattr_L__mod___blocks___7___mlp_channels_drop1(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_7 = x_62.chunk(2, dim = -1);  x_62 = None
    u_7 = chunk_7[0]
    v_21 = chunk_7[1];  chunk_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_22 = self.getattr_L__mod___blocks___7___mlp_channels_gate_norm(v_21);  v_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_15 = v_22.transpose(-1, -2);  v_22 = None
    v_23 = self.getattr_L__mod___blocks___7___mlp_channels_gate_proj(transpose_15);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_16 = v_23.transpose(-1, -2);  v_23 = None
    x_63 = u_7 * transpose_16;  u_7 = transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_64 = self.getattr_L__mod___blocks___7___mlp_channels_norm(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_65 = self.getattr_L__mod___blocks___7___mlp_channels_fc2(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_66 = self.getattr_L__mod___blocks___7___mlp_channels_drop2(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___7___drop_path = self.getattr_L__mod___blocks___7___drop_path(x_66);  x_66 = None
    x_67 = x_59 + getattr_l__mod___blocks___7___drop_path;  x_59 = getattr_l__mod___blocks___7___drop_path = None
    getattr_l__mod___blocks___8___norm = self.getattr_L__mod___blocks___8___norm(x_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_68 = self.getattr_L__mod___blocks___8___mlp_channels_fc1(getattr_l__mod___blocks___8___norm);  getattr_l__mod___blocks___8___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_69 = self.getattr_L__mod___blocks___8___mlp_channels_act(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_70 = self.getattr_L__mod___blocks___8___mlp_channels_drop1(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_8 = x_70.chunk(2, dim = -1);  x_70 = None
    u_8 = chunk_8[0]
    v_24 = chunk_8[1];  chunk_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_25 = self.getattr_L__mod___blocks___8___mlp_channels_gate_norm(v_24);  v_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_17 = v_25.transpose(-1, -2);  v_25 = None
    v_26 = self.getattr_L__mod___blocks___8___mlp_channels_gate_proj(transpose_17);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_18 = v_26.transpose(-1, -2);  v_26 = None
    x_71 = u_8 * transpose_18;  u_8 = transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_72 = self.getattr_L__mod___blocks___8___mlp_channels_norm(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_73 = self.getattr_L__mod___blocks___8___mlp_channels_fc2(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_74 = self.getattr_L__mod___blocks___8___mlp_channels_drop2(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___8___drop_path = self.getattr_L__mod___blocks___8___drop_path(x_74);  x_74 = None
    x_75 = x_67 + getattr_l__mod___blocks___8___drop_path;  x_67 = getattr_l__mod___blocks___8___drop_path = None
    getattr_l__mod___blocks___9___norm = self.getattr_L__mod___blocks___9___norm(x_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_76 = self.getattr_L__mod___blocks___9___mlp_channels_fc1(getattr_l__mod___blocks___9___norm);  getattr_l__mod___blocks___9___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_77 = self.getattr_L__mod___blocks___9___mlp_channels_act(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_78 = self.getattr_L__mod___blocks___9___mlp_channels_drop1(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_9 = x_78.chunk(2, dim = -1);  x_78 = None
    u_9 = chunk_9[0]
    v_27 = chunk_9[1];  chunk_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_28 = self.getattr_L__mod___blocks___9___mlp_channels_gate_norm(v_27);  v_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_19 = v_28.transpose(-1, -2);  v_28 = None
    v_29 = self.getattr_L__mod___blocks___9___mlp_channels_gate_proj(transpose_19);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_20 = v_29.transpose(-1, -2);  v_29 = None
    x_79 = u_9 * transpose_20;  u_9 = transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_80 = self.getattr_L__mod___blocks___9___mlp_channels_norm(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_81 = self.getattr_L__mod___blocks___9___mlp_channels_fc2(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_82 = self.getattr_L__mod___blocks___9___mlp_channels_drop2(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___9___drop_path = self.getattr_L__mod___blocks___9___drop_path(x_82);  x_82 = None
    x_83 = x_75 + getattr_l__mod___blocks___9___drop_path;  x_75 = getattr_l__mod___blocks___9___drop_path = None
    getattr_l__mod___blocks___10___norm = self.getattr_L__mod___blocks___10___norm(x_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_84 = self.getattr_L__mod___blocks___10___mlp_channels_fc1(getattr_l__mod___blocks___10___norm);  getattr_l__mod___blocks___10___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_85 = self.getattr_L__mod___blocks___10___mlp_channels_act(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_86 = self.getattr_L__mod___blocks___10___mlp_channels_drop1(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_10 = x_86.chunk(2, dim = -1);  x_86 = None
    u_10 = chunk_10[0]
    v_30 = chunk_10[1];  chunk_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_31 = self.getattr_L__mod___blocks___10___mlp_channels_gate_norm(v_30);  v_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_21 = v_31.transpose(-1, -2);  v_31 = None
    v_32 = self.getattr_L__mod___blocks___10___mlp_channels_gate_proj(transpose_21);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_22 = v_32.transpose(-1, -2);  v_32 = None
    x_87 = u_10 * transpose_22;  u_10 = transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_88 = self.getattr_L__mod___blocks___10___mlp_channels_norm(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_89 = self.getattr_L__mod___blocks___10___mlp_channels_fc2(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_90 = self.getattr_L__mod___blocks___10___mlp_channels_drop2(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___10___drop_path = self.getattr_L__mod___blocks___10___drop_path(x_90);  x_90 = None
    x_91 = x_83 + getattr_l__mod___blocks___10___drop_path;  x_83 = getattr_l__mod___blocks___10___drop_path = None
    getattr_l__mod___blocks___11___norm = self.getattr_L__mod___blocks___11___norm(x_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_92 = self.getattr_L__mod___blocks___11___mlp_channels_fc1(getattr_l__mod___blocks___11___norm);  getattr_l__mod___blocks___11___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_93 = self.getattr_L__mod___blocks___11___mlp_channels_act(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_94 = self.getattr_L__mod___blocks___11___mlp_channels_drop1(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_11 = x_94.chunk(2, dim = -1);  x_94 = None
    u_11 = chunk_11[0]
    v_33 = chunk_11[1];  chunk_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_34 = self.getattr_L__mod___blocks___11___mlp_channels_gate_norm(v_33);  v_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_23 = v_34.transpose(-1, -2);  v_34 = None
    v_35 = self.getattr_L__mod___blocks___11___mlp_channels_gate_proj(transpose_23);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_24 = v_35.transpose(-1, -2);  v_35 = None
    x_95 = u_11 * transpose_24;  u_11 = transpose_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_96 = self.getattr_L__mod___blocks___11___mlp_channels_norm(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_97 = self.getattr_L__mod___blocks___11___mlp_channels_fc2(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_98 = self.getattr_L__mod___blocks___11___mlp_channels_drop2(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___11___drop_path = self.getattr_L__mod___blocks___11___drop_path(x_98);  x_98 = None
    x_99 = x_91 + getattr_l__mod___blocks___11___drop_path;  x_91 = getattr_l__mod___blocks___11___drop_path = None
    getattr_l__mod___blocks___12___norm = self.getattr_L__mod___blocks___12___norm(x_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_100 = self.getattr_L__mod___blocks___12___mlp_channels_fc1(getattr_l__mod___blocks___12___norm);  getattr_l__mod___blocks___12___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_101 = self.getattr_L__mod___blocks___12___mlp_channels_act(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_102 = self.getattr_L__mod___blocks___12___mlp_channels_drop1(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_12 = x_102.chunk(2, dim = -1);  x_102 = None
    u_12 = chunk_12[0]
    v_36 = chunk_12[1];  chunk_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_37 = self.getattr_L__mod___blocks___12___mlp_channels_gate_norm(v_36);  v_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_25 = v_37.transpose(-1, -2);  v_37 = None
    v_38 = self.getattr_L__mod___blocks___12___mlp_channels_gate_proj(transpose_25);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_26 = v_38.transpose(-1, -2);  v_38 = None
    x_103 = u_12 * transpose_26;  u_12 = transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_104 = self.getattr_L__mod___blocks___12___mlp_channels_norm(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_105 = self.getattr_L__mod___blocks___12___mlp_channels_fc2(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_106 = self.getattr_L__mod___blocks___12___mlp_channels_drop2(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___12___drop_path = self.getattr_L__mod___blocks___12___drop_path(x_106);  x_106 = None
    x_107 = x_99 + getattr_l__mod___blocks___12___drop_path;  x_99 = getattr_l__mod___blocks___12___drop_path = None
    getattr_l__mod___blocks___13___norm = self.getattr_L__mod___blocks___13___norm(x_107)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_108 = self.getattr_L__mod___blocks___13___mlp_channels_fc1(getattr_l__mod___blocks___13___norm);  getattr_l__mod___blocks___13___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_109 = self.getattr_L__mod___blocks___13___mlp_channels_act(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_110 = self.getattr_L__mod___blocks___13___mlp_channels_drop1(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_13 = x_110.chunk(2, dim = -1);  x_110 = None
    u_13 = chunk_13[0]
    v_39 = chunk_13[1];  chunk_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_40 = self.getattr_L__mod___blocks___13___mlp_channels_gate_norm(v_39);  v_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_27 = v_40.transpose(-1, -2);  v_40 = None
    v_41 = self.getattr_L__mod___blocks___13___mlp_channels_gate_proj(transpose_27);  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_28 = v_41.transpose(-1, -2);  v_41 = None
    x_111 = u_13 * transpose_28;  u_13 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_112 = self.getattr_L__mod___blocks___13___mlp_channels_norm(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_113 = self.getattr_L__mod___blocks___13___mlp_channels_fc2(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_114 = self.getattr_L__mod___blocks___13___mlp_channels_drop2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___13___drop_path = self.getattr_L__mod___blocks___13___drop_path(x_114);  x_114 = None
    x_115 = x_107 + getattr_l__mod___blocks___13___drop_path;  x_107 = getattr_l__mod___blocks___13___drop_path = None
    getattr_l__mod___blocks___14___norm = self.getattr_L__mod___blocks___14___norm(x_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_116 = self.getattr_L__mod___blocks___14___mlp_channels_fc1(getattr_l__mod___blocks___14___norm);  getattr_l__mod___blocks___14___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_117 = self.getattr_L__mod___blocks___14___mlp_channels_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_118 = self.getattr_L__mod___blocks___14___mlp_channels_drop1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_14 = x_118.chunk(2, dim = -1);  x_118 = None
    u_14 = chunk_14[0]
    v_42 = chunk_14[1];  chunk_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_43 = self.getattr_L__mod___blocks___14___mlp_channels_gate_norm(v_42);  v_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_29 = v_43.transpose(-1, -2);  v_43 = None
    v_44 = self.getattr_L__mod___blocks___14___mlp_channels_gate_proj(transpose_29);  transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_30 = v_44.transpose(-1, -2);  v_44 = None
    x_119 = u_14 * transpose_30;  u_14 = transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_120 = self.getattr_L__mod___blocks___14___mlp_channels_norm(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_121 = self.getattr_L__mod___blocks___14___mlp_channels_fc2(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_122 = self.getattr_L__mod___blocks___14___mlp_channels_drop2(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___14___drop_path = self.getattr_L__mod___blocks___14___drop_path(x_122);  x_122 = None
    x_123 = x_115 + getattr_l__mod___blocks___14___drop_path;  x_115 = getattr_l__mod___blocks___14___drop_path = None
    getattr_l__mod___blocks___15___norm = self.getattr_L__mod___blocks___15___norm(x_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_124 = self.getattr_L__mod___blocks___15___mlp_channels_fc1(getattr_l__mod___blocks___15___norm);  getattr_l__mod___blocks___15___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_125 = self.getattr_L__mod___blocks___15___mlp_channels_act(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_126 = self.getattr_L__mod___blocks___15___mlp_channels_drop1(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_15 = x_126.chunk(2, dim = -1);  x_126 = None
    u_15 = chunk_15[0]
    v_45 = chunk_15[1];  chunk_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_46 = self.getattr_L__mod___blocks___15___mlp_channels_gate_norm(v_45);  v_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_31 = v_46.transpose(-1, -2);  v_46 = None
    v_47 = self.getattr_L__mod___blocks___15___mlp_channels_gate_proj(transpose_31);  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_32 = v_47.transpose(-1, -2);  v_47 = None
    x_127 = u_15 * transpose_32;  u_15 = transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_128 = self.getattr_L__mod___blocks___15___mlp_channels_norm(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_129 = self.getattr_L__mod___blocks___15___mlp_channels_fc2(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_130 = self.getattr_L__mod___blocks___15___mlp_channels_drop2(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___15___drop_path = self.getattr_L__mod___blocks___15___drop_path(x_130);  x_130 = None
    x_131 = x_123 + getattr_l__mod___blocks___15___drop_path;  x_123 = getattr_l__mod___blocks___15___drop_path = None
    getattr_l__mod___blocks___16___norm = self.getattr_L__mod___blocks___16___norm(x_131)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_132 = self.getattr_L__mod___blocks___16___mlp_channels_fc1(getattr_l__mod___blocks___16___norm);  getattr_l__mod___blocks___16___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_133 = self.getattr_L__mod___blocks___16___mlp_channels_act(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_134 = self.getattr_L__mod___blocks___16___mlp_channels_drop1(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_16 = x_134.chunk(2, dim = -1);  x_134 = None
    u_16 = chunk_16[0]
    v_48 = chunk_16[1];  chunk_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_49 = self.getattr_L__mod___blocks___16___mlp_channels_gate_norm(v_48);  v_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_33 = v_49.transpose(-1, -2);  v_49 = None
    v_50 = self.getattr_L__mod___blocks___16___mlp_channels_gate_proj(transpose_33);  transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_34 = v_50.transpose(-1, -2);  v_50 = None
    x_135 = u_16 * transpose_34;  u_16 = transpose_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_136 = self.getattr_L__mod___blocks___16___mlp_channels_norm(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_137 = self.getattr_L__mod___blocks___16___mlp_channels_fc2(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_138 = self.getattr_L__mod___blocks___16___mlp_channels_drop2(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___16___drop_path = self.getattr_L__mod___blocks___16___drop_path(x_138);  x_138 = None
    x_139 = x_131 + getattr_l__mod___blocks___16___drop_path;  x_131 = getattr_l__mod___blocks___16___drop_path = None
    getattr_l__mod___blocks___17___norm = self.getattr_L__mod___blocks___17___norm(x_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_140 = self.getattr_L__mod___blocks___17___mlp_channels_fc1(getattr_l__mod___blocks___17___norm);  getattr_l__mod___blocks___17___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_141 = self.getattr_L__mod___blocks___17___mlp_channels_act(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_142 = self.getattr_L__mod___blocks___17___mlp_channels_drop1(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_17 = x_142.chunk(2, dim = -1);  x_142 = None
    u_17 = chunk_17[0]
    v_51 = chunk_17[1];  chunk_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_52 = self.getattr_L__mod___blocks___17___mlp_channels_gate_norm(v_51);  v_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_35 = v_52.transpose(-1, -2);  v_52 = None
    v_53 = self.getattr_L__mod___blocks___17___mlp_channels_gate_proj(transpose_35);  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_36 = v_53.transpose(-1, -2);  v_53 = None
    x_143 = u_17 * transpose_36;  u_17 = transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_144 = self.getattr_L__mod___blocks___17___mlp_channels_norm(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_145 = self.getattr_L__mod___blocks___17___mlp_channels_fc2(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_146 = self.getattr_L__mod___blocks___17___mlp_channels_drop2(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___17___drop_path = self.getattr_L__mod___blocks___17___drop_path(x_146);  x_146 = None
    x_147 = x_139 + getattr_l__mod___blocks___17___drop_path;  x_139 = getattr_l__mod___blocks___17___drop_path = None
    getattr_l__mod___blocks___18___norm = self.getattr_L__mod___blocks___18___norm(x_147)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_148 = self.getattr_L__mod___blocks___18___mlp_channels_fc1(getattr_l__mod___blocks___18___norm);  getattr_l__mod___blocks___18___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_149 = self.getattr_L__mod___blocks___18___mlp_channels_act(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_150 = self.getattr_L__mod___blocks___18___mlp_channels_drop1(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_18 = x_150.chunk(2, dim = -1);  x_150 = None
    u_18 = chunk_18[0]
    v_54 = chunk_18[1];  chunk_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_55 = self.getattr_L__mod___blocks___18___mlp_channels_gate_norm(v_54);  v_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_37 = v_55.transpose(-1, -2);  v_55 = None
    v_56 = self.getattr_L__mod___blocks___18___mlp_channels_gate_proj(transpose_37);  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_38 = v_56.transpose(-1, -2);  v_56 = None
    x_151 = u_18 * transpose_38;  u_18 = transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_152 = self.getattr_L__mod___blocks___18___mlp_channels_norm(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_153 = self.getattr_L__mod___blocks___18___mlp_channels_fc2(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_154 = self.getattr_L__mod___blocks___18___mlp_channels_drop2(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___18___drop_path = self.getattr_L__mod___blocks___18___drop_path(x_154);  x_154 = None
    x_155 = x_147 + getattr_l__mod___blocks___18___drop_path;  x_147 = getattr_l__mod___blocks___18___drop_path = None
    getattr_l__mod___blocks___19___norm = self.getattr_L__mod___blocks___19___norm(x_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_156 = self.getattr_L__mod___blocks___19___mlp_channels_fc1(getattr_l__mod___blocks___19___norm);  getattr_l__mod___blocks___19___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_157 = self.getattr_L__mod___blocks___19___mlp_channels_act(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_158 = self.getattr_L__mod___blocks___19___mlp_channels_drop1(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_19 = x_158.chunk(2, dim = -1);  x_158 = None
    u_19 = chunk_19[0]
    v_57 = chunk_19[1];  chunk_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_58 = self.getattr_L__mod___blocks___19___mlp_channels_gate_norm(v_57);  v_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_39 = v_58.transpose(-1, -2);  v_58 = None
    v_59 = self.getattr_L__mod___blocks___19___mlp_channels_gate_proj(transpose_39);  transpose_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_40 = v_59.transpose(-1, -2);  v_59 = None
    x_159 = u_19 * transpose_40;  u_19 = transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_160 = self.getattr_L__mod___blocks___19___mlp_channels_norm(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_161 = self.getattr_L__mod___blocks___19___mlp_channels_fc2(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_162 = self.getattr_L__mod___blocks___19___mlp_channels_drop2(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___19___drop_path = self.getattr_L__mod___blocks___19___drop_path(x_162);  x_162 = None
    x_163 = x_155 + getattr_l__mod___blocks___19___drop_path;  x_155 = getattr_l__mod___blocks___19___drop_path = None
    getattr_l__mod___blocks___20___norm = self.getattr_L__mod___blocks___20___norm(x_163)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_164 = self.getattr_L__mod___blocks___20___mlp_channels_fc1(getattr_l__mod___blocks___20___norm);  getattr_l__mod___blocks___20___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_165 = self.getattr_L__mod___blocks___20___mlp_channels_act(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_166 = self.getattr_L__mod___blocks___20___mlp_channels_drop1(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_20 = x_166.chunk(2, dim = -1);  x_166 = None
    u_20 = chunk_20[0]
    v_60 = chunk_20[1];  chunk_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_61 = self.getattr_L__mod___blocks___20___mlp_channels_gate_norm(v_60);  v_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_41 = v_61.transpose(-1, -2);  v_61 = None
    v_62 = self.getattr_L__mod___blocks___20___mlp_channels_gate_proj(transpose_41);  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_42 = v_62.transpose(-1, -2);  v_62 = None
    x_167 = u_20 * transpose_42;  u_20 = transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_168 = self.getattr_L__mod___blocks___20___mlp_channels_norm(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_169 = self.getattr_L__mod___blocks___20___mlp_channels_fc2(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_170 = self.getattr_L__mod___blocks___20___mlp_channels_drop2(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___20___drop_path = self.getattr_L__mod___blocks___20___drop_path(x_170);  x_170 = None
    x_171 = x_163 + getattr_l__mod___blocks___20___drop_path;  x_163 = getattr_l__mod___blocks___20___drop_path = None
    getattr_l__mod___blocks___21___norm = self.getattr_L__mod___blocks___21___norm(x_171)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_172 = self.getattr_L__mod___blocks___21___mlp_channels_fc1(getattr_l__mod___blocks___21___norm);  getattr_l__mod___blocks___21___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_173 = self.getattr_L__mod___blocks___21___mlp_channels_act(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_174 = self.getattr_L__mod___blocks___21___mlp_channels_drop1(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_21 = x_174.chunk(2, dim = -1);  x_174 = None
    u_21 = chunk_21[0]
    v_63 = chunk_21[1];  chunk_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_64 = self.getattr_L__mod___blocks___21___mlp_channels_gate_norm(v_63);  v_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_43 = v_64.transpose(-1, -2);  v_64 = None
    v_65 = self.getattr_L__mod___blocks___21___mlp_channels_gate_proj(transpose_43);  transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_44 = v_65.transpose(-1, -2);  v_65 = None
    x_175 = u_21 * transpose_44;  u_21 = transpose_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_176 = self.getattr_L__mod___blocks___21___mlp_channels_norm(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_177 = self.getattr_L__mod___blocks___21___mlp_channels_fc2(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_178 = self.getattr_L__mod___blocks___21___mlp_channels_drop2(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___21___drop_path = self.getattr_L__mod___blocks___21___drop_path(x_178);  x_178 = None
    x_179 = x_171 + getattr_l__mod___blocks___21___drop_path;  x_171 = getattr_l__mod___blocks___21___drop_path = None
    getattr_l__mod___blocks___22___norm = self.getattr_L__mod___blocks___22___norm(x_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_180 = self.getattr_L__mod___blocks___22___mlp_channels_fc1(getattr_l__mod___blocks___22___norm);  getattr_l__mod___blocks___22___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_181 = self.getattr_L__mod___blocks___22___mlp_channels_act(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_182 = self.getattr_L__mod___blocks___22___mlp_channels_drop1(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_22 = x_182.chunk(2, dim = -1);  x_182 = None
    u_22 = chunk_22[0]
    v_66 = chunk_22[1];  chunk_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_67 = self.getattr_L__mod___blocks___22___mlp_channels_gate_norm(v_66);  v_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_45 = v_67.transpose(-1, -2);  v_67 = None
    v_68 = self.getattr_L__mod___blocks___22___mlp_channels_gate_proj(transpose_45);  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_46 = v_68.transpose(-1, -2);  v_68 = None
    x_183 = u_22 * transpose_46;  u_22 = transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_184 = self.getattr_L__mod___blocks___22___mlp_channels_norm(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_185 = self.getattr_L__mod___blocks___22___mlp_channels_fc2(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_186 = self.getattr_L__mod___blocks___22___mlp_channels_drop2(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___22___drop_path = self.getattr_L__mod___blocks___22___drop_path(x_186);  x_186 = None
    x_187 = x_179 + getattr_l__mod___blocks___22___drop_path;  x_179 = getattr_l__mod___blocks___22___drop_path = None
    getattr_l__mod___blocks___23___norm = self.getattr_L__mod___blocks___23___norm(x_187)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_188 = self.getattr_L__mod___blocks___23___mlp_channels_fc1(getattr_l__mod___blocks___23___norm);  getattr_l__mod___blocks___23___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_189 = self.getattr_L__mod___blocks___23___mlp_channels_act(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_190 = self.getattr_L__mod___blocks___23___mlp_channels_drop1(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_23 = x_190.chunk(2, dim = -1);  x_190 = None
    u_23 = chunk_23[0]
    v_69 = chunk_23[1];  chunk_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_70 = self.getattr_L__mod___blocks___23___mlp_channels_gate_norm(v_69);  v_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_47 = v_70.transpose(-1, -2);  v_70 = None
    v_71 = self.getattr_L__mod___blocks___23___mlp_channels_gate_proj(transpose_47);  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_48 = v_71.transpose(-1, -2);  v_71 = None
    x_191 = u_23 * transpose_48;  u_23 = transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_192 = self.getattr_L__mod___blocks___23___mlp_channels_norm(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_193 = self.getattr_L__mod___blocks___23___mlp_channels_fc2(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_194 = self.getattr_L__mod___blocks___23___mlp_channels_drop2(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___23___drop_path = self.getattr_L__mod___blocks___23___drop_path(x_194);  x_194 = None
    x_195 = x_187 + getattr_l__mod___blocks___23___drop_path;  x_187 = getattr_l__mod___blocks___23___drop_path = None
    getattr_l__mod___blocks___24___norm = self.getattr_L__mod___blocks___24___norm(x_195)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_196 = self.getattr_L__mod___blocks___24___mlp_channels_fc1(getattr_l__mod___blocks___24___norm);  getattr_l__mod___blocks___24___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_197 = self.getattr_L__mod___blocks___24___mlp_channels_act(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_198 = self.getattr_L__mod___blocks___24___mlp_channels_drop1(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_24 = x_198.chunk(2, dim = -1);  x_198 = None
    u_24 = chunk_24[0]
    v_72 = chunk_24[1];  chunk_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_73 = self.getattr_L__mod___blocks___24___mlp_channels_gate_norm(v_72);  v_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_49 = v_73.transpose(-1, -2);  v_73 = None
    v_74 = self.getattr_L__mod___blocks___24___mlp_channels_gate_proj(transpose_49);  transpose_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_50 = v_74.transpose(-1, -2);  v_74 = None
    x_199 = u_24 * transpose_50;  u_24 = transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_200 = self.getattr_L__mod___blocks___24___mlp_channels_norm(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_201 = self.getattr_L__mod___blocks___24___mlp_channels_fc2(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_202 = self.getattr_L__mod___blocks___24___mlp_channels_drop2(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___24___drop_path = self.getattr_L__mod___blocks___24___drop_path(x_202);  x_202 = None
    x_203 = x_195 + getattr_l__mod___blocks___24___drop_path;  x_195 = getattr_l__mod___blocks___24___drop_path = None
    getattr_l__mod___blocks___25___norm = self.getattr_L__mod___blocks___25___norm(x_203)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_204 = self.getattr_L__mod___blocks___25___mlp_channels_fc1(getattr_l__mod___blocks___25___norm);  getattr_l__mod___blocks___25___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_205 = self.getattr_L__mod___blocks___25___mlp_channels_act(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_206 = self.getattr_L__mod___blocks___25___mlp_channels_drop1(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_25 = x_206.chunk(2, dim = -1);  x_206 = None
    u_25 = chunk_25[0]
    v_75 = chunk_25[1];  chunk_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_76 = self.getattr_L__mod___blocks___25___mlp_channels_gate_norm(v_75);  v_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_51 = v_76.transpose(-1, -2);  v_76 = None
    v_77 = self.getattr_L__mod___blocks___25___mlp_channels_gate_proj(transpose_51);  transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_52 = v_77.transpose(-1, -2);  v_77 = None
    x_207 = u_25 * transpose_52;  u_25 = transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_208 = self.getattr_L__mod___blocks___25___mlp_channels_norm(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_209 = self.getattr_L__mod___blocks___25___mlp_channels_fc2(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_210 = self.getattr_L__mod___blocks___25___mlp_channels_drop2(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___25___drop_path = self.getattr_L__mod___blocks___25___drop_path(x_210);  x_210 = None
    x_211 = x_203 + getattr_l__mod___blocks___25___drop_path;  x_203 = getattr_l__mod___blocks___25___drop_path = None
    getattr_l__mod___blocks___26___norm = self.getattr_L__mod___blocks___26___norm(x_211)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_212 = self.getattr_L__mod___blocks___26___mlp_channels_fc1(getattr_l__mod___blocks___26___norm);  getattr_l__mod___blocks___26___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_213 = self.getattr_L__mod___blocks___26___mlp_channels_act(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_214 = self.getattr_L__mod___blocks___26___mlp_channels_drop1(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_26 = x_214.chunk(2, dim = -1);  x_214 = None
    u_26 = chunk_26[0]
    v_78 = chunk_26[1];  chunk_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_79 = self.getattr_L__mod___blocks___26___mlp_channels_gate_norm(v_78);  v_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_53 = v_79.transpose(-1, -2);  v_79 = None
    v_80 = self.getattr_L__mod___blocks___26___mlp_channels_gate_proj(transpose_53);  transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_54 = v_80.transpose(-1, -2);  v_80 = None
    x_215 = u_26 * transpose_54;  u_26 = transpose_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_216 = self.getattr_L__mod___blocks___26___mlp_channels_norm(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_217 = self.getattr_L__mod___blocks___26___mlp_channels_fc2(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_218 = self.getattr_L__mod___blocks___26___mlp_channels_drop2(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___26___drop_path = self.getattr_L__mod___blocks___26___drop_path(x_218);  x_218 = None
    x_219 = x_211 + getattr_l__mod___blocks___26___drop_path;  x_211 = getattr_l__mod___blocks___26___drop_path = None
    getattr_l__mod___blocks___27___norm = self.getattr_L__mod___blocks___27___norm(x_219)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_220 = self.getattr_L__mod___blocks___27___mlp_channels_fc1(getattr_l__mod___blocks___27___norm);  getattr_l__mod___blocks___27___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_221 = self.getattr_L__mod___blocks___27___mlp_channels_act(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_222 = self.getattr_L__mod___blocks___27___mlp_channels_drop1(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_27 = x_222.chunk(2, dim = -1);  x_222 = None
    u_27 = chunk_27[0]
    v_81 = chunk_27[1];  chunk_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_82 = self.getattr_L__mod___blocks___27___mlp_channels_gate_norm(v_81);  v_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_55 = v_82.transpose(-1, -2);  v_82 = None
    v_83 = self.getattr_L__mod___blocks___27___mlp_channels_gate_proj(transpose_55);  transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_56 = v_83.transpose(-1, -2);  v_83 = None
    x_223 = u_27 * transpose_56;  u_27 = transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_224 = self.getattr_L__mod___blocks___27___mlp_channels_norm(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_225 = self.getattr_L__mod___blocks___27___mlp_channels_fc2(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_226 = self.getattr_L__mod___blocks___27___mlp_channels_drop2(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___27___drop_path = self.getattr_L__mod___blocks___27___drop_path(x_226);  x_226 = None
    x_227 = x_219 + getattr_l__mod___blocks___27___drop_path;  x_219 = getattr_l__mod___blocks___27___drop_path = None
    getattr_l__mod___blocks___28___norm = self.getattr_L__mod___blocks___28___norm(x_227)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_228 = self.getattr_L__mod___blocks___28___mlp_channels_fc1(getattr_l__mod___blocks___28___norm);  getattr_l__mod___blocks___28___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_229 = self.getattr_L__mod___blocks___28___mlp_channels_act(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_230 = self.getattr_L__mod___blocks___28___mlp_channels_drop1(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_28 = x_230.chunk(2, dim = -1);  x_230 = None
    u_28 = chunk_28[0]
    v_84 = chunk_28[1];  chunk_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_85 = self.getattr_L__mod___blocks___28___mlp_channels_gate_norm(v_84);  v_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_57 = v_85.transpose(-1, -2);  v_85 = None
    v_86 = self.getattr_L__mod___blocks___28___mlp_channels_gate_proj(transpose_57);  transpose_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_58 = v_86.transpose(-1, -2);  v_86 = None
    x_231 = u_28 * transpose_58;  u_28 = transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_232 = self.getattr_L__mod___blocks___28___mlp_channels_norm(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_233 = self.getattr_L__mod___blocks___28___mlp_channels_fc2(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_234 = self.getattr_L__mod___blocks___28___mlp_channels_drop2(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___28___drop_path = self.getattr_L__mod___blocks___28___drop_path(x_234);  x_234 = None
    x_235 = x_227 + getattr_l__mod___blocks___28___drop_path;  x_227 = getattr_l__mod___blocks___28___drop_path = None
    getattr_l__mod___blocks___29___norm = self.getattr_L__mod___blocks___29___norm(x_235)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    x_236 = self.getattr_L__mod___blocks___29___mlp_channels_fc1(getattr_l__mod___blocks___29___norm);  getattr_l__mod___blocks___29___norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    x_237 = self.getattr_L__mod___blocks___29___mlp_channels_act(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    x_238 = self.getattr_L__mod___blocks___29___mlp_channels_drop1(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    chunk_29 = x_238.chunk(2, dim = -1);  x_238 = None
    u_29 = chunk_29[0]
    v_87 = chunk_29[1];  chunk_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    v_88 = self.getattr_L__mod___blocks___29___mlp_channels_gate_norm(v_87);  v_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    transpose_59 = v_88.transpose(-1, -2);  v_88 = None
    v_89 = self.getattr_L__mod___blocks___29___mlp_channels_gate_proj(transpose_59);  transpose_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    transpose_60 = v_89.transpose(-1, -2);  v_89 = None
    x_239 = u_29 * transpose_60;  u_29 = transpose_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:189, code: x = self.norm(x)
    x_240 = self.getattr_L__mod___blocks___29___mlp_channels_norm(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    x_241 = self.getattr_L__mod___blocks___29___mlp_channels_fc2(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    x_242 = self.getattr_L__mod___blocks___29___mlp_channels_drop2(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    getattr_l__mod___blocks___29___drop_path = self.getattr_L__mod___blocks___29___drop_path(x_242);  x_242 = None
    x_244 = x_235 + getattr_l__mod___blocks___29___drop_path;  x_235 = getattr_l__mod___blocks___29___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    x_246 = self.L__mod___norm(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    x_247 = x_246.mean(dim = 1);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    x_248 = self.L__mod___head_drop(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    x_249 = self.L__mod___head(x_248);  x_248 = None
    return (x_249,)
    