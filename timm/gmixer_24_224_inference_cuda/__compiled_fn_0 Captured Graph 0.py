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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___0___norm1 = self.getattr_L__mod___blocks___0___norm1(x_3)
    transpose_1 = getattr_l__mod___blocks___0___norm1.transpose(1, 2);  getattr_l__mod___blocks___0___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_4 = self.getattr_L__mod___blocks___0___mlp_tokens_fc1(transpose_1);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk = x_4.chunk(2, dim = -1);  x_4 = None
    x1 = chunk[0]
    x2 = chunk[1];  chunk = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___0___mlp_tokens_act = self.getattr_L__mod___blocks___0___mlp_tokens_act(x2);  x2 = None
    x_5 = x1 * getattr_l__mod___blocks___0___mlp_tokens_act;  x1 = getattr_l__mod___blocks___0___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_6 = self.getattr_L__mod___blocks___0___mlp_tokens_drop1(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_7 = self.getattr_L__mod___blocks___0___mlp_tokens_norm(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_8 = self.getattr_L__mod___blocks___0___mlp_tokens_fc2(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_9 = self.getattr_L__mod___blocks___0___mlp_tokens_drop2(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_2 = x_9.transpose(1, 2);  x_9 = None
    getattr_l__mod___blocks___0___drop_path = self.getattr_L__mod___blocks___0___drop_path(transpose_2);  transpose_2 = None
    x_10 = x_3 + getattr_l__mod___blocks___0___drop_path;  x_3 = getattr_l__mod___blocks___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___0___norm2 = self.getattr_L__mod___blocks___0___norm2(x_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_11 = self.getattr_L__mod___blocks___0___mlp_channels_fc1(getattr_l__mod___blocks___0___norm2);  getattr_l__mod___blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_1 = x_11.chunk(2, dim = -1);  x_11 = None
    x1_1 = chunk_1[0]
    x2_1 = chunk_1[1];  chunk_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___0___mlp_channels_act = self.getattr_L__mod___blocks___0___mlp_channels_act(x2_1);  x2_1 = None
    x_12 = x1_1 * getattr_l__mod___blocks___0___mlp_channels_act;  x1_1 = getattr_l__mod___blocks___0___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_13 = self.getattr_L__mod___blocks___0___mlp_channels_drop1(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_14 = self.getattr_L__mod___blocks___0___mlp_channels_norm(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_15 = self.getattr_L__mod___blocks___0___mlp_channels_fc2(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_16 = self.getattr_L__mod___blocks___0___mlp_channels_drop2(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___0___drop_path_1 = self.getattr_L__mod___blocks___0___drop_path(x_16);  x_16 = None
    x_17 = x_10 + getattr_l__mod___blocks___0___drop_path_1;  x_10 = getattr_l__mod___blocks___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___1___norm1 = self.getattr_L__mod___blocks___1___norm1(x_17)
    transpose_3 = getattr_l__mod___blocks___1___norm1.transpose(1, 2);  getattr_l__mod___blocks___1___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_18 = self.getattr_L__mod___blocks___1___mlp_tokens_fc1(transpose_3);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_2 = x_18.chunk(2, dim = -1);  x_18 = None
    x1_2 = chunk_2[0]
    x2_2 = chunk_2[1];  chunk_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___1___mlp_tokens_act = self.getattr_L__mod___blocks___1___mlp_tokens_act(x2_2);  x2_2 = None
    x_19 = x1_2 * getattr_l__mod___blocks___1___mlp_tokens_act;  x1_2 = getattr_l__mod___blocks___1___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_20 = self.getattr_L__mod___blocks___1___mlp_tokens_drop1(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_21 = self.getattr_L__mod___blocks___1___mlp_tokens_norm(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_22 = self.getattr_L__mod___blocks___1___mlp_tokens_fc2(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_23 = self.getattr_L__mod___blocks___1___mlp_tokens_drop2(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_4 = x_23.transpose(1, 2);  x_23 = None
    getattr_l__mod___blocks___1___drop_path = self.getattr_L__mod___blocks___1___drop_path(transpose_4);  transpose_4 = None
    x_24 = x_17 + getattr_l__mod___blocks___1___drop_path;  x_17 = getattr_l__mod___blocks___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___1___norm2 = self.getattr_L__mod___blocks___1___norm2(x_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_25 = self.getattr_L__mod___blocks___1___mlp_channels_fc1(getattr_l__mod___blocks___1___norm2);  getattr_l__mod___blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_3 = x_25.chunk(2, dim = -1);  x_25 = None
    x1_3 = chunk_3[0]
    x2_3 = chunk_3[1];  chunk_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___1___mlp_channels_act = self.getattr_L__mod___blocks___1___mlp_channels_act(x2_3);  x2_3 = None
    x_26 = x1_3 * getattr_l__mod___blocks___1___mlp_channels_act;  x1_3 = getattr_l__mod___blocks___1___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_27 = self.getattr_L__mod___blocks___1___mlp_channels_drop1(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_28 = self.getattr_L__mod___blocks___1___mlp_channels_norm(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_29 = self.getattr_L__mod___blocks___1___mlp_channels_fc2(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_30 = self.getattr_L__mod___blocks___1___mlp_channels_drop2(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___1___drop_path_1 = self.getattr_L__mod___blocks___1___drop_path(x_30);  x_30 = None
    x_31 = x_24 + getattr_l__mod___blocks___1___drop_path_1;  x_24 = getattr_l__mod___blocks___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___2___norm1 = self.getattr_L__mod___blocks___2___norm1(x_31)
    transpose_5 = getattr_l__mod___blocks___2___norm1.transpose(1, 2);  getattr_l__mod___blocks___2___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_32 = self.getattr_L__mod___blocks___2___mlp_tokens_fc1(transpose_5);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_4 = x_32.chunk(2, dim = -1);  x_32 = None
    x1_4 = chunk_4[0]
    x2_4 = chunk_4[1];  chunk_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___2___mlp_tokens_act = self.getattr_L__mod___blocks___2___mlp_tokens_act(x2_4);  x2_4 = None
    x_33 = x1_4 * getattr_l__mod___blocks___2___mlp_tokens_act;  x1_4 = getattr_l__mod___blocks___2___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_34 = self.getattr_L__mod___blocks___2___mlp_tokens_drop1(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_35 = self.getattr_L__mod___blocks___2___mlp_tokens_norm(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_36 = self.getattr_L__mod___blocks___2___mlp_tokens_fc2(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_37 = self.getattr_L__mod___blocks___2___mlp_tokens_drop2(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_6 = x_37.transpose(1, 2);  x_37 = None
    getattr_l__mod___blocks___2___drop_path = self.getattr_L__mod___blocks___2___drop_path(transpose_6);  transpose_6 = None
    x_38 = x_31 + getattr_l__mod___blocks___2___drop_path;  x_31 = getattr_l__mod___blocks___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___2___norm2 = self.getattr_L__mod___blocks___2___norm2(x_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_39 = self.getattr_L__mod___blocks___2___mlp_channels_fc1(getattr_l__mod___blocks___2___norm2);  getattr_l__mod___blocks___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_5 = x_39.chunk(2, dim = -1);  x_39 = None
    x1_5 = chunk_5[0]
    x2_5 = chunk_5[1];  chunk_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___2___mlp_channels_act = self.getattr_L__mod___blocks___2___mlp_channels_act(x2_5);  x2_5 = None
    x_40 = x1_5 * getattr_l__mod___blocks___2___mlp_channels_act;  x1_5 = getattr_l__mod___blocks___2___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_41 = self.getattr_L__mod___blocks___2___mlp_channels_drop1(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_42 = self.getattr_L__mod___blocks___2___mlp_channels_norm(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_43 = self.getattr_L__mod___blocks___2___mlp_channels_fc2(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_44 = self.getattr_L__mod___blocks___2___mlp_channels_drop2(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___2___drop_path_1 = self.getattr_L__mod___blocks___2___drop_path(x_44);  x_44 = None
    x_45 = x_38 + getattr_l__mod___blocks___2___drop_path_1;  x_38 = getattr_l__mod___blocks___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___3___norm1 = self.getattr_L__mod___blocks___3___norm1(x_45)
    transpose_7 = getattr_l__mod___blocks___3___norm1.transpose(1, 2);  getattr_l__mod___blocks___3___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_46 = self.getattr_L__mod___blocks___3___mlp_tokens_fc1(transpose_7);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_6 = x_46.chunk(2, dim = -1);  x_46 = None
    x1_6 = chunk_6[0]
    x2_6 = chunk_6[1];  chunk_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___3___mlp_tokens_act = self.getattr_L__mod___blocks___3___mlp_tokens_act(x2_6);  x2_6 = None
    x_47 = x1_6 * getattr_l__mod___blocks___3___mlp_tokens_act;  x1_6 = getattr_l__mod___blocks___3___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_48 = self.getattr_L__mod___blocks___3___mlp_tokens_drop1(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_49 = self.getattr_L__mod___blocks___3___mlp_tokens_norm(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_50 = self.getattr_L__mod___blocks___3___mlp_tokens_fc2(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_51 = self.getattr_L__mod___blocks___3___mlp_tokens_drop2(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_8 = x_51.transpose(1, 2);  x_51 = None
    getattr_l__mod___blocks___3___drop_path = self.getattr_L__mod___blocks___3___drop_path(transpose_8);  transpose_8 = None
    x_52 = x_45 + getattr_l__mod___blocks___3___drop_path;  x_45 = getattr_l__mod___blocks___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___3___norm2 = self.getattr_L__mod___blocks___3___norm2(x_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_53 = self.getattr_L__mod___blocks___3___mlp_channels_fc1(getattr_l__mod___blocks___3___norm2);  getattr_l__mod___blocks___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_7 = x_53.chunk(2, dim = -1);  x_53 = None
    x1_7 = chunk_7[0]
    x2_7 = chunk_7[1];  chunk_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___3___mlp_channels_act = self.getattr_L__mod___blocks___3___mlp_channels_act(x2_7);  x2_7 = None
    x_54 = x1_7 * getattr_l__mod___blocks___3___mlp_channels_act;  x1_7 = getattr_l__mod___blocks___3___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_55 = self.getattr_L__mod___blocks___3___mlp_channels_drop1(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_56 = self.getattr_L__mod___blocks___3___mlp_channels_norm(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_57 = self.getattr_L__mod___blocks___3___mlp_channels_fc2(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_58 = self.getattr_L__mod___blocks___3___mlp_channels_drop2(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___3___drop_path_1 = self.getattr_L__mod___blocks___3___drop_path(x_58);  x_58 = None
    x_59 = x_52 + getattr_l__mod___blocks___3___drop_path_1;  x_52 = getattr_l__mod___blocks___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___4___norm1 = self.getattr_L__mod___blocks___4___norm1(x_59)
    transpose_9 = getattr_l__mod___blocks___4___norm1.transpose(1, 2);  getattr_l__mod___blocks___4___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_60 = self.getattr_L__mod___blocks___4___mlp_tokens_fc1(transpose_9);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_8 = x_60.chunk(2, dim = -1);  x_60 = None
    x1_8 = chunk_8[0]
    x2_8 = chunk_8[1];  chunk_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___4___mlp_tokens_act = self.getattr_L__mod___blocks___4___mlp_tokens_act(x2_8);  x2_8 = None
    x_61 = x1_8 * getattr_l__mod___blocks___4___mlp_tokens_act;  x1_8 = getattr_l__mod___blocks___4___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_62 = self.getattr_L__mod___blocks___4___mlp_tokens_drop1(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_63 = self.getattr_L__mod___blocks___4___mlp_tokens_norm(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_64 = self.getattr_L__mod___blocks___4___mlp_tokens_fc2(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_65 = self.getattr_L__mod___blocks___4___mlp_tokens_drop2(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_10 = x_65.transpose(1, 2);  x_65 = None
    getattr_l__mod___blocks___4___drop_path = self.getattr_L__mod___blocks___4___drop_path(transpose_10);  transpose_10 = None
    x_66 = x_59 + getattr_l__mod___blocks___4___drop_path;  x_59 = getattr_l__mod___blocks___4___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___4___norm2 = self.getattr_L__mod___blocks___4___norm2(x_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_67 = self.getattr_L__mod___blocks___4___mlp_channels_fc1(getattr_l__mod___blocks___4___norm2);  getattr_l__mod___blocks___4___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_9 = x_67.chunk(2, dim = -1);  x_67 = None
    x1_9 = chunk_9[0]
    x2_9 = chunk_9[1];  chunk_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___4___mlp_channels_act = self.getattr_L__mod___blocks___4___mlp_channels_act(x2_9);  x2_9 = None
    x_68 = x1_9 * getattr_l__mod___blocks___4___mlp_channels_act;  x1_9 = getattr_l__mod___blocks___4___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_69 = self.getattr_L__mod___blocks___4___mlp_channels_drop1(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_70 = self.getattr_L__mod___blocks___4___mlp_channels_norm(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_71 = self.getattr_L__mod___blocks___4___mlp_channels_fc2(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_72 = self.getattr_L__mod___blocks___4___mlp_channels_drop2(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___4___drop_path_1 = self.getattr_L__mod___blocks___4___drop_path(x_72);  x_72 = None
    x_73 = x_66 + getattr_l__mod___blocks___4___drop_path_1;  x_66 = getattr_l__mod___blocks___4___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___5___norm1 = self.getattr_L__mod___blocks___5___norm1(x_73)
    transpose_11 = getattr_l__mod___blocks___5___norm1.transpose(1, 2);  getattr_l__mod___blocks___5___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_74 = self.getattr_L__mod___blocks___5___mlp_tokens_fc1(transpose_11);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_10 = x_74.chunk(2, dim = -1);  x_74 = None
    x1_10 = chunk_10[0]
    x2_10 = chunk_10[1];  chunk_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___5___mlp_tokens_act = self.getattr_L__mod___blocks___5___mlp_tokens_act(x2_10);  x2_10 = None
    x_75 = x1_10 * getattr_l__mod___blocks___5___mlp_tokens_act;  x1_10 = getattr_l__mod___blocks___5___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_76 = self.getattr_L__mod___blocks___5___mlp_tokens_drop1(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_77 = self.getattr_L__mod___blocks___5___mlp_tokens_norm(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_78 = self.getattr_L__mod___blocks___5___mlp_tokens_fc2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_79 = self.getattr_L__mod___blocks___5___mlp_tokens_drop2(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_12 = x_79.transpose(1, 2);  x_79 = None
    getattr_l__mod___blocks___5___drop_path = self.getattr_L__mod___blocks___5___drop_path(transpose_12);  transpose_12 = None
    x_80 = x_73 + getattr_l__mod___blocks___5___drop_path;  x_73 = getattr_l__mod___blocks___5___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___5___norm2 = self.getattr_L__mod___blocks___5___norm2(x_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_81 = self.getattr_L__mod___blocks___5___mlp_channels_fc1(getattr_l__mod___blocks___5___norm2);  getattr_l__mod___blocks___5___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_11 = x_81.chunk(2, dim = -1);  x_81 = None
    x1_11 = chunk_11[0]
    x2_11 = chunk_11[1];  chunk_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___5___mlp_channels_act = self.getattr_L__mod___blocks___5___mlp_channels_act(x2_11);  x2_11 = None
    x_82 = x1_11 * getattr_l__mod___blocks___5___mlp_channels_act;  x1_11 = getattr_l__mod___blocks___5___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_83 = self.getattr_L__mod___blocks___5___mlp_channels_drop1(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_84 = self.getattr_L__mod___blocks___5___mlp_channels_norm(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_85 = self.getattr_L__mod___blocks___5___mlp_channels_fc2(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_86 = self.getattr_L__mod___blocks___5___mlp_channels_drop2(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___5___drop_path_1 = self.getattr_L__mod___blocks___5___drop_path(x_86);  x_86 = None
    x_87 = x_80 + getattr_l__mod___blocks___5___drop_path_1;  x_80 = getattr_l__mod___blocks___5___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___6___norm1 = self.getattr_L__mod___blocks___6___norm1(x_87)
    transpose_13 = getattr_l__mod___blocks___6___norm1.transpose(1, 2);  getattr_l__mod___blocks___6___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_88 = self.getattr_L__mod___blocks___6___mlp_tokens_fc1(transpose_13);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_12 = x_88.chunk(2, dim = -1);  x_88 = None
    x1_12 = chunk_12[0]
    x2_12 = chunk_12[1];  chunk_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___6___mlp_tokens_act = self.getattr_L__mod___blocks___6___mlp_tokens_act(x2_12);  x2_12 = None
    x_89 = x1_12 * getattr_l__mod___blocks___6___mlp_tokens_act;  x1_12 = getattr_l__mod___blocks___6___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_90 = self.getattr_L__mod___blocks___6___mlp_tokens_drop1(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_91 = self.getattr_L__mod___blocks___6___mlp_tokens_norm(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_92 = self.getattr_L__mod___blocks___6___mlp_tokens_fc2(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_93 = self.getattr_L__mod___blocks___6___mlp_tokens_drop2(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_14 = x_93.transpose(1, 2);  x_93 = None
    getattr_l__mod___blocks___6___drop_path = self.getattr_L__mod___blocks___6___drop_path(transpose_14);  transpose_14 = None
    x_94 = x_87 + getattr_l__mod___blocks___6___drop_path;  x_87 = getattr_l__mod___blocks___6___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___6___norm2 = self.getattr_L__mod___blocks___6___norm2(x_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_95 = self.getattr_L__mod___blocks___6___mlp_channels_fc1(getattr_l__mod___blocks___6___norm2);  getattr_l__mod___blocks___6___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_13 = x_95.chunk(2, dim = -1);  x_95 = None
    x1_13 = chunk_13[0]
    x2_13 = chunk_13[1];  chunk_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___6___mlp_channels_act = self.getattr_L__mod___blocks___6___mlp_channels_act(x2_13);  x2_13 = None
    x_96 = x1_13 * getattr_l__mod___blocks___6___mlp_channels_act;  x1_13 = getattr_l__mod___blocks___6___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_97 = self.getattr_L__mod___blocks___6___mlp_channels_drop1(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_98 = self.getattr_L__mod___blocks___6___mlp_channels_norm(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_99 = self.getattr_L__mod___blocks___6___mlp_channels_fc2(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_100 = self.getattr_L__mod___blocks___6___mlp_channels_drop2(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___6___drop_path_1 = self.getattr_L__mod___blocks___6___drop_path(x_100);  x_100 = None
    x_101 = x_94 + getattr_l__mod___blocks___6___drop_path_1;  x_94 = getattr_l__mod___blocks___6___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___7___norm1 = self.getattr_L__mod___blocks___7___norm1(x_101)
    transpose_15 = getattr_l__mod___blocks___7___norm1.transpose(1, 2);  getattr_l__mod___blocks___7___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_102 = self.getattr_L__mod___blocks___7___mlp_tokens_fc1(transpose_15);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_14 = x_102.chunk(2, dim = -1);  x_102 = None
    x1_14 = chunk_14[0]
    x2_14 = chunk_14[1];  chunk_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___7___mlp_tokens_act = self.getattr_L__mod___blocks___7___mlp_tokens_act(x2_14);  x2_14 = None
    x_103 = x1_14 * getattr_l__mod___blocks___7___mlp_tokens_act;  x1_14 = getattr_l__mod___blocks___7___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_104 = self.getattr_L__mod___blocks___7___mlp_tokens_drop1(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_105 = self.getattr_L__mod___blocks___7___mlp_tokens_norm(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_106 = self.getattr_L__mod___blocks___7___mlp_tokens_fc2(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_107 = self.getattr_L__mod___blocks___7___mlp_tokens_drop2(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_16 = x_107.transpose(1, 2);  x_107 = None
    getattr_l__mod___blocks___7___drop_path = self.getattr_L__mod___blocks___7___drop_path(transpose_16);  transpose_16 = None
    x_108 = x_101 + getattr_l__mod___blocks___7___drop_path;  x_101 = getattr_l__mod___blocks___7___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___7___norm2 = self.getattr_L__mod___blocks___7___norm2(x_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_109 = self.getattr_L__mod___blocks___7___mlp_channels_fc1(getattr_l__mod___blocks___7___norm2);  getattr_l__mod___blocks___7___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_15 = x_109.chunk(2, dim = -1);  x_109 = None
    x1_15 = chunk_15[0]
    x2_15 = chunk_15[1];  chunk_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___7___mlp_channels_act = self.getattr_L__mod___blocks___7___mlp_channels_act(x2_15);  x2_15 = None
    x_110 = x1_15 * getattr_l__mod___blocks___7___mlp_channels_act;  x1_15 = getattr_l__mod___blocks___7___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_111 = self.getattr_L__mod___blocks___7___mlp_channels_drop1(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_112 = self.getattr_L__mod___blocks___7___mlp_channels_norm(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_113 = self.getattr_L__mod___blocks___7___mlp_channels_fc2(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_114 = self.getattr_L__mod___blocks___7___mlp_channels_drop2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___7___drop_path_1 = self.getattr_L__mod___blocks___7___drop_path(x_114);  x_114 = None
    x_115 = x_108 + getattr_l__mod___blocks___7___drop_path_1;  x_108 = getattr_l__mod___blocks___7___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___8___norm1 = self.getattr_L__mod___blocks___8___norm1(x_115)
    transpose_17 = getattr_l__mod___blocks___8___norm1.transpose(1, 2);  getattr_l__mod___blocks___8___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_116 = self.getattr_L__mod___blocks___8___mlp_tokens_fc1(transpose_17);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_16 = x_116.chunk(2, dim = -1);  x_116 = None
    x1_16 = chunk_16[0]
    x2_16 = chunk_16[1];  chunk_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___8___mlp_tokens_act = self.getattr_L__mod___blocks___8___mlp_tokens_act(x2_16);  x2_16 = None
    x_117 = x1_16 * getattr_l__mod___blocks___8___mlp_tokens_act;  x1_16 = getattr_l__mod___blocks___8___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_118 = self.getattr_L__mod___blocks___8___mlp_tokens_drop1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_119 = self.getattr_L__mod___blocks___8___mlp_tokens_norm(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_120 = self.getattr_L__mod___blocks___8___mlp_tokens_fc2(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_121 = self.getattr_L__mod___blocks___8___mlp_tokens_drop2(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_18 = x_121.transpose(1, 2);  x_121 = None
    getattr_l__mod___blocks___8___drop_path = self.getattr_L__mod___blocks___8___drop_path(transpose_18);  transpose_18 = None
    x_122 = x_115 + getattr_l__mod___blocks___8___drop_path;  x_115 = getattr_l__mod___blocks___8___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___8___norm2 = self.getattr_L__mod___blocks___8___norm2(x_122)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_123 = self.getattr_L__mod___blocks___8___mlp_channels_fc1(getattr_l__mod___blocks___8___norm2);  getattr_l__mod___blocks___8___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_17 = x_123.chunk(2, dim = -1);  x_123 = None
    x1_17 = chunk_17[0]
    x2_17 = chunk_17[1];  chunk_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___8___mlp_channels_act = self.getattr_L__mod___blocks___8___mlp_channels_act(x2_17);  x2_17 = None
    x_124 = x1_17 * getattr_l__mod___blocks___8___mlp_channels_act;  x1_17 = getattr_l__mod___blocks___8___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_125 = self.getattr_L__mod___blocks___8___mlp_channels_drop1(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_126 = self.getattr_L__mod___blocks___8___mlp_channels_norm(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_127 = self.getattr_L__mod___blocks___8___mlp_channels_fc2(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_128 = self.getattr_L__mod___blocks___8___mlp_channels_drop2(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___8___drop_path_1 = self.getattr_L__mod___blocks___8___drop_path(x_128);  x_128 = None
    x_129 = x_122 + getattr_l__mod___blocks___8___drop_path_1;  x_122 = getattr_l__mod___blocks___8___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___9___norm1 = self.getattr_L__mod___blocks___9___norm1(x_129)
    transpose_19 = getattr_l__mod___blocks___9___norm1.transpose(1, 2);  getattr_l__mod___blocks___9___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_130 = self.getattr_L__mod___blocks___9___mlp_tokens_fc1(transpose_19);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_18 = x_130.chunk(2, dim = -1);  x_130 = None
    x1_18 = chunk_18[0]
    x2_18 = chunk_18[1];  chunk_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___9___mlp_tokens_act = self.getattr_L__mod___blocks___9___mlp_tokens_act(x2_18);  x2_18 = None
    x_131 = x1_18 * getattr_l__mod___blocks___9___mlp_tokens_act;  x1_18 = getattr_l__mod___blocks___9___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_132 = self.getattr_L__mod___blocks___9___mlp_tokens_drop1(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_133 = self.getattr_L__mod___blocks___9___mlp_tokens_norm(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_134 = self.getattr_L__mod___blocks___9___mlp_tokens_fc2(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_135 = self.getattr_L__mod___blocks___9___mlp_tokens_drop2(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_20 = x_135.transpose(1, 2);  x_135 = None
    getattr_l__mod___blocks___9___drop_path = self.getattr_L__mod___blocks___9___drop_path(transpose_20);  transpose_20 = None
    x_136 = x_129 + getattr_l__mod___blocks___9___drop_path;  x_129 = getattr_l__mod___blocks___9___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___9___norm2 = self.getattr_L__mod___blocks___9___norm2(x_136)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_137 = self.getattr_L__mod___blocks___9___mlp_channels_fc1(getattr_l__mod___blocks___9___norm2);  getattr_l__mod___blocks___9___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_19 = x_137.chunk(2, dim = -1);  x_137 = None
    x1_19 = chunk_19[0]
    x2_19 = chunk_19[1];  chunk_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___9___mlp_channels_act = self.getattr_L__mod___blocks___9___mlp_channels_act(x2_19);  x2_19 = None
    x_138 = x1_19 * getattr_l__mod___blocks___9___mlp_channels_act;  x1_19 = getattr_l__mod___blocks___9___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_139 = self.getattr_L__mod___blocks___9___mlp_channels_drop1(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_140 = self.getattr_L__mod___blocks___9___mlp_channels_norm(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_141 = self.getattr_L__mod___blocks___9___mlp_channels_fc2(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_142 = self.getattr_L__mod___blocks___9___mlp_channels_drop2(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___9___drop_path_1 = self.getattr_L__mod___blocks___9___drop_path(x_142);  x_142 = None
    x_143 = x_136 + getattr_l__mod___blocks___9___drop_path_1;  x_136 = getattr_l__mod___blocks___9___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___10___norm1 = self.getattr_L__mod___blocks___10___norm1(x_143)
    transpose_21 = getattr_l__mod___blocks___10___norm1.transpose(1, 2);  getattr_l__mod___blocks___10___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_144 = self.getattr_L__mod___blocks___10___mlp_tokens_fc1(transpose_21);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_20 = x_144.chunk(2, dim = -1);  x_144 = None
    x1_20 = chunk_20[0]
    x2_20 = chunk_20[1];  chunk_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___10___mlp_tokens_act = self.getattr_L__mod___blocks___10___mlp_tokens_act(x2_20);  x2_20 = None
    x_145 = x1_20 * getattr_l__mod___blocks___10___mlp_tokens_act;  x1_20 = getattr_l__mod___blocks___10___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_146 = self.getattr_L__mod___blocks___10___mlp_tokens_drop1(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_147 = self.getattr_L__mod___blocks___10___mlp_tokens_norm(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_148 = self.getattr_L__mod___blocks___10___mlp_tokens_fc2(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_149 = self.getattr_L__mod___blocks___10___mlp_tokens_drop2(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_22 = x_149.transpose(1, 2);  x_149 = None
    getattr_l__mod___blocks___10___drop_path = self.getattr_L__mod___blocks___10___drop_path(transpose_22);  transpose_22 = None
    x_150 = x_143 + getattr_l__mod___blocks___10___drop_path;  x_143 = getattr_l__mod___blocks___10___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___10___norm2 = self.getattr_L__mod___blocks___10___norm2(x_150)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_151 = self.getattr_L__mod___blocks___10___mlp_channels_fc1(getattr_l__mod___blocks___10___norm2);  getattr_l__mod___blocks___10___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_21 = x_151.chunk(2, dim = -1);  x_151 = None
    x1_21 = chunk_21[0]
    x2_21 = chunk_21[1];  chunk_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___10___mlp_channels_act = self.getattr_L__mod___blocks___10___mlp_channels_act(x2_21);  x2_21 = None
    x_152 = x1_21 * getattr_l__mod___blocks___10___mlp_channels_act;  x1_21 = getattr_l__mod___blocks___10___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_153 = self.getattr_L__mod___blocks___10___mlp_channels_drop1(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_154 = self.getattr_L__mod___blocks___10___mlp_channels_norm(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_155 = self.getattr_L__mod___blocks___10___mlp_channels_fc2(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_156 = self.getattr_L__mod___blocks___10___mlp_channels_drop2(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___10___drop_path_1 = self.getattr_L__mod___blocks___10___drop_path(x_156);  x_156 = None
    x_157 = x_150 + getattr_l__mod___blocks___10___drop_path_1;  x_150 = getattr_l__mod___blocks___10___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___11___norm1 = self.getattr_L__mod___blocks___11___norm1(x_157)
    transpose_23 = getattr_l__mod___blocks___11___norm1.transpose(1, 2);  getattr_l__mod___blocks___11___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_158 = self.getattr_L__mod___blocks___11___mlp_tokens_fc1(transpose_23);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_22 = x_158.chunk(2, dim = -1);  x_158 = None
    x1_22 = chunk_22[0]
    x2_22 = chunk_22[1];  chunk_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___11___mlp_tokens_act = self.getattr_L__mod___blocks___11___mlp_tokens_act(x2_22);  x2_22 = None
    x_159 = x1_22 * getattr_l__mod___blocks___11___mlp_tokens_act;  x1_22 = getattr_l__mod___blocks___11___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_160 = self.getattr_L__mod___blocks___11___mlp_tokens_drop1(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_161 = self.getattr_L__mod___blocks___11___mlp_tokens_norm(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_162 = self.getattr_L__mod___blocks___11___mlp_tokens_fc2(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_163 = self.getattr_L__mod___blocks___11___mlp_tokens_drop2(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_24 = x_163.transpose(1, 2);  x_163 = None
    getattr_l__mod___blocks___11___drop_path = self.getattr_L__mod___blocks___11___drop_path(transpose_24);  transpose_24 = None
    x_164 = x_157 + getattr_l__mod___blocks___11___drop_path;  x_157 = getattr_l__mod___blocks___11___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___11___norm2 = self.getattr_L__mod___blocks___11___norm2(x_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_165 = self.getattr_L__mod___blocks___11___mlp_channels_fc1(getattr_l__mod___blocks___11___norm2);  getattr_l__mod___blocks___11___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_23 = x_165.chunk(2, dim = -1);  x_165 = None
    x1_23 = chunk_23[0]
    x2_23 = chunk_23[1];  chunk_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___11___mlp_channels_act = self.getattr_L__mod___blocks___11___mlp_channels_act(x2_23);  x2_23 = None
    x_166 = x1_23 * getattr_l__mod___blocks___11___mlp_channels_act;  x1_23 = getattr_l__mod___blocks___11___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_167 = self.getattr_L__mod___blocks___11___mlp_channels_drop1(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_168 = self.getattr_L__mod___blocks___11___mlp_channels_norm(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_169 = self.getattr_L__mod___blocks___11___mlp_channels_fc2(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_170 = self.getattr_L__mod___blocks___11___mlp_channels_drop2(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___11___drop_path_1 = self.getattr_L__mod___blocks___11___drop_path(x_170);  x_170 = None
    x_171 = x_164 + getattr_l__mod___blocks___11___drop_path_1;  x_164 = getattr_l__mod___blocks___11___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___12___norm1 = self.getattr_L__mod___blocks___12___norm1(x_171)
    transpose_25 = getattr_l__mod___blocks___12___norm1.transpose(1, 2);  getattr_l__mod___blocks___12___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_172 = self.getattr_L__mod___blocks___12___mlp_tokens_fc1(transpose_25);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_24 = x_172.chunk(2, dim = -1);  x_172 = None
    x1_24 = chunk_24[0]
    x2_24 = chunk_24[1];  chunk_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___12___mlp_tokens_act = self.getattr_L__mod___blocks___12___mlp_tokens_act(x2_24);  x2_24 = None
    x_173 = x1_24 * getattr_l__mod___blocks___12___mlp_tokens_act;  x1_24 = getattr_l__mod___blocks___12___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_174 = self.getattr_L__mod___blocks___12___mlp_tokens_drop1(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_175 = self.getattr_L__mod___blocks___12___mlp_tokens_norm(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_176 = self.getattr_L__mod___blocks___12___mlp_tokens_fc2(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_177 = self.getattr_L__mod___blocks___12___mlp_tokens_drop2(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_26 = x_177.transpose(1, 2);  x_177 = None
    getattr_l__mod___blocks___12___drop_path = self.getattr_L__mod___blocks___12___drop_path(transpose_26);  transpose_26 = None
    x_178 = x_171 + getattr_l__mod___blocks___12___drop_path;  x_171 = getattr_l__mod___blocks___12___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___12___norm2 = self.getattr_L__mod___blocks___12___norm2(x_178)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_179 = self.getattr_L__mod___blocks___12___mlp_channels_fc1(getattr_l__mod___blocks___12___norm2);  getattr_l__mod___blocks___12___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_25 = x_179.chunk(2, dim = -1);  x_179 = None
    x1_25 = chunk_25[0]
    x2_25 = chunk_25[1];  chunk_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___12___mlp_channels_act = self.getattr_L__mod___blocks___12___mlp_channels_act(x2_25);  x2_25 = None
    x_180 = x1_25 * getattr_l__mod___blocks___12___mlp_channels_act;  x1_25 = getattr_l__mod___blocks___12___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_181 = self.getattr_L__mod___blocks___12___mlp_channels_drop1(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_182 = self.getattr_L__mod___blocks___12___mlp_channels_norm(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_183 = self.getattr_L__mod___blocks___12___mlp_channels_fc2(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_184 = self.getattr_L__mod___blocks___12___mlp_channels_drop2(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___12___drop_path_1 = self.getattr_L__mod___blocks___12___drop_path(x_184);  x_184 = None
    x_185 = x_178 + getattr_l__mod___blocks___12___drop_path_1;  x_178 = getattr_l__mod___blocks___12___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___13___norm1 = self.getattr_L__mod___blocks___13___norm1(x_185)
    transpose_27 = getattr_l__mod___blocks___13___norm1.transpose(1, 2);  getattr_l__mod___blocks___13___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_186 = self.getattr_L__mod___blocks___13___mlp_tokens_fc1(transpose_27);  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_26 = x_186.chunk(2, dim = -1);  x_186 = None
    x1_26 = chunk_26[0]
    x2_26 = chunk_26[1];  chunk_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___13___mlp_tokens_act = self.getattr_L__mod___blocks___13___mlp_tokens_act(x2_26);  x2_26 = None
    x_187 = x1_26 * getattr_l__mod___blocks___13___mlp_tokens_act;  x1_26 = getattr_l__mod___blocks___13___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_188 = self.getattr_L__mod___blocks___13___mlp_tokens_drop1(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_189 = self.getattr_L__mod___blocks___13___mlp_tokens_norm(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_190 = self.getattr_L__mod___blocks___13___mlp_tokens_fc2(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_191 = self.getattr_L__mod___blocks___13___mlp_tokens_drop2(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_28 = x_191.transpose(1, 2);  x_191 = None
    getattr_l__mod___blocks___13___drop_path = self.getattr_L__mod___blocks___13___drop_path(transpose_28);  transpose_28 = None
    x_192 = x_185 + getattr_l__mod___blocks___13___drop_path;  x_185 = getattr_l__mod___blocks___13___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___13___norm2 = self.getattr_L__mod___blocks___13___norm2(x_192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_193 = self.getattr_L__mod___blocks___13___mlp_channels_fc1(getattr_l__mod___blocks___13___norm2);  getattr_l__mod___blocks___13___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_27 = x_193.chunk(2, dim = -1);  x_193 = None
    x1_27 = chunk_27[0]
    x2_27 = chunk_27[1];  chunk_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___13___mlp_channels_act = self.getattr_L__mod___blocks___13___mlp_channels_act(x2_27);  x2_27 = None
    x_194 = x1_27 * getattr_l__mod___blocks___13___mlp_channels_act;  x1_27 = getattr_l__mod___blocks___13___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_195 = self.getattr_L__mod___blocks___13___mlp_channels_drop1(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_196 = self.getattr_L__mod___blocks___13___mlp_channels_norm(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_197 = self.getattr_L__mod___blocks___13___mlp_channels_fc2(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_198 = self.getattr_L__mod___blocks___13___mlp_channels_drop2(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___13___drop_path_1 = self.getattr_L__mod___blocks___13___drop_path(x_198);  x_198 = None
    x_199 = x_192 + getattr_l__mod___blocks___13___drop_path_1;  x_192 = getattr_l__mod___blocks___13___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___14___norm1 = self.getattr_L__mod___blocks___14___norm1(x_199)
    transpose_29 = getattr_l__mod___blocks___14___norm1.transpose(1, 2);  getattr_l__mod___blocks___14___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_200 = self.getattr_L__mod___blocks___14___mlp_tokens_fc1(transpose_29);  transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_28 = x_200.chunk(2, dim = -1);  x_200 = None
    x1_28 = chunk_28[0]
    x2_28 = chunk_28[1];  chunk_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___14___mlp_tokens_act = self.getattr_L__mod___blocks___14___mlp_tokens_act(x2_28);  x2_28 = None
    x_201 = x1_28 * getattr_l__mod___blocks___14___mlp_tokens_act;  x1_28 = getattr_l__mod___blocks___14___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_202 = self.getattr_L__mod___blocks___14___mlp_tokens_drop1(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_203 = self.getattr_L__mod___blocks___14___mlp_tokens_norm(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_204 = self.getattr_L__mod___blocks___14___mlp_tokens_fc2(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_205 = self.getattr_L__mod___blocks___14___mlp_tokens_drop2(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_30 = x_205.transpose(1, 2);  x_205 = None
    getattr_l__mod___blocks___14___drop_path = self.getattr_L__mod___blocks___14___drop_path(transpose_30);  transpose_30 = None
    x_206 = x_199 + getattr_l__mod___blocks___14___drop_path;  x_199 = getattr_l__mod___blocks___14___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___14___norm2 = self.getattr_L__mod___blocks___14___norm2(x_206)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_207 = self.getattr_L__mod___blocks___14___mlp_channels_fc1(getattr_l__mod___blocks___14___norm2);  getattr_l__mod___blocks___14___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_29 = x_207.chunk(2, dim = -1);  x_207 = None
    x1_29 = chunk_29[0]
    x2_29 = chunk_29[1];  chunk_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___14___mlp_channels_act = self.getattr_L__mod___blocks___14___mlp_channels_act(x2_29);  x2_29 = None
    x_208 = x1_29 * getattr_l__mod___blocks___14___mlp_channels_act;  x1_29 = getattr_l__mod___blocks___14___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_209 = self.getattr_L__mod___blocks___14___mlp_channels_drop1(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_210 = self.getattr_L__mod___blocks___14___mlp_channels_norm(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_211 = self.getattr_L__mod___blocks___14___mlp_channels_fc2(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_212 = self.getattr_L__mod___blocks___14___mlp_channels_drop2(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___14___drop_path_1 = self.getattr_L__mod___blocks___14___drop_path(x_212);  x_212 = None
    x_213 = x_206 + getattr_l__mod___blocks___14___drop_path_1;  x_206 = getattr_l__mod___blocks___14___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___15___norm1 = self.getattr_L__mod___blocks___15___norm1(x_213)
    transpose_31 = getattr_l__mod___blocks___15___norm1.transpose(1, 2);  getattr_l__mod___blocks___15___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_214 = self.getattr_L__mod___blocks___15___mlp_tokens_fc1(transpose_31);  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_30 = x_214.chunk(2, dim = -1);  x_214 = None
    x1_30 = chunk_30[0]
    x2_30 = chunk_30[1];  chunk_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___15___mlp_tokens_act = self.getattr_L__mod___blocks___15___mlp_tokens_act(x2_30);  x2_30 = None
    x_215 = x1_30 * getattr_l__mod___blocks___15___mlp_tokens_act;  x1_30 = getattr_l__mod___blocks___15___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_216 = self.getattr_L__mod___blocks___15___mlp_tokens_drop1(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_217 = self.getattr_L__mod___blocks___15___mlp_tokens_norm(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_218 = self.getattr_L__mod___blocks___15___mlp_tokens_fc2(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_219 = self.getattr_L__mod___blocks___15___mlp_tokens_drop2(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_32 = x_219.transpose(1, 2);  x_219 = None
    getattr_l__mod___blocks___15___drop_path = self.getattr_L__mod___blocks___15___drop_path(transpose_32);  transpose_32 = None
    x_220 = x_213 + getattr_l__mod___blocks___15___drop_path;  x_213 = getattr_l__mod___blocks___15___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___15___norm2 = self.getattr_L__mod___blocks___15___norm2(x_220)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_221 = self.getattr_L__mod___blocks___15___mlp_channels_fc1(getattr_l__mod___blocks___15___norm2);  getattr_l__mod___blocks___15___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_31 = x_221.chunk(2, dim = -1);  x_221 = None
    x1_31 = chunk_31[0]
    x2_31 = chunk_31[1];  chunk_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___15___mlp_channels_act = self.getattr_L__mod___blocks___15___mlp_channels_act(x2_31);  x2_31 = None
    x_222 = x1_31 * getattr_l__mod___blocks___15___mlp_channels_act;  x1_31 = getattr_l__mod___blocks___15___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_223 = self.getattr_L__mod___blocks___15___mlp_channels_drop1(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_224 = self.getattr_L__mod___blocks___15___mlp_channels_norm(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_225 = self.getattr_L__mod___blocks___15___mlp_channels_fc2(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_226 = self.getattr_L__mod___blocks___15___mlp_channels_drop2(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___15___drop_path_1 = self.getattr_L__mod___blocks___15___drop_path(x_226);  x_226 = None
    x_227 = x_220 + getattr_l__mod___blocks___15___drop_path_1;  x_220 = getattr_l__mod___blocks___15___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___16___norm1 = self.getattr_L__mod___blocks___16___norm1(x_227)
    transpose_33 = getattr_l__mod___blocks___16___norm1.transpose(1, 2);  getattr_l__mod___blocks___16___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_228 = self.getattr_L__mod___blocks___16___mlp_tokens_fc1(transpose_33);  transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_32 = x_228.chunk(2, dim = -1);  x_228 = None
    x1_32 = chunk_32[0]
    x2_32 = chunk_32[1];  chunk_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___16___mlp_tokens_act = self.getattr_L__mod___blocks___16___mlp_tokens_act(x2_32);  x2_32 = None
    x_229 = x1_32 * getattr_l__mod___blocks___16___mlp_tokens_act;  x1_32 = getattr_l__mod___blocks___16___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_230 = self.getattr_L__mod___blocks___16___mlp_tokens_drop1(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_231 = self.getattr_L__mod___blocks___16___mlp_tokens_norm(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_232 = self.getattr_L__mod___blocks___16___mlp_tokens_fc2(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_233 = self.getattr_L__mod___blocks___16___mlp_tokens_drop2(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_34 = x_233.transpose(1, 2);  x_233 = None
    getattr_l__mod___blocks___16___drop_path = self.getattr_L__mod___blocks___16___drop_path(transpose_34);  transpose_34 = None
    x_234 = x_227 + getattr_l__mod___blocks___16___drop_path;  x_227 = getattr_l__mod___blocks___16___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___16___norm2 = self.getattr_L__mod___blocks___16___norm2(x_234)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_235 = self.getattr_L__mod___blocks___16___mlp_channels_fc1(getattr_l__mod___blocks___16___norm2);  getattr_l__mod___blocks___16___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_33 = x_235.chunk(2, dim = -1);  x_235 = None
    x1_33 = chunk_33[0]
    x2_33 = chunk_33[1];  chunk_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___16___mlp_channels_act = self.getattr_L__mod___blocks___16___mlp_channels_act(x2_33);  x2_33 = None
    x_236 = x1_33 * getattr_l__mod___blocks___16___mlp_channels_act;  x1_33 = getattr_l__mod___blocks___16___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_237 = self.getattr_L__mod___blocks___16___mlp_channels_drop1(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_238 = self.getattr_L__mod___blocks___16___mlp_channels_norm(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_239 = self.getattr_L__mod___blocks___16___mlp_channels_fc2(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_240 = self.getattr_L__mod___blocks___16___mlp_channels_drop2(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___16___drop_path_1 = self.getattr_L__mod___blocks___16___drop_path(x_240);  x_240 = None
    x_241 = x_234 + getattr_l__mod___blocks___16___drop_path_1;  x_234 = getattr_l__mod___blocks___16___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___17___norm1 = self.getattr_L__mod___blocks___17___norm1(x_241)
    transpose_35 = getattr_l__mod___blocks___17___norm1.transpose(1, 2);  getattr_l__mod___blocks___17___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_242 = self.getattr_L__mod___blocks___17___mlp_tokens_fc1(transpose_35);  transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_34 = x_242.chunk(2, dim = -1);  x_242 = None
    x1_34 = chunk_34[0]
    x2_34 = chunk_34[1];  chunk_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___17___mlp_tokens_act = self.getattr_L__mod___blocks___17___mlp_tokens_act(x2_34);  x2_34 = None
    x_243 = x1_34 * getattr_l__mod___blocks___17___mlp_tokens_act;  x1_34 = getattr_l__mod___blocks___17___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_244 = self.getattr_L__mod___blocks___17___mlp_tokens_drop1(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_245 = self.getattr_L__mod___blocks___17___mlp_tokens_norm(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_246 = self.getattr_L__mod___blocks___17___mlp_tokens_fc2(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_247 = self.getattr_L__mod___blocks___17___mlp_tokens_drop2(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_36 = x_247.transpose(1, 2);  x_247 = None
    getattr_l__mod___blocks___17___drop_path = self.getattr_L__mod___blocks___17___drop_path(transpose_36);  transpose_36 = None
    x_248 = x_241 + getattr_l__mod___blocks___17___drop_path;  x_241 = getattr_l__mod___blocks___17___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___17___norm2 = self.getattr_L__mod___blocks___17___norm2(x_248)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_249 = self.getattr_L__mod___blocks___17___mlp_channels_fc1(getattr_l__mod___blocks___17___norm2);  getattr_l__mod___blocks___17___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_35 = x_249.chunk(2, dim = -1);  x_249 = None
    x1_35 = chunk_35[0]
    x2_35 = chunk_35[1];  chunk_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___17___mlp_channels_act = self.getattr_L__mod___blocks___17___mlp_channels_act(x2_35);  x2_35 = None
    x_250 = x1_35 * getattr_l__mod___blocks___17___mlp_channels_act;  x1_35 = getattr_l__mod___blocks___17___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_251 = self.getattr_L__mod___blocks___17___mlp_channels_drop1(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_252 = self.getattr_L__mod___blocks___17___mlp_channels_norm(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_253 = self.getattr_L__mod___blocks___17___mlp_channels_fc2(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_254 = self.getattr_L__mod___blocks___17___mlp_channels_drop2(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___17___drop_path_1 = self.getattr_L__mod___blocks___17___drop_path(x_254);  x_254 = None
    x_255 = x_248 + getattr_l__mod___blocks___17___drop_path_1;  x_248 = getattr_l__mod___blocks___17___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___18___norm1 = self.getattr_L__mod___blocks___18___norm1(x_255)
    transpose_37 = getattr_l__mod___blocks___18___norm1.transpose(1, 2);  getattr_l__mod___blocks___18___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_256 = self.getattr_L__mod___blocks___18___mlp_tokens_fc1(transpose_37);  transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_36 = x_256.chunk(2, dim = -1);  x_256 = None
    x1_36 = chunk_36[0]
    x2_36 = chunk_36[1];  chunk_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___18___mlp_tokens_act = self.getattr_L__mod___blocks___18___mlp_tokens_act(x2_36);  x2_36 = None
    x_257 = x1_36 * getattr_l__mod___blocks___18___mlp_tokens_act;  x1_36 = getattr_l__mod___blocks___18___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_258 = self.getattr_L__mod___blocks___18___mlp_tokens_drop1(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_259 = self.getattr_L__mod___blocks___18___mlp_tokens_norm(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_260 = self.getattr_L__mod___blocks___18___mlp_tokens_fc2(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_261 = self.getattr_L__mod___blocks___18___mlp_tokens_drop2(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_38 = x_261.transpose(1, 2);  x_261 = None
    getattr_l__mod___blocks___18___drop_path = self.getattr_L__mod___blocks___18___drop_path(transpose_38);  transpose_38 = None
    x_262 = x_255 + getattr_l__mod___blocks___18___drop_path;  x_255 = getattr_l__mod___blocks___18___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___18___norm2 = self.getattr_L__mod___blocks___18___norm2(x_262)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_263 = self.getattr_L__mod___blocks___18___mlp_channels_fc1(getattr_l__mod___blocks___18___norm2);  getattr_l__mod___blocks___18___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_37 = x_263.chunk(2, dim = -1);  x_263 = None
    x1_37 = chunk_37[0]
    x2_37 = chunk_37[1];  chunk_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___18___mlp_channels_act = self.getattr_L__mod___blocks___18___mlp_channels_act(x2_37);  x2_37 = None
    x_264 = x1_37 * getattr_l__mod___blocks___18___mlp_channels_act;  x1_37 = getattr_l__mod___blocks___18___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_265 = self.getattr_L__mod___blocks___18___mlp_channels_drop1(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_266 = self.getattr_L__mod___blocks___18___mlp_channels_norm(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_267 = self.getattr_L__mod___blocks___18___mlp_channels_fc2(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_268 = self.getattr_L__mod___blocks___18___mlp_channels_drop2(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___18___drop_path_1 = self.getattr_L__mod___blocks___18___drop_path(x_268);  x_268 = None
    x_269 = x_262 + getattr_l__mod___blocks___18___drop_path_1;  x_262 = getattr_l__mod___blocks___18___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___19___norm1 = self.getattr_L__mod___blocks___19___norm1(x_269)
    transpose_39 = getattr_l__mod___blocks___19___norm1.transpose(1, 2);  getattr_l__mod___blocks___19___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_270 = self.getattr_L__mod___blocks___19___mlp_tokens_fc1(transpose_39);  transpose_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_38 = x_270.chunk(2, dim = -1);  x_270 = None
    x1_38 = chunk_38[0]
    x2_38 = chunk_38[1];  chunk_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___19___mlp_tokens_act = self.getattr_L__mod___blocks___19___mlp_tokens_act(x2_38);  x2_38 = None
    x_271 = x1_38 * getattr_l__mod___blocks___19___mlp_tokens_act;  x1_38 = getattr_l__mod___blocks___19___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_272 = self.getattr_L__mod___blocks___19___mlp_tokens_drop1(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_273 = self.getattr_L__mod___blocks___19___mlp_tokens_norm(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_274 = self.getattr_L__mod___blocks___19___mlp_tokens_fc2(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_275 = self.getattr_L__mod___blocks___19___mlp_tokens_drop2(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_40 = x_275.transpose(1, 2);  x_275 = None
    getattr_l__mod___blocks___19___drop_path = self.getattr_L__mod___blocks___19___drop_path(transpose_40);  transpose_40 = None
    x_276 = x_269 + getattr_l__mod___blocks___19___drop_path;  x_269 = getattr_l__mod___blocks___19___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___19___norm2 = self.getattr_L__mod___blocks___19___norm2(x_276)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_277 = self.getattr_L__mod___blocks___19___mlp_channels_fc1(getattr_l__mod___blocks___19___norm2);  getattr_l__mod___blocks___19___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_39 = x_277.chunk(2, dim = -1);  x_277 = None
    x1_39 = chunk_39[0]
    x2_39 = chunk_39[1];  chunk_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___19___mlp_channels_act = self.getattr_L__mod___blocks___19___mlp_channels_act(x2_39);  x2_39 = None
    x_278 = x1_39 * getattr_l__mod___blocks___19___mlp_channels_act;  x1_39 = getattr_l__mod___blocks___19___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_279 = self.getattr_L__mod___blocks___19___mlp_channels_drop1(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_280 = self.getattr_L__mod___blocks___19___mlp_channels_norm(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_281 = self.getattr_L__mod___blocks___19___mlp_channels_fc2(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_282 = self.getattr_L__mod___blocks___19___mlp_channels_drop2(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___19___drop_path_1 = self.getattr_L__mod___blocks___19___drop_path(x_282);  x_282 = None
    x_283 = x_276 + getattr_l__mod___blocks___19___drop_path_1;  x_276 = getattr_l__mod___blocks___19___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___20___norm1 = self.getattr_L__mod___blocks___20___norm1(x_283)
    transpose_41 = getattr_l__mod___blocks___20___norm1.transpose(1, 2);  getattr_l__mod___blocks___20___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_284 = self.getattr_L__mod___blocks___20___mlp_tokens_fc1(transpose_41);  transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_40 = x_284.chunk(2, dim = -1);  x_284 = None
    x1_40 = chunk_40[0]
    x2_40 = chunk_40[1];  chunk_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___20___mlp_tokens_act = self.getattr_L__mod___blocks___20___mlp_tokens_act(x2_40);  x2_40 = None
    x_285 = x1_40 * getattr_l__mod___blocks___20___mlp_tokens_act;  x1_40 = getattr_l__mod___blocks___20___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_286 = self.getattr_L__mod___blocks___20___mlp_tokens_drop1(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_287 = self.getattr_L__mod___blocks___20___mlp_tokens_norm(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_288 = self.getattr_L__mod___blocks___20___mlp_tokens_fc2(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_289 = self.getattr_L__mod___blocks___20___mlp_tokens_drop2(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_42 = x_289.transpose(1, 2);  x_289 = None
    getattr_l__mod___blocks___20___drop_path = self.getattr_L__mod___blocks___20___drop_path(transpose_42);  transpose_42 = None
    x_290 = x_283 + getattr_l__mod___blocks___20___drop_path;  x_283 = getattr_l__mod___blocks___20___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___20___norm2 = self.getattr_L__mod___blocks___20___norm2(x_290)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_291 = self.getattr_L__mod___blocks___20___mlp_channels_fc1(getattr_l__mod___blocks___20___norm2);  getattr_l__mod___blocks___20___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_41 = x_291.chunk(2, dim = -1);  x_291 = None
    x1_41 = chunk_41[0]
    x2_41 = chunk_41[1];  chunk_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___20___mlp_channels_act = self.getattr_L__mod___blocks___20___mlp_channels_act(x2_41);  x2_41 = None
    x_292 = x1_41 * getattr_l__mod___blocks___20___mlp_channels_act;  x1_41 = getattr_l__mod___blocks___20___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_293 = self.getattr_L__mod___blocks___20___mlp_channels_drop1(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_294 = self.getattr_L__mod___blocks___20___mlp_channels_norm(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_295 = self.getattr_L__mod___blocks___20___mlp_channels_fc2(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_296 = self.getattr_L__mod___blocks___20___mlp_channels_drop2(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___20___drop_path_1 = self.getattr_L__mod___blocks___20___drop_path(x_296);  x_296 = None
    x_297 = x_290 + getattr_l__mod___blocks___20___drop_path_1;  x_290 = getattr_l__mod___blocks___20___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___21___norm1 = self.getattr_L__mod___blocks___21___norm1(x_297)
    transpose_43 = getattr_l__mod___blocks___21___norm1.transpose(1, 2);  getattr_l__mod___blocks___21___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_298 = self.getattr_L__mod___blocks___21___mlp_tokens_fc1(transpose_43);  transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_42 = x_298.chunk(2, dim = -1);  x_298 = None
    x1_42 = chunk_42[0]
    x2_42 = chunk_42[1];  chunk_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___21___mlp_tokens_act = self.getattr_L__mod___blocks___21___mlp_tokens_act(x2_42);  x2_42 = None
    x_299 = x1_42 * getattr_l__mod___blocks___21___mlp_tokens_act;  x1_42 = getattr_l__mod___blocks___21___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_300 = self.getattr_L__mod___blocks___21___mlp_tokens_drop1(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_301 = self.getattr_L__mod___blocks___21___mlp_tokens_norm(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_302 = self.getattr_L__mod___blocks___21___mlp_tokens_fc2(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_303 = self.getattr_L__mod___blocks___21___mlp_tokens_drop2(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_44 = x_303.transpose(1, 2);  x_303 = None
    getattr_l__mod___blocks___21___drop_path = self.getattr_L__mod___blocks___21___drop_path(transpose_44);  transpose_44 = None
    x_304 = x_297 + getattr_l__mod___blocks___21___drop_path;  x_297 = getattr_l__mod___blocks___21___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___21___norm2 = self.getattr_L__mod___blocks___21___norm2(x_304)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_305 = self.getattr_L__mod___blocks___21___mlp_channels_fc1(getattr_l__mod___blocks___21___norm2);  getattr_l__mod___blocks___21___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_43 = x_305.chunk(2, dim = -1);  x_305 = None
    x1_43 = chunk_43[0]
    x2_43 = chunk_43[1];  chunk_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___21___mlp_channels_act = self.getattr_L__mod___blocks___21___mlp_channels_act(x2_43);  x2_43 = None
    x_306 = x1_43 * getattr_l__mod___blocks___21___mlp_channels_act;  x1_43 = getattr_l__mod___blocks___21___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_307 = self.getattr_L__mod___blocks___21___mlp_channels_drop1(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_308 = self.getattr_L__mod___blocks___21___mlp_channels_norm(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_309 = self.getattr_L__mod___blocks___21___mlp_channels_fc2(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_310 = self.getattr_L__mod___blocks___21___mlp_channels_drop2(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___21___drop_path_1 = self.getattr_L__mod___blocks___21___drop_path(x_310);  x_310 = None
    x_311 = x_304 + getattr_l__mod___blocks___21___drop_path_1;  x_304 = getattr_l__mod___blocks___21___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___22___norm1 = self.getattr_L__mod___blocks___22___norm1(x_311)
    transpose_45 = getattr_l__mod___blocks___22___norm1.transpose(1, 2);  getattr_l__mod___blocks___22___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_312 = self.getattr_L__mod___blocks___22___mlp_tokens_fc1(transpose_45);  transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_44 = x_312.chunk(2, dim = -1);  x_312 = None
    x1_44 = chunk_44[0]
    x2_44 = chunk_44[1];  chunk_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___22___mlp_tokens_act = self.getattr_L__mod___blocks___22___mlp_tokens_act(x2_44);  x2_44 = None
    x_313 = x1_44 * getattr_l__mod___blocks___22___mlp_tokens_act;  x1_44 = getattr_l__mod___blocks___22___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_314 = self.getattr_L__mod___blocks___22___mlp_tokens_drop1(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_315 = self.getattr_L__mod___blocks___22___mlp_tokens_norm(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_316 = self.getattr_L__mod___blocks___22___mlp_tokens_fc2(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_317 = self.getattr_L__mod___blocks___22___mlp_tokens_drop2(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_46 = x_317.transpose(1, 2);  x_317 = None
    getattr_l__mod___blocks___22___drop_path = self.getattr_L__mod___blocks___22___drop_path(transpose_46);  transpose_46 = None
    x_318 = x_311 + getattr_l__mod___blocks___22___drop_path;  x_311 = getattr_l__mod___blocks___22___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___22___norm2 = self.getattr_L__mod___blocks___22___norm2(x_318)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_319 = self.getattr_L__mod___blocks___22___mlp_channels_fc1(getattr_l__mod___blocks___22___norm2);  getattr_l__mod___blocks___22___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_45 = x_319.chunk(2, dim = -1);  x_319 = None
    x1_45 = chunk_45[0]
    x2_45 = chunk_45[1];  chunk_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___22___mlp_channels_act = self.getattr_L__mod___blocks___22___mlp_channels_act(x2_45);  x2_45 = None
    x_320 = x1_45 * getattr_l__mod___blocks___22___mlp_channels_act;  x1_45 = getattr_l__mod___blocks___22___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_321 = self.getattr_L__mod___blocks___22___mlp_channels_drop1(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_322 = self.getattr_L__mod___blocks___22___mlp_channels_norm(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_323 = self.getattr_L__mod___blocks___22___mlp_channels_fc2(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_324 = self.getattr_L__mod___blocks___22___mlp_channels_drop2(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___22___drop_path_1 = self.getattr_L__mod___blocks___22___drop_path(x_324);  x_324 = None
    x_325 = x_318 + getattr_l__mod___blocks___22___drop_path_1;  x_318 = getattr_l__mod___blocks___22___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___23___norm1 = self.getattr_L__mod___blocks___23___norm1(x_325)
    transpose_47 = getattr_l__mod___blocks___23___norm1.transpose(1, 2);  getattr_l__mod___blocks___23___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_326 = self.getattr_L__mod___blocks___23___mlp_tokens_fc1(transpose_47);  transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_46 = x_326.chunk(2, dim = -1);  x_326 = None
    x1_46 = chunk_46[0]
    x2_46 = chunk_46[1];  chunk_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___23___mlp_tokens_act = self.getattr_L__mod___blocks___23___mlp_tokens_act(x2_46);  x2_46 = None
    x_327 = x1_46 * getattr_l__mod___blocks___23___mlp_tokens_act;  x1_46 = getattr_l__mod___blocks___23___mlp_tokens_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_328 = self.getattr_L__mod___blocks___23___mlp_tokens_drop1(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_329 = self.getattr_L__mod___blocks___23___mlp_tokens_norm(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_330 = self.getattr_L__mod___blocks___23___mlp_tokens_fc2(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_331 = self.getattr_L__mod___blocks___23___mlp_tokens_drop2(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_48 = x_331.transpose(1, 2);  x_331 = None
    getattr_l__mod___blocks___23___drop_path = self.getattr_L__mod___blocks___23___drop_path(transpose_48);  transpose_48 = None
    x_332 = x_325 + getattr_l__mod___blocks___23___drop_path;  x_325 = getattr_l__mod___blocks___23___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___23___norm2 = self.getattr_L__mod___blocks___23___norm2(x_332)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    x_333 = self.getattr_L__mod___blocks___23___mlp_channels_fc1(getattr_l__mod___blocks___23___norm2);  getattr_l__mod___blocks___23___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    chunk_47 = x_333.chunk(2, dim = -1);  x_333 = None
    x1_47 = chunk_47[0]
    x2_47 = chunk_47[1];  chunk_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    getattr_l__mod___blocks___23___mlp_channels_act = self.getattr_L__mod___blocks___23___mlp_channels_act(x2_47);  x2_47 = None
    x_334 = x1_47 * getattr_l__mod___blocks___23___mlp_channels_act;  x1_47 = getattr_l__mod___blocks___23___mlp_channels_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    x_335 = self.getattr_L__mod___blocks___23___mlp_channels_drop1(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:95, code: x = self.norm(x)
    x_336 = self.getattr_L__mod___blocks___23___mlp_channels_norm(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    x_337 = self.getattr_L__mod___blocks___23___mlp_channels_fc2(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    x_338 = self.getattr_L__mod___blocks___23___mlp_channels_drop2(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___23___drop_path_1 = self.getattr_L__mod___blocks___23___drop_path(x_338);  x_338 = None
    x_340 = x_332 + getattr_l__mod___blocks___23___drop_path_1;  x_332 = getattr_l__mod___blocks___23___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    x_342 = self.L__mod___norm(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    x_343 = x_342.mean(dim = 1);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    x_344 = self.L__mod___head_drop(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    x_345 = self.L__mod___head(x_344);  x_344 = None
    return (x_345,)
    