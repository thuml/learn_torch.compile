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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_4 = self.getattr_L__mod___blocks___0___mlp_tokens_fc1(transpose_1);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_5 = self.getattr_L__mod___blocks___0___mlp_tokens_act(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_6 = self.getattr_L__mod___blocks___0___mlp_tokens_drop1(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_7 = self.getattr_L__mod___blocks___0___mlp_tokens_norm(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_8 = self.getattr_L__mod___blocks___0___mlp_tokens_fc2(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_9 = self.getattr_L__mod___blocks___0___mlp_tokens_drop2(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_2 = x_9.transpose(1, 2);  x_9 = None
    getattr_l__mod___blocks___0___drop_path = self.getattr_L__mod___blocks___0___drop_path(transpose_2);  transpose_2 = None
    x_10 = x_3 + getattr_l__mod___blocks___0___drop_path;  x_3 = getattr_l__mod___blocks___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___0___norm2 = self.getattr_L__mod___blocks___0___norm2(x_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_11 = self.getattr_L__mod___blocks___0___mlp_channels_fc1(getattr_l__mod___blocks___0___norm2);  getattr_l__mod___blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_12 = self.getattr_L__mod___blocks___0___mlp_channels_act(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_13 = self.getattr_L__mod___blocks___0___mlp_channels_drop1(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_14 = self.getattr_L__mod___blocks___0___mlp_channels_norm(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_15 = self.getattr_L__mod___blocks___0___mlp_channels_fc2(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_16 = self.getattr_L__mod___blocks___0___mlp_channels_drop2(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___0___drop_path_1 = self.getattr_L__mod___blocks___0___drop_path(x_16);  x_16 = None
    x_17 = x_10 + getattr_l__mod___blocks___0___drop_path_1;  x_10 = getattr_l__mod___blocks___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___1___norm1 = self.getattr_L__mod___blocks___1___norm1(x_17)
    transpose_3 = getattr_l__mod___blocks___1___norm1.transpose(1, 2);  getattr_l__mod___blocks___1___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_18 = self.getattr_L__mod___blocks___1___mlp_tokens_fc1(transpose_3);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_19 = self.getattr_L__mod___blocks___1___mlp_tokens_act(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_20 = self.getattr_L__mod___blocks___1___mlp_tokens_drop1(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_21 = self.getattr_L__mod___blocks___1___mlp_tokens_norm(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_22 = self.getattr_L__mod___blocks___1___mlp_tokens_fc2(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_23 = self.getattr_L__mod___blocks___1___mlp_tokens_drop2(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_4 = x_23.transpose(1, 2);  x_23 = None
    getattr_l__mod___blocks___1___drop_path = self.getattr_L__mod___blocks___1___drop_path(transpose_4);  transpose_4 = None
    x_24 = x_17 + getattr_l__mod___blocks___1___drop_path;  x_17 = getattr_l__mod___blocks___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___1___norm2 = self.getattr_L__mod___blocks___1___norm2(x_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_25 = self.getattr_L__mod___blocks___1___mlp_channels_fc1(getattr_l__mod___blocks___1___norm2);  getattr_l__mod___blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_26 = self.getattr_L__mod___blocks___1___mlp_channels_act(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_27 = self.getattr_L__mod___blocks___1___mlp_channels_drop1(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_28 = self.getattr_L__mod___blocks___1___mlp_channels_norm(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_29 = self.getattr_L__mod___blocks___1___mlp_channels_fc2(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_30 = self.getattr_L__mod___blocks___1___mlp_channels_drop2(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___1___drop_path_1 = self.getattr_L__mod___blocks___1___drop_path(x_30);  x_30 = None
    x_31 = x_24 + getattr_l__mod___blocks___1___drop_path_1;  x_24 = getattr_l__mod___blocks___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___2___norm1 = self.getattr_L__mod___blocks___2___norm1(x_31)
    transpose_5 = getattr_l__mod___blocks___2___norm1.transpose(1, 2);  getattr_l__mod___blocks___2___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_32 = self.getattr_L__mod___blocks___2___mlp_tokens_fc1(transpose_5);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_33 = self.getattr_L__mod___blocks___2___mlp_tokens_act(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_34 = self.getattr_L__mod___blocks___2___mlp_tokens_drop1(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_35 = self.getattr_L__mod___blocks___2___mlp_tokens_norm(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_36 = self.getattr_L__mod___blocks___2___mlp_tokens_fc2(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_37 = self.getattr_L__mod___blocks___2___mlp_tokens_drop2(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_6 = x_37.transpose(1, 2);  x_37 = None
    getattr_l__mod___blocks___2___drop_path = self.getattr_L__mod___blocks___2___drop_path(transpose_6);  transpose_6 = None
    x_38 = x_31 + getattr_l__mod___blocks___2___drop_path;  x_31 = getattr_l__mod___blocks___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___2___norm2 = self.getattr_L__mod___blocks___2___norm2(x_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_39 = self.getattr_L__mod___blocks___2___mlp_channels_fc1(getattr_l__mod___blocks___2___norm2);  getattr_l__mod___blocks___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_40 = self.getattr_L__mod___blocks___2___mlp_channels_act(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_41 = self.getattr_L__mod___blocks___2___mlp_channels_drop1(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_42 = self.getattr_L__mod___blocks___2___mlp_channels_norm(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_43 = self.getattr_L__mod___blocks___2___mlp_channels_fc2(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_44 = self.getattr_L__mod___blocks___2___mlp_channels_drop2(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___2___drop_path_1 = self.getattr_L__mod___blocks___2___drop_path(x_44);  x_44 = None
    x_45 = x_38 + getattr_l__mod___blocks___2___drop_path_1;  x_38 = getattr_l__mod___blocks___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___3___norm1 = self.getattr_L__mod___blocks___3___norm1(x_45)
    transpose_7 = getattr_l__mod___blocks___3___norm1.transpose(1, 2);  getattr_l__mod___blocks___3___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_46 = self.getattr_L__mod___blocks___3___mlp_tokens_fc1(transpose_7);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_47 = self.getattr_L__mod___blocks___3___mlp_tokens_act(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_48 = self.getattr_L__mod___blocks___3___mlp_tokens_drop1(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_49 = self.getattr_L__mod___blocks___3___mlp_tokens_norm(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_50 = self.getattr_L__mod___blocks___3___mlp_tokens_fc2(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_51 = self.getattr_L__mod___blocks___3___mlp_tokens_drop2(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_8 = x_51.transpose(1, 2);  x_51 = None
    getattr_l__mod___blocks___3___drop_path = self.getattr_L__mod___blocks___3___drop_path(transpose_8);  transpose_8 = None
    x_52 = x_45 + getattr_l__mod___blocks___3___drop_path;  x_45 = getattr_l__mod___blocks___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___3___norm2 = self.getattr_L__mod___blocks___3___norm2(x_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_53 = self.getattr_L__mod___blocks___3___mlp_channels_fc1(getattr_l__mod___blocks___3___norm2);  getattr_l__mod___blocks___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_54 = self.getattr_L__mod___blocks___3___mlp_channels_act(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_55 = self.getattr_L__mod___blocks___3___mlp_channels_drop1(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_56 = self.getattr_L__mod___blocks___3___mlp_channels_norm(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_57 = self.getattr_L__mod___blocks___3___mlp_channels_fc2(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_58 = self.getattr_L__mod___blocks___3___mlp_channels_drop2(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___3___drop_path_1 = self.getattr_L__mod___blocks___3___drop_path(x_58);  x_58 = None
    x_59 = x_52 + getattr_l__mod___blocks___3___drop_path_1;  x_52 = getattr_l__mod___blocks___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___4___norm1 = self.getattr_L__mod___blocks___4___norm1(x_59)
    transpose_9 = getattr_l__mod___blocks___4___norm1.transpose(1, 2);  getattr_l__mod___blocks___4___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_60 = self.getattr_L__mod___blocks___4___mlp_tokens_fc1(transpose_9);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_61 = self.getattr_L__mod___blocks___4___mlp_tokens_act(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_62 = self.getattr_L__mod___blocks___4___mlp_tokens_drop1(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_63 = self.getattr_L__mod___blocks___4___mlp_tokens_norm(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_64 = self.getattr_L__mod___blocks___4___mlp_tokens_fc2(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_65 = self.getattr_L__mod___blocks___4___mlp_tokens_drop2(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_10 = x_65.transpose(1, 2);  x_65 = None
    getattr_l__mod___blocks___4___drop_path = self.getattr_L__mod___blocks___4___drop_path(transpose_10);  transpose_10 = None
    x_66 = x_59 + getattr_l__mod___blocks___4___drop_path;  x_59 = getattr_l__mod___blocks___4___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___4___norm2 = self.getattr_L__mod___blocks___4___norm2(x_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_67 = self.getattr_L__mod___blocks___4___mlp_channels_fc1(getattr_l__mod___blocks___4___norm2);  getattr_l__mod___blocks___4___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_68 = self.getattr_L__mod___blocks___4___mlp_channels_act(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_69 = self.getattr_L__mod___blocks___4___mlp_channels_drop1(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_70 = self.getattr_L__mod___blocks___4___mlp_channels_norm(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_71 = self.getattr_L__mod___blocks___4___mlp_channels_fc2(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_72 = self.getattr_L__mod___blocks___4___mlp_channels_drop2(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___4___drop_path_1 = self.getattr_L__mod___blocks___4___drop_path(x_72);  x_72 = None
    x_73 = x_66 + getattr_l__mod___blocks___4___drop_path_1;  x_66 = getattr_l__mod___blocks___4___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___5___norm1 = self.getattr_L__mod___blocks___5___norm1(x_73)
    transpose_11 = getattr_l__mod___blocks___5___norm1.transpose(1, 2);  getattr_l__mod___blocks___5___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_74 = self.getattr_L__mod___blocks___5___mlp_tokens_fc1(transpose_11);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_75 = self.getattr_L__mod___blocks___5___mlp_tokens_act(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_76 = self.getattr_L__mod___blocks___5___mlp_tokens_drop1(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_77 = self.getattr_L__mod___blocks___5___mlp_tokens_norm(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_78 = self.getattr_L__mod___blocks___5___mlp_tokens_fc2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_79 = self.getattr_L__mod___blocks___5___mlp_tokens_drop2(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_12 = x_79.transpose(1, 2);  x_79 = None
    getattr_l__mod___blocks___5___drop_path = self.getattr_L__mod___blocks___5___drop_path(transpose_12);  transpose_12 = None
    x_80 = x_73 + getattr_l__mod___blocks___5___drop_path;  x_73 = getattr_l__mod___blocks___5___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___5___norm2 = self.getattr_L__mod___blocks___5___norm2(x_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_81 = self.getattr_L__mod___blocks___5___mlp_channels_fc1(getattr_l__mod___blocks___5___norm2);  getattr_l__mod___blocks___5___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_82 = self.getattr_L__mod___blocks___5___mlp_channels_act(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_83 = self.getattr_L__mod___blocks___5___mlp_channels_drop1(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_84 = self.getattr_L__mod___blocks___5___mlp_channels_norm(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_85 = self.getattr_L__mod___blocks___5___mlp_channels_fc2(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_86 = self.getattr_L__mod___blocks___5___mlp_channels_drop2(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___5___drop_path_1 = self.getattr_L__mod___blocks___5___drop_path(x_86);  x_86 = None
    x_87 = x_80 + getattr_l__mod___blocks___5___drop_path_1;  x_80 = getattr_l__mod___blocks___5___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___6___norm1 = self.getattr_L__mod___blocks___6___norm1(x_87)
    transpose_13 = getattr_l__mod___blocks___6___norm1.transpose(1, 2);  getattr_l__mod___blocks___6___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_88 = self.getattr_L__mod___blocks___6___mlp_tokens_fc1(transpose_13);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_89 = self.getattr_L__mod___blocks___6___mlp_tokens_act(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_90 = self.getattr_L__mod___blocks___6___mlp_tokens_drop1(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_91 = self.getattr_L__mod___blocks___6___mlp_tokens_norm(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_92 = self.getattr_L__mod___blocks___6___mlp_tokens_fc2(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_93 = self.getattr_L__mod___blocks___6___mlp_tokens_drop2(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_14 = x_93.transpose(1, 2);  x_93 = None
    getattr_l__mod___blocks___6___drop_path = self.getattr_L__mod___blocks___6___drop_path(transpose_14);  transpose_14 = None
    x_94 = x_87 + getattr_l__mod___blocks___6___drop_path;  x_87 = getattr_l__mod___blocks___6___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___6___norm2 = self.getattr_L__mod___blocks___6___norm2(x_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_95 = self.getattr_L__mod___blocks___6___mlp_channels_fc1(getattr_l__mod___blocks___6___norm2);  getattr_l__mod___blocks___6___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_96 = self.getattr_L__mod___blocks___6___mlp_channels_act(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_97 = self.getattr_L__mod___blocks___6___mlp_channels_drop1(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_98 = self.getattr_L__mod___blocks___6___mlp_channels_norm(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_99 = self.getattr_L__mod___blocks___6___mlp_channels_fc2(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_100 = self.getattr_L__mod___blocks___6___mlp_channels_drop2(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___6___drop_path_1 = self.getattr_L__mod___blocks___6___drop_path(x_100);  x_100 = None
    x_101 = x_94 + getattr_l__mod___blocks___6___drop_path_1;  x_94 = getattr_l__mod___blocks___6___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___7___norm1 = self.getattr_L__mod___blocks___7___norm1(x_101)
    transpose_15 = getattr_l__mod___blocks___7___norm1.transpose(1, 2);  getattr_l__mod___blocks___7___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_102 = self.getattr_L__mod___blocks___7___mlp_tokens_fc1(transpose_15);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_103 = self.getattr_L__mod___blocks___7___mlp_tokens_act(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_104 = self.getattr_L__mod___blocks___7___mlp_tokens_drop1(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_105 = self.getattr_L__mod___blocks___7___mlp_tokens_norm(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_106 = self.getattr_L__mod___blocks___7___mlp_tokens_fc2(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_107 = self.getattr_L__mod___blocks___7___mlp_tokens_drop2(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_16 = x_107.transpose(1, 2);  x_107 = None
    getattr_l__mod___blocks___7___drop_path = self.getattr_L__mod___blocks___7___drop_path(transpose_16);  transpose_16 = None
    x_108 = x_101 + getattr_l__mod___blocks___7___drop_path;  x_101 = getattr_l__mod___blocks___7___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___7___norm2 = self.getattr_L__mod___blocks___7___norm2(x_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_109 = self.getattr_L__mod___blocks___7___mlp_channels_fc1(getattr_l__mod___blocks___7___norm2);  getattr_l__mod___blocks___7___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_110 = self.getattr_L__mod___blocks___7___mlp_channels_act(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_111 = self.getattr_L__mod___blocks___7___mlp_channels_drop1(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_112 = self.getattr_L__mod___blocks___7___mlp_channels_norm(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_113 = self.getattr_L__mod___blocks___7___mlp_channels_fc2(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_114 = self.getattr_L__mod___blocks___7___mlp_channels_drop2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___7___drop_path_1 = self.getattr_L__mod___blocks___7___drop_path(x_114);  x_114 = None
    x_115 = x_108 + getattr_l__mod___blocks___7___drop_path_1;  x_108 = getattr_l__mod___blocks___7___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___8___norm1 = self.getattr_L__mod___blocks___8___norm1(x_115)
    transpose_17 = getattr_l__mod___blocks___8___norm1.transpose(1, 2);  getattr_l__mod___blocks___8___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_116 = self.getattr_L__mod___blocks___8___mlp_tokens_fc1(transpose_17);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_117 = self.getattr_L__mod___blocks___8___mlp_tokens_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_118 = self.getattr_L__mod___blocks___8___mlp_tokens_drop1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_119 = self.getattr_L__mod___blocks___8___mlp_tokens_norm(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_120 = self.getattr_L__mod___blocks___8___mlp_tokens_fc2(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_121 = self.getattr_L__mod___blocks___8___mlp_tokens_drop2(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_18 = x_121.transpose(1, 2);  x_121 = None
    getattr_l__mod___blocks___8___drop_path = self.getattr_L__mod___blocks___8___drop_path(transpose_18);  transpose_18 = None
    x_122 = x_115 + getattr_l__mod___blocks___8___drop_path;  x_115 = getattr_l__mod___blocks___8___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___8___norm2 = self.getattr_L__mod___blocks___8___norm2(x_122)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_123 = self.getattr_L__mod___blocks___8___mlp_channels_fc1(getattr_l__mod___blocks___8___norm2);  getattr_l__mod___blocks___8___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_124 = self.getattr_L__mod___blocks___8___mlp_channels_act(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_125 = self.getattr_L__mod___blocks___8___mlp_channels_drop1(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_126 = self.getattr_L__mod___blocks___8___mlp_channels_norm(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_127 = self.getattr_L__mod___blocks___8___mlp_channels_fc2(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_128 = self.getattr_L__mod___blocks___8___mlp_channels_drop2(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___8___drop_path_1 = self.getattr_L__mod___blocks___8___drop_path(x_128);  x_128 = None
    x_129 = x_122 + getattr_l__mod___blocks___8___drop_path_1;  x_122 = getattr_l__mod___blocks___8___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___9___norm1 = self.getattr_L__mod___blocks___9___norm1(x_129)
    transpose_19 = getattr_l__mod___blocks___9___norm1.transpose(1, 2);  getattr_l__mod___blocks___9___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_130 = self.getattr_L__mod___blocks___9___mlp_tokens_fc1(transpose_19);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_131 = self.getattr_L__mod___blocks___9___mlp_tokens_act(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_132 = self.getattr_L__mod___blocks___9___mlp_tokens_drop1(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_133 = self.getattr_L__mod___blocks___9___mlp_tokens_norm(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_134 = self.getattr_L__mod___blocks___9___mlp_tokens_fc2(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_135 = self.getattr_L__mod___blocks___9___mlp_tokens_drop2(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_20 = x_135.transpose(1, 2);  x_135 = None
    getattr_l__mod___blocks___9___drop_path = self.getattr_L__mod___blocks___9___drop_path(transpose_20);  transpose_20 = None
    x_136 = x_129 + getattr_l__mod___blocks___9___drop_path;  x_129 = getattr_l__mod___blocks___9___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___9___norm2 = self.getattr_L__mod___blocks___9___norm2(x_136)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_137 = self.getattr_L__mod___blocks___9___mlp_channels_fc1(getattr_l__mod___blocks___9___norm2);  getattr_l__mod___blocks___9___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_138 = self.getattr_L__mod___blocks___9___mlp_channels_act(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_139 = self.getattr_L__mod___blocks___9___mlp_channels_drop1(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_140 = self.getattr_L__mod___blocks___9___mlp_channels_norm(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_141 = self.getattr_L__mod___blocks___9___mlp_channels_fc2(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_142 = self.getattr_L__mod___blocks___9___mlp_channels_drop2(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___9___drop_path_1 = self.getattr_L__mod___blocks___9___drop_path(x_142);  x_142 = None
    x_143 = x_136 + getattr_l__mod___blocks___9___drop_path_1;  x_136 = getattr_l__mod___blocks___9___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___10___norm1 = self.getattr_L__mod___blocks___10___norm1(x_143)
    transpose_21 = getattr_l__mod___blocks___10___norm1.transpose(1, 2);  getattr_l__mod___blocks___10___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_144 = self.getattr_L__mod___blocks___10___mlp_tokens_fc1(transpose_21);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_145 = self.getattr_L__mod___blocks___10___mlp_tokens_act(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_146 = self.getattr_L__mod___blocks___10___mlp_tokens_drop1(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_147 = self.getattr_L__mod___blocks___10___mlp_tokens_norm(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_148 = self.getattr_L__mod___blocks___10___mlp_tokens_fc2(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_149 = self.getattr_L__mod___blocks___10___mlp_tokens_drop2(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_22 = x_149.transpose(1, 2);  x_149 = None
    getattr_l__mod___blocks___10___drop_path = self.getattr_L__mod___blocks___10___drop_path(transpose_22);  transpose_22 = None
    x_150 = x_143 + getattr_l__mod___blocks___10___drop_path;  x_143 = getattr_l__mod___blocks___10___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___10___norm2 = self.getattr_L__mod___blocks___10___norm2(x_150)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_151 = self.getattr_L__mod___blocks___10___mlp_channels_fc1(getattr_l__mod___blocks___10___norm2);  getattr_l__mod___blocks___10___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_152 = self.getattr_L__mod___blocks___10___mlp_channels_act(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_153 = self.getattr_L__mod___blocks___10___mlp_channels_drop1(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_154 = self.getattr_L__mod___blocks___10___mlp_channels_norm(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_155 = self.getattr_L__mod___blocks___10___mlp_channels_fc2(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_156 = self.getattr_L__mod___blocks___10___mlp_channels_drop2(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___10___drop_path_1 = self.getattr_L__mod___blocks___10___drop_path(x_156);  x_156 = None
    x_157 = x_150 + getattr_l__mod___blocks___10___drop_path_1;  x_150 = getattr_l__mod___blocks___10___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    getattr_l__mod___blocks___11___norm1 = self.getattr_L__mod___blocks___11___norm1(x_157)
    transpose_23 = getattr_l__mod___blocks___11___norm1.transpose(1, 2);  getattr_l__mod___blocks___11___norm1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_158 = self.getattr_L__mod___blocks___11___mlp_tokens_fc1(transpose_23);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_159 = self.getattr_L__mod___blocks___11___mlp_tokens_act(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_160 = self.getattr_L__mod___blocks___11___mlp_tokens_drop1(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_161 = self.getattr_L__mod___blocks___11___mlp_tokens_norm(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_162 = self.getattr_L__mod___blocks___11___mlp_tokens_fc2(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_163 = self.getattr_L__mod___blocks___11___mlp_tokens_drop2(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    transpose_24 = x_163.transpose(1, 2);  x_163 = None
    getattr_l__mod___blocks___11___drop_path = self.getattr_L__mod___blocks___11___drop_path(transpose_24);  transpose_24 = None
    x_164 = x_157 + getattr_l__mod___blocks___11___drop_path;  x_157 = getattr_l__mod___blocks___11___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___11___norm2 = self.getattr_L__mod___blocks___11___norm2(x_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_165 = self.getattr_L__mod___blocks___11___mlp_channels_fc1(getattr_l__mod___blocks___11___norm2);  getattr_l__mod___blocks___11___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_166 = self.getattr_L__mod___blocks___11___mlp_channels_act(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_167 = self.getattr_L__mod___blocks___11___mlp_channels_drop1(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_168 = self.getattr_L__mod___blocks___11___mlp_channels_norm(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_169 = self.getattr_L__mod___blocks___11___mlp_channels_fc2(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_170 = self.getattr_L__mod___blocks___11___mlp_channels_drop2(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    getattr_l__mod___blocks___11___drop_path_1 = self.getattr_L__mod___blocks___11___drop_path(x_170);  x_170 = None
    x_172 = x_164 + getattr_l__mod___blocks___11___drop_path_1;  x_164 = getattr_l__mod___blocks___11___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    x_174 = self.L__mod___norm(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    x_175 = x_174.mean(dim = 1);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    x_176 = self.L__mod___head_drop(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    x_177 = self.L__mod___head(x_176);  x_176 = None
    return (x_177,)
    