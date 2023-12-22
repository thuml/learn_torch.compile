from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:396, code: x = self.stem(x)
    l__mod___stem_0 = self.L__mod___stem_0(l_inputs_0_);  l_inputs_0_ = None
    l__mod___stem_1 = self.L__mod___stem_1(l__mod___stem_0);  l__mod___stem_0 = None
    x = self.L__mod___stem_2(l__mod___stem_1);  l__mod___stem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x_1 = self.L__mod___patch_embed1_proj(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_3 = self.L__mod___patch_embed1_norm(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:400, code: if self.pos_embed1 is not None:
    l__mod___pos_embed1 = self.L__mod___pos_embed1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:401, code: x = self.pos_drop(x + self.pos_embed1)
    add = x_3 + l__mod___pos_embed1;  x_3 = l__mod___pos_embed1 = None
    x_4 = self.L__mod___pos_drop(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage1___0___norm2 = self.getattr_L__mod___stage1___0___norm2(x_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_5 = self.getattr_L__mod___stage1___0___mlp_conv1(getattr_l__mod___stage1___0___norm2);  getattr_l__mod___stage1___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_6 = self.getattr_L__mod___stage1___0___mlp_act1(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_7 = self.getattr_L__mod___stage1___0___mlp_drop1(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    x_8 = self.getattr_L__mod___stage1___0___mlp_conv2(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    x_9 = self.getattr_L__mod___stage1___0___mlp_act2(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_10 = self.getattr_L__mod___stage1___0___mlp_conv3(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_11 = self.getattr_L__mod___stage1___0___mlp_drop3(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage1___0___drop_path = self.getattr_L__mod___stage1___0___drop_path(x_11);  x_11 = None
    x_12 = x_4 + getattr_l__mod___stage1___0___drop_path;  x_4 = getattr_l__mod___stage1___0___drop_path = None
    getattr_l__mod___stage1___1___norm2 = self.getattr_L__mod___stage1___1___norm2(x_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_13 = self.getattr_L__mod___stage1___1___mlp_conv1(getattr_l__mod___stage1___1___norm2);  getattr_l__mod___stage1___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_14 = self.getattr_L__mod___stage1___1___mlp_act1(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_15 = self.getattr_L__mod___stage1___1___mlp_drop1(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    x_16 = self.getattr_L__mod___stage1___1___mlp_conv2(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    x_17 = self.getattr_L__mod___stage1___1___mlp_act2(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_18 = self.getattr_L__mod___stage1___1___mlp_conv3(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_19 = self.getattr_L__mod___stage1___1___mlp_drop3(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage1___1___drop_path = self.getattr_L__mod___stage1___1___drop_path(x_19);  x_19 = None
    x_20 = x_12 + getattr_l__mod___stage1___1___drop_path;  x_12 = getattr_l__mod___stage1___1___drop_path = None
    getattr_l__mod___stage1___2___norm2 = self.getattr_L__mod___stage1___2___norm2(x_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_21 = self.getattr_L__mod___stage1___2___mlp_conv1(getattr_l__mod___stage1___2___norm2);  getattr_l__mod___stage1___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_22 = self.getattr_L__mod___stage1___2___mlp_act1(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_23 = self.getattr_L__mod___stage1___2___mlp_drop1(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    x_24 = self.getattr_L__mod___stage1___2___mlp_conv2(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    x_25 = self.getattr_L__mod___stage1___2___mlp_act2(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_26 = self.getattr_L__mod___stage1___2___mlp_conv3(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_27 = self.getattr_L__mod___stage1___2___mlp_drop3(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage1___2___drop_path = self.getattr_L__mod___stage1___2___drop_path(x_27);  x_27 = None
    x_28 = x_20 + getattr_l__mod___stage1___2___drop_path;  x_20 = getattr_l__mod___stage1___2___drop_path = None
    getattr_l__mod___stage1___3___norm2 = self.getattr_L__mod___stage1___3___norm2(x_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_29 = self.getattr_L__mod___stage1___3___mlp_conv1(getattr_l__mod___stage1___3___norm2);  getattr_l__mod___stage1___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_30 = self.getattr_L__mod___stage1___3___mlp_act1(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_31 = self.getattr_L__mod___stage1___3___mlp_drop1(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    x_32 = self.getattr_L__mod___stage1___3___mlp_conv2(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    x_33 = self.getattr_L__mod___stage1___3___mlp_act2(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_34 = self.getattr_L__mod___stage1___3___mlp_conv3(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_35 = self.getattr_L__mod___stage1___3___mlp_drop3(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage1___3___drop_path = self.getattr_L__mod___stage1___3___drop_path(x_35);  x_35 = None
    x_36 = x_28 + getattr_l__mod___stage1___3___drop_path;  x_28 = getattr_l__mod___stage1___3___drop_path = None
    getattr_l__mod___stage1___4___norm2 = self.getattr_L__mod___stage1___4___norm2(x_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_37 = self.getattr_L__mod___stage1___4___mlp_conv1(getattr_l__mod___stage1___4___norm2);  getattr_l__mod___stage1___4___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_38 = self.getattr_L__mod___stage1___4___mlp_act1(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_39 = self.getattr_L__mod___stage1___4___mlp_drop1(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    x_40 = self.getattr_L__mod___stage1___4___mlp_conv2(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    x_41 = self.getattr_L__mod___stage1___4___mlp_act2(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_42 = self.getattr_L__mod___stage1___4___mlp_conv3(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_43 = self.getattr_L__mod___stage1___4___mlp_drop3(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage1___4___drop_path = self.getattr_L__mod___stage1___4___drop_path(x_43);  x_43 = None
    x_44 = x_36 + getattr_l__mod___stage1___4___drop_path;  x_36 = getattr_l__mod___stage1___4___drop_path = None
    getattr_l__mod___stage1___5___norm2 = self.getattr_L__mod___stage1___5___norm2(x_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_45 = self.getattr_L__mod___stage1___5___mlp_conv1(getattr_l__mod___stage1___5___norm2);  getattr_l__mod___stage1___5___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_46 = self.getattr_L__mod___stage1___5___mlp_act1(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_47 = self.getattr_L__mod___stage1___5___mlp_drop1(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    x_48 = self.getattr_L__mod___stage1___5___mlp_conv2(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    x_49 = self.getattr_L__mod___stage1___5___mlp_act2(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_50 = self.getattr_L__mod___stage1___5___mlp_conv3(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_51 = self.getattr_L__mod___stage1___5___mlp_drop3(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage1___5___drop_path = self.getattr_L__mod___stage1___5___drop_path(x_51);  x_51 = None
    x_52 = x_44 + getattr_l__mod___stage1___5___drop_path;  x_44 = getattr_l__mod___stage1___5___drop_path = None
    getattr_l__mod___stage1___6___norm2 = self.getattr_L__mod___stage1___6___norm2(x_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_53 = self.getattr_L__mod___stage1___6___mlp_conv1(getattr_l__mod___stage1___6___norm2);  getattr_l__mod___stage1___6___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_54 = self.getattr_L__mod___stage1___6___mlp_act1(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_55 = self.getattr_L__mod___stage1___6___mlp_drop1(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    x_56 = self.getattr_L__mod___stage1___6___mlp_conv2(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    x_57 = self.getattr_L__mod___stage1___6___mlp_act2(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_58 = self.getattr_L__mod___stage1___6___mlp_conv3(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_59 = self.getattr_L__mod___stage1___6___mlp_drop3(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage1___6___drop_path = self.getattr_L__mod___stage1___6___drop_path(x_59);  x_59 = None
    x_61 = x_52 + getattr_l__mod___stage1___6___drop_path;  x_52 = getattr_l__mod___stage1___6___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x_62 = self.L__mod___patch_embed2_proj(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_64 = self.L__mod___patch_embed2_norm(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:410, code: if self.pos_embed2 is not None:
    l__mod___pos_embed2 = self.L__mod___pos_embed2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:411, code: x = self.pos_drop(x + self.pos_embed2)
    add_8 = x_64 + l__mod___pos_embed2;  x_64 = l__mod___pos_embed2 = None
    x_65 = self.L__mod___pos_drop(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage2___0___norm1 = self.getattr_L__mod___stage2___0___norm1(x_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    getattr_l__mod___stage2___0___attn_qkv = self.getattr_L__mod___stage2___0___attn_qkv(getattr_l__mod___stage2___0___norm1);  getattr_l__mod___stage2___0___norm1 = None
    reshape = getattr_l__mod___stage2___0___attn_qkv.reshape(8, 3, 6, 64, -1);  getattr_l__mod___stage2___0___attn_qkv = None
    x_66 = reshape.permute(1, 0, 2, 4, 3);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind = x_66.unbind(0);  x_66 = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose = k.transpose(-2, -1);  k = None
    matmul = q @ transpose;  q = transpose = None
    attn = matmul * 0.125;  matmul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    attn_1 = attn.softmax(dim = -1);  attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    attn_2 = self.getattr_L__mod___stage2___0___attn_attn_drop(attn_1);  attn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    x_67 = attn_2 @ v;  attn_2 = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_1 = x_67.permute(0, 1, 3, 2);  x_67 = None
    x_68 = permute_1.reshape(8, -1, 14, 14);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    x_69 = self.getattr_L__mod___stage2___0___attn_proj(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    x_70 = self.getattr_L__mod___stage2___0___attn_proj_drop(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage2___0___drop_path = self.getattr_L__mod___stage2___0___drop_path(x_70);  x_70 = None
    x_71 = x_65 + getattr_l__mod___stage2___0___drop_path;  x_65 = getattr_l__mod___stage2___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage2___0___norm2 = self.getattr_L__mod___stage2___0___norm2(x_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_72 = self.getattr_L__mod___stage2___0___mlp_conv1(getattr_l__mod___stage2___0___norm2);  getattr_l__mod___stage2___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_73 = self.getattr_L__mod___stage2___0___mlp_act1(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_74 = self.getattr_L__mod___stage2___0___mlp_drop1(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_75 = self.getattr_L__mod___stage2___0___mlp_conv3(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_76 = self.getattr_L__mod___stage2___0___mlp_drop3(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage2___0___drop_path_1 = self.getattr_L__mod___stage2___0___drop_path(x_76);  x_76 = None
    x_77 = x_71 + getattr_l__mod___stage2___0___drop_path_1;  x_71 = getattr_l__mod___stage2___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage2___1___norm1 = self.getattr_L__mod___stage2___1___norm1(x_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    getattr_l__mod___stage2___1___attn_qkv = self.getattr_L__mod___stage2___1___attn_qkv(getattr_l__mod___stage2___1___norm1);  getattr_l__mod___stage2___1___norm1 = None
    reshape_2 = getattr_l__mod___stage2___1___attn_qkv.reshape(8, 3, 6, 64, -1);  getattr_l__mod___stage2___1___attn_qkv = None
    x_78 = reshape_2.permute(1, 0, 2, 4, 3);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_1 = x_78.unbind(0);  x_78 = None
    q_1 = unbind_1[0]
    k_1 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_1 = k_1.transpose(-2, -1);  k_1 = None
    matmul_2 = q_1 @ transpose_1;  q_1 = transpose_1 = None
    attn_3 = matmul_2 * 0.125;  matmul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    attn_4 = attn_3.softmax(dim = -1);  attn_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    attn_5 = self.getattr_L__mod___stage2___1___attn_attn_drop(attn_4);  attn_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    x_79 = attn_5 @ v_1;  attn_5 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_3 = x_79.permute(0, 1, 3, 2);  x_79 = None
    x_80 = permute_3.reshape(8, -1, 14, 14);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    x_81 = self.getattr_L__mod___stage2___1___attn_proj(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    x_82 = self.getattr_L__mod___stage2___1___attn_proj_drop(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage2___1___drop_path = self.getattr_L__mod___stage2___1___drop_path(x_82);  x_82 = None
    x_83 = x_77 + getattr_l__mod___stage2___1___drop_path;  x_77 = getattr_l__mod___stage2___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage2___1___norm2 = self.getattr_L__mod___stage2___1___norm2(x_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_84 = self.getattr_L__mod___stage2___1___mlp_conv1(getattr_l__mod___stage2___1___norm2);  getattr_l__mod___stage2___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_85 = self.getattr_L__mod___stage2___1___mlp_act1(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_86 = self.getattr_L__mod___stage2___1___mlp_drop1(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_87 = self.getattr_L__mod___stage2___1___mlp_conv3(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_88 = self.getattr_L__mod___stage2___1___mlp_drop3(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage2___1___drop_path_1 = self.getattr_L__mod___stage2___1___drop_path(x_88);  x_88 = None
    x_89 = x_83 + getattr_l__mod___stage2___1___drop_path_1;  x_83 = getattr_l__mod___stage2___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage2___2___norm1 = self.getattr_L__mod___stage2___2___norm1(x_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    getattr_l__mod___stage2___2___attn_qkv = self.getattr_L__mod___stage2___2___attn_qkv(getattr_l__mod___stage2___2___norm1);  getattr_l__mod___stage2___2___norm1 = None
    reshape_4 = getattr_l__mod___stage2___2___attn_qkv.reshape(8, 3, 6, 64, -1);  getattr_l__mod___stage2___2___attn_qkv = None
    x_90 = reshape_4.permute(1, 0, 2, 4, 3);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_2 = x_90.unbind(0);  x_90 = None
    q_2 = unbind_2[0]
    k_2 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_2 = k_2.transpose(-2, -1);  k_2 = None
    matmul_4 = q_2 @ transpose_2;  q_2 = transpose_2 = None
    attn_6 = matmul_4 * 0.125;  matmul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    attn_7 = attn_6.softmax(dim = -1);  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    attn_8 = self.getattr_L__mod___stage2___2___attn_attn_drop(attn_7);  attn_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    x_91 = attn_8 @ v_2;  attn_8 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_5 = x_91.permute(0, 1, 3, 2);  x_91 = None
    x_92 = permute_5.reshape(8, -1, 14, 14);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    x_93 = self.getattr_L__mod___stage2___2___attn_proj(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    x_94 = self.getattr_L__mod___stage2___2___attn_proj_drop(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage2___2___drop_path = self.getattr_L__mod___stage2___2___drop_path(x_94);  x_94 = None
    x_95 = x_89 + getattr_l__mod___stage2___2___drop_path;  x_89 = getattr_l__mod___stage2___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage2___2___norm2 = self.getattr_L__mod___stage2___2___norm2(x_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_96 = self.getattr_L__mod___stage2___2___mlp_conv1(getattr_l__mod___stage2___2___norm2);  getattr_l__mod___stage2___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_97 = self.getattr_L__mod___stage2___2___mlp_act1(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_98 = self.getattr_L__mod___stage2___2___mlp_drop1(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_99 = self.getattr_L__mod___stage2___2___mlp_conv3(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_100 = self.getattr_L__mod___stage2___2___mlp_drop3(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage2___2___drop_path_1 = self.getattr_L__mod___stage2___2___drop_path(x_100);  x_100 = None
    x_101 = x_95 + getattr_l__mod___stage2___2___drop_path_1;  x_95 = getattr_l__mod___stage2___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage2___3___norm1 = self.getattr_L__mod___stage2___3___norm1(x_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    getattr_l__mod___stage2___3___attn_qkv = self.getattr_L__mod___stage2___3___attn_qkv(getattr_l__mod___stage2___3___norm1);  getattr_l__mod___stage2___3___norm1 = None
    reshape_6 = getattr_l__mod___stage2___3___attn_qkv.reshape(8, 3, 6, 64, -1);  getattr_l__mod___stage2___3___attn_qkv = None
    x_102 = reshape_6.permute(1, 0, 2, 4, 3);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_3 = x_102.unbind(0);  x_102 = None
    q_3 = unbind_3[0]
    k_3 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_3 = k_3.transpose(-2, -1);  k_3 = None
    matmul_6 = q_3 @ transpose_3;  q_3 = transpose_3 = None
    attn_9 = matmul_6 * 0.125;  matmul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    attn_10 = attn_9.softmax(dim = -1);  attn_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    attn_11 = self.getattr_L__mod___stage2___3___attn_attn_drop(attn_10);  attn_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    x_103 = attn_11 @ v_3;  attn_11 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_7 = x_103.permute(0, 1, 3, 2);  x_103 = None
    x_104 = permute_7.reshape(8, -1, 14, 14);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    x_105 = self.getattr_L__mod___stage2___3___attn_proj(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    x_106 = self.getattr_L__mod___stage2___3___attn_proj_drop(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage2___3___drop_path = self.getattr_L__mod___stage2___3___drop_path(x_106);  x_106 = None
    x_107 = x_101 + getattr_l__mod___stage2___3___drop_path;  x_101 = getattr_l__mod___stage2___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage2___3___norm2 = self.getattr_L__mod___stage2___3___norm2(x_107)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_108 = self.getattr_L__mod___stage2___3___mlp_conv1(getattr_l__mod___stage2___3___norm2);  getattr_l__mod___stage2___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_109 = self.getattr_L__mod___stage2___3___mlp_act1(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_110 = self.getattr_L__mod___stage2___3___mlp_drop1(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_111 = self.getattr_L__mod___stage2___3___mlp_conv3(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_112 = self.getattr_L__mod___stage2___3___mlp_drop3(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage2___3___drop_path_1 = self.getattr_L__mod___stage2___3___drop_path(x_112);  x_112 = None
    x_114 = x_107 + getattr_l__mod___stage2___3___drop_path_1;  x_107 = getattr_l__mod___stage2___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x_115 = self.L__mod___patch_embed3_proj(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_117 = self.L__mod___patch_embed3_norm(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:420, code: if self.pos_embed3 is not None:
    l__mod___pos_embed3 = self.L__mod___pos_embed3
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:421, code: x = self.pos_drop(x + self.pos_embed3)
    add_17 = x_117 + l__mod___pos_embed3;  x_117 = l__mod___pos_embed3 = None
    x_118 = self.L__mod___pos_drop(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage3___0___norm1 = self.getattr_L__mod___stage3___0___norm1(x_118)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    getattr_l__mod___stage3___0___attn_qkv = self.getattr_L__mod___stage3___0___attn_qkv(getattr_l__mod___stage3___0___norm1);  getattr_l__mod___stage3___0___norm1 = None
    reshape_8 = getattr_l__mod___stage3___0___attn_qkv.reshape(8, 3, 6, 128, -1);  getattr_l__mod___stage3___0___attn_qkv = None
    x_119 = reshape_8.permute(1, 0, 2, 4, 3);  reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_4 = x_119.unbind(0);  x_119 = None
    q_4 = unbind_4[0]
    k_4 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_4 = k_4.transpose(-2, -1);  k_4 = None
    matmul_8 = q_4 @ transpose_4;  q_4 = transpose_4 = None
    attn_12 = matmul_8 * 0.08838834764831845;  matmul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    attn_13 = attn_12.softmax(dim = -1);  attn_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    attn_14 = self.getattr_L__mod___stage3___0___attn_attn_drop(attn_13);  attn_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    x_120 = attn_14 @ v_4;  attn_14 = v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_9 = x_120.permute(0, 1, 3, 2);  x_120 = None
    x_121 = permute_9.reshape(8, -1, 7, 7);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    x_122 = self.getattr_L__mod___stage3___0___attn_proj(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    x_123 = self.getattr_L__mod___stage3___0___attn_proj_drop(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage3___0___drop_path = self.getattr_L__mod___stage3___0___drop_path(x_123);  x_123 = None
    x_124 = x_118 + getattr_l__mod___stage3___0___drop_path;  x_118 = getattr_l__mod___stage3___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage3___0___norm2 = self.getattr_L__mod___stage3___0___norm2(x_124)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_125 = self.getattr_L__mod___stage3___0___mlp_conv1(getattr_l__mod___stage3___0___norm2);  getattr_l__mod___stage3___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_126 = self.getattr_L__mod___stage3___0___mlp_act1(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_127 = self.getattr_L__mod___stage3___0___mlp_drop1(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_128 = self.getattr_L__mod___stage3___0___mlp_conv3(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_129 = self.getattr_L__mod___stage3___0___mlp_drop3(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage3___0___drop_path_1 = self.getattr_L__mod___stage3___0___drop_path(x_129);  x_129 = None
    x_130 = x_124 + getattr_l__mod___stage3___0___drop_path_1;  x_124 = getattr_l__mod___stage3___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage3___1___norm1 = self.getattr_L__mod___stage3___1___norm1(x_130)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    getattr_l__mod___stage3___1___attn_qkv = self.getattr_L__mod___stage3___1___attn_qkv(getattr_l__mod___stage3___1___norm1);  getattr_l__mod___stage3___1___norm1 = None
    reshape_10 = getattr_l__mod___stage3___1___attn_qkv.reshape(8, 3, 6, 128, -1);  getattr_l__mod___stage3___1___attn_qkv = None
    x_131 = reshape_10.permute(1, 0, 2, 4, 3);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_5 = x_131.unbind(0);  x_131 = None
    q_5 = unbind_5[0]
    k_5 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_5 = k_5.transpose(-2, -1);  k_5 = None
    matmul_10 = q_5 @ transpose_5;  q_5 = transpose_5 = None
    attn_15 = matmul_10 * 0.08838834764831845;  matmul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    attn_16 = attn_15.softmax(dim = -1);  attn_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    attn_17 = self.getattr_L__mod___stage3___1___attn_attn_drop(attn_16);  attn_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    x_132 = attn_17 @ v_5;  attn_17 = v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_11 = x_132.permute(0, 1, 3, 2);  x_132 = None
    x_133 = permute_11.reshape(8, -1, 7, 7);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    x_134 = self.getattr_L__mod___stage3___1___attn_proj(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    x_135 = self.getattr_L__mod___stage3___1___attn_proj_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage3___1___drop_path = self.getattr_L__mod___stage3___1___drop_path(x_135);  x_135 = None
    x_136 = x_130 + getattr_l__mod___stage3___1___drop_path;  x_130 = getattr_l__mod___stage3___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage3___1___norm2 = self.getattr_L__mod___stage3___1___norm2(x_136)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_137 = self.getattr_L__mod___stage3___1___mlp_conv1(getattr_l__mod___stage3___1___norm2);  getattr_l__mod___stage3___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_138 = self.getattr_L__mod___stage3___1___mlp_act1(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_139 = self.getattr_L__mod___stage3___1___mlp_drop1(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_140 = self.getattr_L__mod___stage3___1___mlp_conv3(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_141 = self.getattr_L__mod___stage3___1___mlp_drop3(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage3___1___drop_path_1 = self.getattr_L__mod___stage3___1___drop_path(x_141);  x_141 = None
    x_142 = x_136 + getattr_l__mod___stage3___1___drop_path_1;  x_136 = getattr_l__mod___stage3___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage3___2___norm1 = self.getattr_L__mod___stage3___2___norm1(x_142)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    getattr_l__mod___stage3___2___attn_qkv = self.getattr_L__mod___stage3___2___attn_qkv(getattr_l__mod___stage3___2___norm1);  getattr_l__mod___stage3___2___norm1 = None
    reshape_12 = getattr_l__mod___stage3___2___attn_qkv.reshape(8, 3, 6, 128, -1);  getattr_l__mod___stage3___2___attn_qkv = None
    x_143 = reshape_12.permute(1, 0, 2, 4, 3);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_6 = x_143.unbind(0);  x_143 = None
    q_6 = unbind_6[0]
    k_6 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_6 = k_6.transpose(-2, -1);  k_6 = None
    matmul_12 = q_6 @ transpose_6;  q_6 = transpose_6 = None
    attn_18 = matmul_12 * 0.08838834764831845;  matmul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    attn_19 = attn_18.softmax(dim = -1);  attn_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    attn_20 = self.getattr_L__mod___stage3___2___attn_attn_drop(attn_19);  attn_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    x_144 = attn_20 @ v_6;  attn_20 = v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_13 = x_144.permute(0, 1, 3, 2);  x_144 = None
    x_145 = permute_13.reshape(8, -1, 7, 7);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    x_146 = self.getattr_L__mod___stage3___2___attn_proj(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    x_147 = self.getattr_L__mod___stage3___2___attn_proj_drop(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage3___2___drop_path = self.getattr_L__mod___stage3___2___drop_path(x_147);  x_147 = None
    x_148 = x_142 + getattr_l__mod___stage3___2___drop_path;  x_142 = getattr_l__mod___stage3___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage3___2___norm2 = self.getattr_L__mod___stage3___2___norm2(x_148)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_149 = self.getattr_L__mod___stage3___2___mlp_conv1(getattr_l__mod___stage3___2___norm2);  getattr_l__mod___stage3___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_150 = self.getattr_L__mod___stage3___2___mlp_act1(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_151 = self.getattr_L__mod___stage3___2___mlp_drop1(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_152 = self.getattr_L__mod___stage3___2___mlp_conv3(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_153 = self.getattr_L__mod___stage3___2___mlp_drop3(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage3___2___drop_path_1 = self.getattr_L__mod___stage3___2___drop_path(x_153);  x_153 = None
    x_154 = x_148 + getattr_l__mod___stage3___2___drop_path_1;  x_148 = getattr_l__mod___stage3___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage3___3___norm1 = self.getattr_L__mod___stage3___3___norm1(x_154)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    getattr_l__mod___stage3___3___attn_qkv = self.getattr_L__mod___stage3___3___attn_qkv(getattr_l__mod___stage3___3___norm1);  getattr_l__mod___stage3___3___norm1 = None
    reshape_14 = getattr_l__mod___stage3___3___attn_qkv.reshape(8, 3, 6, 128, -1);  getattr_l__mod___stage3___3___attn_qkv = None
    x_155 = reshape_14.permute(1, 0, 2, 4, 3);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_7 = x_155.unbind(0);  x_155 = None
    q_7 = unbind_7[0]
    k_7 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_7 = k_7.transpose(-2, -1);  k_7 = None
    matmul_14 = q_7 @ transpose_7;  q_7 = transpose_7 = None
    attn_21 = matmul_14 * 0.08838834764831845;  matmul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    attn_22 = attn_21.softmax(dim = -1);  attn_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    attn_23 = self.getattr_L__mod___stage3___3___attn_attn_drop(attn_22);  attn_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    x_156 = attn_23 @ v_7;  attn_23 = v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_15 = x_156.permute(0, 1, 3, 2);  x_156 = None
    x_157 = permute_15.reshape(8, -1, 7, 7);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    x_158 = self.getattr_L__mod___stage3___3___attn_proj(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    x_159 = self.getattr_L__mod___stage3___3___attn_proj_drop(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___stage3___3___drop_path = self.getattr_L__mod___stage3___3___drop_path(x_159);  x_159 = None
    x_160 = x_154 + getattr_l__mod___stage3___3___drop_path;  x_154 = getattr_l__mod___stage3___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage3___3___norm2 = self.getattr_L__mod___stage3___3___norm2(x_160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    x_161 = self.getattr_L__mod___stage3___3___mlp_conv1(getattr_l__mod___stage3___3___norm2);  getattr_l__mod___stage3___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    x_162 = self.getattr_L__mod___stage3___3___mlp_act1(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    x_163 = self.getattr_L__mod___stage3___3___mlp_drop1(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    x_164 = self.getattr_L__mod___stage3___3___mlp_conv3(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    x_165 = self.getattr_L__mod___stage3___3___mlp_drop3(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___stage3___3___drop_path_1 = self.getattr_L__mod___stage3___3___drop_path(x_165);  x_165 = None
    x_167 = x_160 + getattr_l__mod___stage3___3___drop_path_1;  x_160 = getattr_l__mod___stage3___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:427, code: x = self.norm(x)
    x_169 = self.L__mod___norm(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_170 = self.L__mod___global_pool_pool(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_172 = self.L__mod___global_pool_flatten(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:432, code: x = self.head_drop(x)
    x_173 = self.L__mod___head_drop(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:433, code: return x if pre_logits else self.head(x)
    x_174 = self.L__mod___head(x_173);  x_173 = None
    return (x_174,)
    