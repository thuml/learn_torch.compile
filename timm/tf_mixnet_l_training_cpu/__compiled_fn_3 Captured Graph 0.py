from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___conv_stem_weight = self.L__mod___conv_stem_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x = torch.nn.functional.pad(l_cloned_inputs_0_, (0, 1, 0, 1), value = 0);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_1 = torch.conv2d(x, l__mod___conv_stem_weight, None, (2, 2), (0, 0), (1, 1), 1);  x = l__mod___conv_stem_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___bn1_num_batches_tracked = self.L__mod___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_ = l__mod___bn1_num_batches_tracked.add_(1);  l__mod___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___bn1_running_mean = self.L__mod___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___bn1_running_var = self.L__mod___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___bn1_weight = self.L__mod___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___bn1_bias = self.L__mod___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_2 = torch.nn.functional.batch_norm(x_1, l__mod___bn1_running_mean, l__mod___bn1_running_var, l__mod___bn1_weight, l__mod___bn1_bias, True, 0.1, 0.001);  x_1 = l__mod___bn1_running_mean = l__mod___bn1_running_var = l__mod___bn1_weight = l__mod___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_3 = self.L__mod___bn1_drop(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___bn1_act(x_3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    x_6 = self.getattr_getattr_L__mod___blocks___0_____0___conv_dw(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___0_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___0_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__1 = getattr_getattr_l__mod___blocks___0_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___0_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___0_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___0_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___0_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___0_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___0_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___0_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn1_running_var, getattr_getattr_l__mod___blocks___0_____0___bn1_weight, getattr_getattr_l__mod___blocks___0_____0___bn1_bias, True, 0.1, 0.001);  x_6 = getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn1_running_var = getattr_getattr_l__mod___blocks___0_____0___bn1_weight = getattr_getattr_l__mod___blocks___0_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.getattr_getattr_L__mod___blocks___0_____0___bn1_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_10 = self.getattr_getattr_L__mod___blocks___0_____0___bn1_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:125, code: x = self.se(x)
    x_11 = self.getattr_getattr_L__mod___blocks___0_____0___se(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    x_12 = self.getattr_getattr_L__mod___blocks___0_____0___conv_pw(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___0_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___0_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__2 = getattr_getattr_l__mod___blocks___0_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___0_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___0_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___0_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___0_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___0_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___0_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___0_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_13 = torch.nn.functional.batch_norm(x_12, getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn2_running_var, getattr_getattr_l__mod___blocks___0_____0___bn2_weight, getattr_getattr_l__mod___blocks___0_____0___bn2_bias, True, 0.1, 0.001);  x_12 = getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn2_running_var = getattr_getattr_l__mod___blocks___0_____0___bn2_weight = getattr_getattr_l__mod___blocks___0_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_14 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_drop(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_16 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_act(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___0_____0___drop_path = self.getattr_getattr_L__mod___blocks___0_____0___drop_path(x_16);  x_16 = None
    shortcut_1 = getattr_getattr_l__mod___blocks___0_____0___drop_path + shortcut;  getattr_getattr_l__mod___blocks___0_____0___drop_path = shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split = torch.functional.split(shortcut_1, [16, 16], 1);  shortcut_1 = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____0___conv_pw_0 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pw_0(getitem);  getitem = None
    getattr_getattr_l__mod___blocks___1_____0___conv_pw_1 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pw_1(getitem_1);  getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_19 = torch.cat([getattr_getattr_l__mod___blocks___1_____0___conv_pw_0, getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___1_____0___conv_pw_0 = getattr_getattr_l__mod___blocks___1_____0___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___1_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___1_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__3 = getattr_getattr_l__mod___blocks___1_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___1_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_20 = torch.nn.functional.batch_norm(x_19, getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn1_running_var, getattr_getattr_l__mod___blocks___1_____0___bn1_weight, getattr_getattr_l__mod___blocks___1_____0___bn1_bias, True, 0.1, 0.001);  x_19 = getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = getattr_getattr_l__mod___blocks___1_____0___bn1_weight = getattr_getattr_l__mod___blocks___1_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_21 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_drop(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_23 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_1 = torch.functional.split(x_23, [64, 64, 64], 1);  x_23 = None
    getitem_2 = split_1[0]
    getitem_3 = split_1[1]
    getitem_4 = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___0___weight = self.getattr_getattr_getattr_L__mod___blocks___1_____0___conv_dw___0___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_25 = torch.nn.functional.pad(getitem_2, (0, 1, 0, 1), value = 0);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_1 = torch.conv2d(x_25, getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___0___weight, None, (2, 2), (0, 0), (1, 1), 64);  x_25 = getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___0___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___1___weight = self.getattr_getattr_getattr_L__mod___blocks___1_____0___conv_dw___1___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_27 = torch.nn.functional.pad(getitem_3, (1, 2, 1, 2), value = 0);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_2 = torch.conv2d(x_27, getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___1___weight, None, (2, 2), (0, 0), (1, 1), 64);  x_27 = getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___1___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___2___weight = self.getattr_getattr_getattr_L__mod___blocks___1_____0___conv_dw___2___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_29 = torch.nn.functional.pad(getitem_4, (2, 3, 2, 3), value = 0);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_3 = torch.conv2d(x_29, getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___2___weight, None, (2, 2), (0, 0), (1, 1), 64);  x_29 = getattr_getattr_getattr_l__mod___blocks___1_____0___conv_dw___2___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_31 = torch.cat([conv2d_1, conv2d_2, conv2d_3], 1);  conv2d_1 = conv2d_2 = conv2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___1_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___1_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__4 = getattr_getattr_l__mod___blocks___1_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___1_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_32 = torch.nn.functional.batch_norm(x_31, getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn2_running_var, getattr_getattr_l__mod___blocks___1_____0___bn2_weight, getattr_getattr_l__mod___blocks___1_____0___bn2_bias, True, 0.1, 0.001);  x_31 = getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = getattr_getattr_l__mod___blocks___1_____0___bn2_weight = getattr_getattr_l__mod___blocks___1_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_33 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_drop(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_35 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_act(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_36 = self.getattr_getattr_L__mod___blocks___1_____0___se(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_2 = torch.functional.split(x_36, [96, 96], 1);  x_36 = None
    getitem_5 = split_2[0]
    getitem_6 = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pwl_0(getitem_5);  getitem_5 = None
    getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pwl_1(getitem_6);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_38 = torch.cat([getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0, getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0 = getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___1_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___1_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__5 = getattr_getattr_l__mod___blocks___1_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___1_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_39 = torch.nn.functional.batch_norm(x_38, getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn3_running_var, getattr_getattr_l__mod___blocks___1_____0___bn3_weight, getattr_getattr_l__mod___blocks___1_____0___bn3_bias, True, 0.1, 0.001);  x_38 = getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = getattr_getattr_l__mod___blocks___1_____0___bn3_weight = getattr_getattr_l__mod___blocks___1_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_40 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_drop(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_act(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_3 = torch.functional.split(shortcut_2, [20, 20], 1)
    getitem_7 = split_3[0]
    getitem_8 = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____1___conv_pw_0 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pw_0(getitem_7);  getitem_7 = None
    getattr_getattr_l__mod___blocks___1_____1___conv_pw_1 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pw_1(getitem_8);  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_44 = torch.cat([getattr_getattr_l__mod___blocks___1_____1___conv_pw_0, getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___1_____1___conv_pw_0 = getattr_getattr_l__mod___blocks___1_____1___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___1_____1___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___1_____1___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__6 = getattr_getattr_l__mod___blocks___1_____1___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___1_____1___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_45 = torch.nn.functional.batch_norm(x_44, getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn1_running_var, getattr_getattr_l__mod___blocks___1_____1___bn1_weight, getattr_getattr_l__mod___blocks___1_____1___bn1_bias, True, 0.1, 0.001);  x_44 = getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = getattr_getattr_l__mod___blocks___1_____1___bn1_weight = getattr_getattr_l__mod___blocks___1_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_46 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_drop(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_48 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_act(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_49 = self.getattr_getattr_L__mod___blocks___1_____1___conv_dw(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___1_____1___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___1_____1___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__7 = getattr_getattr_l__mod___blocks___1_____1___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___1_____1___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_50 = torch.nn.functional.batch_norm(x_49, getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn2_running_var, getattr_getattr_l__mod___blocks___1_____1___bn2_weight, getattr_getattr_l__mod___blocks___1_____1___bn2_bias, True, 0.1, 0.001);  x_49 = getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = getattr_getattr_l__mod___blocks___1_____1___bn2_weight = getattr_getattr_l__mod___blocks___1_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_51 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_drop(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_53 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_act(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_54 = self.getattr_getattr_L__mod___blocks___1_____1___se(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_4 = torch.functional.split(x_54, [60, 60], 1);  x_54 = None
    getitem_9 = split_4[0]
    getitem_10 = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pwl_0(getitem_9);  getitem_9 = None
    getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pwl_1(getitem_10);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_56 = torch.cat([getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___1_____1___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___1_____1___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__8 = getattr_getattr_l__mod___blocks___1_____1___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___1_____1___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_57 = torch.nn.functional.batch_norm(x_56, getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn3_running_var, getattr_getattr_l__mod___blocks___1_____1___bn3_weight, getattr_getattr_l__mod___blocks___1_____1___bn3_bias, True, 0.1, 0.001);  x_56 = getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = getattr_getattr_l__mod___blocks___1_____1___bn3_weight = getattr_getattr_l__mod___blocks___1_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_58 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_drop(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_60 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_act(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___1_____1___drop_path = self.getattr_getattr_L__mod___blocks___1_____1___drop_path(x_60);  x_60 = None
    shortcut_3 = getattr_getattr_l__mod___blocks___1_____1___drop_path + shortcut_2;  getattr_getattr_l__mod___blocks___1_____1___drop_path = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_62 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pw(shortcut_3);  shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__9 = getattr_getattr_l__mod___blocks___2_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_63 = torch.nn.functional.batch_norm(x_62, getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn1_running_var, getattr_getattr_l__mod___blocks___2_____0___bn1_weight, getattr_getattr_l__mod___blocks___2_____0___bn1_bias, True, 0.1, 0.001);  x_62 = getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = getattr_getattr_l__mod___blocks___2_____0___bn1_weight = getattr_getattr_l__mod___blocks___2_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_64 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_drop(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_66 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_act(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_5 = torch.functional.split(x_66, [60, 60, 60, 60], 1);  x_66 = None
    getitem_11 = split_5[0]
    getitem_12 = split_5[1]
    getitem_13 = split_5[2]
    getitem_14 = split_5[3];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___0___weight = self.getattr_getattr_getattr_L__mod___blocks___2_____0___conv_dw___0___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_68 = torch.nn.functional.pad(getitem_11, (0, 1, 0, 1), value = 0);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_4 = torch.conv2d(x_68, getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___0___weight, None, (2, 2), (0, 0), (1, 1), 60);  x_68 = getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___0___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___1___weight = self.getattr_getattr_getattr_L__mod___blocks___2_____0___conv_dw___1___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_70 = torch.nn.functional.pad(getitem_12, (1, 2, 1, 2), value = 0);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_5 = torch.conv2d(x_70, getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___1___weight, None, (2, 2), (0, 0), (1, 1), 60);  x_70 = getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___1___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___2___weight = self.getattr_getattr_getattr_L__mod___blocks___2_____0___conv_dw___2___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_72 = torch.nn.functional.pad(getitem_13, (2, 3, 2, 3), value = 0);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_6 = torch.conv2d(x_72, getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___2___weight, None, (2, 2), (0, 0), (1, 1), 60);  x_72 = getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___2___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___3___weight = self.getattr_getattr_getattr_L__mod___blocks___2_____0___conv_dw___3___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_74 = torch.nn.functional.pad(getitem_14, (3, 4, 3, 4), value = 0);  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_7 = torch.conv2d(x_74, getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___3___weight, None, (2, 2), (0, 0), (1, 1), 60);  x_74 = getattr_getattr_getattr_l__mod___blocks___2_____0___conv_dw___3___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_76 = torch.cat([conv2d_4, conv2d_5, conv2d_6, conv2d_7], 1);  conv2d_4 = conv2d_5 = conv2d_6 = conv2d_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__10 = getattr_getattr_l__mod___blocks___2_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_77 = torch.nn.functional.batch_norm(x_76, getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn2_running_var, getattr_getattr_l__mod___blocks___2_____0___bn2_weight, getattr_getattr_l__mod___blocks___2_____0___bn2_bias, True, 0.1, 0.001);  x_76 = getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = getattr_getattr_l__mod___blocks___2_____0___bn2_weight = getattr_getattr_l__mod___blocks___2_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_78 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_drop(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_80 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_act(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_80.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_1 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_reduce(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_2 = self.getattr_getattr_L__mod___blocks___2_____0___se_act1(x_se_1);  x_se_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_3 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_expand(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____0___se_gate = self.getattr_getattr_L__mod___blocks___2_____0___se_gate(x_se_3);  x_se_3 = None
    x_81 = x_80 * getattr_getattr_l__mod___blocks___2_____0___se_gate;  x_80 = getattr_getattr_l__mod___blocks___2_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_82 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pwl(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__11 = getattr_getattr_l__mod___blocks___2_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_83 = torch.nn.functional.batch_norm(x_82, getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn3_running_var, getattr_getattr_l__mod___blocks___2_____0___bn3_weight, getattr_getattr_l__mod___blocks___2_____0___bn3_bias, True, 0.1, 0.001);  x_82 = getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = getattr_getattr_l__mod___blocks___2_____0___bn3_weight = getattr_getattr_l__mod___blocks___2_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_84 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_drop(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_4 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_act(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_6 = torch.functional.split(shortcut_4, [28, 28], 1)
    getitem_15 = split_6[0]
    getitem_16 = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____1___conv_pw_0 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pw_0(getitem_15);  getitem_15 = None
    getattr_getattr_l__mod___blocks___2_____1___conv_pw_1 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pw_1(getitem_16);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_88 = torch.cat([getattr_getattr_l__mod___blocks___2_____1___conv_pw_0, getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___2_____1___conv_pw_0 = getattr_getattr_l__mod___blocks___2_____1___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____1___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____1___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__12 = getattr_getattr_l__mod___blocks___2_____1___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____1___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_89 = torch.nn.functional.batch_norm(x_88, getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn1_running_var, getattr_getattr_l__mod___blocks___2_____1___bn1_weight, getattr_getattr_l__mod___blocks___2_____1___bn1_bias, True, 0.1, 0.001);  x_88 = getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = getattr_getattr_l__mod___blocks___2_____1___bn1_weight = getattr_getattr_l__mod___blocks___2_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_90 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_drop(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_92 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_act(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_7 = torch.functional.split(x_92, [168, 168], 1);  x_92 = None
    getitem_17 = split_7[0]
    getitem_18 = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____1___conv_dw_0 = self.getattr_getattr_L__mod___blocks___2_____1___conv_dw_0(getitem_17);  getitem_17 = None
    getattr_getattr_l__mod___blocks___2_____1___conv_dw_1 = self.getattr_getattr_L__mod___blocks___2_____1___conv_dw_1(getitem_18);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_94 = torch.cat([getattr_getattr_l__mod___blocks___2_____1___conv_dw_0, getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], 1);  getattr_getattr_l__mod___blocks___2_____1___conv_dw_0 = getattr_getattr_l__mod___blocks___2_____1___conv_dw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____1___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____1___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__13 = getattr_getattr_l__mod___blocks___2_____1___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____1___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_95 = torch.nn.functional.batch_norm(x_94, getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn2_running_var, getattr_getattr_l__mod___blocks___2_____1___bn2_weight, getattr_getattr_l__mod___blocks___2_____1___bn2_bias, True, 0.1, 0.001);  x_94 = getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = getattr_getattr_l__mod___blocks___2_____1___bn2_weight = getattr_getattr_l__mod___blocks___2_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_96 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_drop(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_98 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_act(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_98.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_5 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_reduce(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_6 = self.getattr_getattr_L__mod___blocks___2_____1___se_act1(x_se_5);  x_se_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_7 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_expand(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____1___se_gate = self.getattr_getattr_L__mod___blocks___2_____1___se_gate(x_se_7);  x_se_7 = None
    x_99 = x_98 * getattr_getattr_l__mod___blocks___2_____1___se_gate;  x_98 = getattr_getattr_l__mod___blocks___2_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_8 = torch.functional.split(x_99, [168, 168], 1);  x_99 = None
    getitem_19 = split_8[0]
    getitem_20 = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pwl_0(getitem_19);  getitem_19 = None
    getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pwl_1(getitem_20);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_101 = torch.cat([getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____1___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____1___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__14 = getattr_getattr_l__mod___blocks___2_____1___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____1___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_102 = torch.nn.functional.batch_norm(x_101, getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn3_running_var, getattr_getattr_l__mod___blocks___2_____1___bn3_weight, getattr_getattr_l__mod___blocks___2_____1___bn3_bias, True, 0.1, 0.001);  x_101 = getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = getattr_getattr_l__mod___blocks___2_____1___bn3_weight = getattr_getattr_l__mod___blocks___2_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_103 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_drop(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_105 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_act(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____1___drop_path = self.getattr_getattr_L__mod___blocks___2_____1___drop_path(x_105);  x_105 = None
    shortcut_5 = getattr_getattr_l__mod___blocks___2_____1___drop_path + shortcut_4;  getattr_getattr_l__mod___blocks___2_____1___drop_path = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_9 = torch.functional.split(shortcut_5, [28, 28], 1)
    getitem_21 = split_9[0]
    getitem_22 = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____2___conv_pw_0 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pw_0(getitem_21);  getitem_21 = None
    getattr_getattr_l__mod___blocks___2_____2___conv_pw_1 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pw_1(getitem_22);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_108 = torch.cat([getattr_getattr_l__mod___blocks___2_____2___conv_pw_0, getattr_getattr_l__mod___blocks___2_____2___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___2_____2___conv_pw_0 = getattr_getattr_l__mod___blocks___2_____2___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____2___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____2___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__15 = getattr_getattr_l__mod___blocks___2_____2___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____2___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_109 = torch.nn.functional.batch_norm(x_108, getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn1_running_var, getattr_getattr_l__mod___blocks___2_____2___bn1_weight, getattr_getattr_l__mod___blocks___2_____2___bn1_bias, True, 0.1, 0.001);  x_108 = getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn1_running_var = getattr_getattr_l__mod___blocks___2_____2___bn1_weight = getattr_getattr_l__mod___blocks___2_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_110 = self.getattr_getattr_L__mod___blocks___2_____2___bn1_drop(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_112 = self.getattr_getattr_L__mod___blocks___2_____2___bn1_act(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_10 = torch.functional.split(x_112, [168, 168], 1);  x_112 = None
    getitem_23 = split_10[0]
    getitem_24 = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____2___conv_dw_0 = self.getattr_getattr_L__mod___blocks___2_____2___conv_dw_0(getitem_23);  getitem_23 = None
    getattr_getattr_l__mod___blocks___2_____2___conv_dw_1 = self.getattr_getattr_L__mod___blocks___2_____2___conv_dw_1(getitem_24);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_114 = torch.cat([getattr_getattr_l__mod___blocks___2_____2___conv_dw_0, getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], 1);  getattr_getattr_l__mod___blocks___2_____2___conv_dw_0 = getattr_getattr_l__mod___blocks___2_____2___conv_dw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____2___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____2___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__16 = getattr_getattr_l__mod___blocks___2_____2___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____2___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_115 = torch.nn.functional.batch_norm(x_114, getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn2_running_var, getattr_getattr_l__mod___blocks___2_____2___bn2_weight, getattr_getattr_l__mod___blocks___2_____2___bn2_bias, True, 0.1, 0.001);  x_114 = getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn2_running_var = getattr_getattr_l__mod___blocks___2_____2___bn2_weight = getattr_getattr_l__mod___blocks___2_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_116 = self.getattr_getattr_L__mod___blocks___2_____2___bn2_drop(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_118 = self.getattr_getattr_L__mod___blocks___2_____2___bn2_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_118.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_9 = self.getattr_getattr_L__mod___blocks___2_____2___se_conv_reduce(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_10 = self.getattr_getattr_L__mod___blocks___2_____2___se_act1(x_se_9);  x_se_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_11 = self.getattr_getattr_L__mod___blocks___2_____2___se_conv_expand(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____2___se_gate = self.getattr_getattr_L__mod___blocks___2_____2___se_gate(x_se_11);  x_se_11 = None
    x_119 = x_118 * getattr_getattr_l__mod___blocks___2_____2___se_gate;  x_118 = getattr_getattr_l__mod___blocks___2_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_11 = torch.functional.split(x_119, [168, 168], 1);  x_119 = None
    getitem_25 = split_11[0]
    getitem_26 = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pwl_0(getitem_25);  getitem_25 = None
    getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pwl_1(getitem_26);  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_121 = torch.cat([getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0, getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0 = getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____2___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____2___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__17 = getattr_getattr_l__mod___blocks___2_____2___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____2___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_122 = torch.nn.functional.batch_norm(x_121, getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn3_running_var, getattr_getattr_l__mod___blocks___2_____2___bn3_weight, getattr_getattr_l__mod___blocks___2_____2___bn3_bias, True, 0.1, 0.001);  x_121 = getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn3_running_var = getattr_getattr_l__mod___blocks___2_____2___bn3_weight = getattr_getattr_l__mod___blocks___2_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_123 = self.getattr_getattr_L__mod___blocks___2_____2___bn3_drop(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_125 = self.getattr_getattr_L__mod___blocks___2_____2___bn3_act(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____2___drop_path = self.getattr_getattr_L__mod___blocks___2_____2___drop_path(x_125);  x_125 = None
    shortcut_6 = getattr_getattr_l__mod___blocks___2_____2___drop_path + shortcut_5;  getattr_getattr_l__mod___blocks___2_____2___drop_path = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_12 = torch.functional.split(shortcut_6, [28, 28], 1)
    getitem_27 = split_12[0]
    getitem_28 = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____3___conv_pw_0 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pw_0(getitem_27);  getitem_27 = None
    getattr_getattr_l__mod___blocks___2_____3___conv_pw_1 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pw_1(getitem_28);  getitem_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_128 = torch.cat([getattr_getattr_l__mod___blocks___2_____3___conv_pw_0, getattr_getattr_l__mod___blocks___2_____3___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___2_____3___conv_pw_0 = getattr_getattr_l__mod___blocks___2_____3___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____3___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____3___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__18 = getattr_getattr_l__mod___blocks___2_____3___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____3___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_129 = torch.nn.functional.batch_norm(x_128, getattr_getattr_l__mod___blocks___2_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn1_running_var, getattr_getattr_l__mod___blocks___2_____3___bn1_weight, getattr_getattr_l__mod___blocks___2_____3___bn1_bias, True, 0.1, 0.001);  x_128 = getattr_getattr_l__mod___blocks___2_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn1_running_var = getattr_getattr_l__mod___blocks___2_____3___bn1_weight = getattr_getattr_l__mod___blocks___2_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_130 = self.getattr_getattr_L__mod___blocks___2_____3___bn1_drop(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_132 = self.getattr_getattr_L__mod___blocks___2_____3___bn1_act(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_13 = torch.functional.split(x_132, [168, 168], 1);  x_132 = None
    getitem_29 = split_13[0]
    getitem_30 = split_13[1];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____3___conv_dw_0 = self.getattr_getattr_L__mod___blocks___2_____3___conv_dw_0(getitem_29);  getitem_29 = None
    getattr_getattr_l__mod___blocks___2_____3___conv_dw_1 = self.getattr_getattr_L__mod___blocks___2_____3___conv_dw_1(getitem_30);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_134 = torch.cat([getattr_getattr_l__mod___blocks___2_____3___conv_dw_0, getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], 1);  getattr_getattr_l__mod___blocks___2_____3___conv_dw_0 = getattr_getattr_l__mod___blocks___2_____3___conv_dw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____3___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____3___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__19 = getattr_getattr_l__mod___blocks___2_____3___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____3___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_135 = torch.nn.functional.batch_norm(x_134, getattr_getattr_l__mod___blocks___2_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn2_running_var, getattr_getattr_l__mod___blocks___2_____3___bn2_weight, getattr_getattr_l__mod___blocks___2_____3___bn2_bias, True, 0.1, 0.001);  x_134 = getattr_getattr_l__mod___blocks___2_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn2_running_var = getattr_getattr_l__mod___blocks___2_____3___bn2_weight = getattr_getattr_l__mod___blocks___2_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_136 = self.getattr_getattr_L__mod___blocks___2_____3___bn2_drop(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_138 = self.getattr_getattr_L__mod___blocks___2_____3___bn2_act(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_138.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_13 = self.getattr_getattr_L__mod___blocks___2_____3___se_conv_reduce(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_14 = self.getattr_getattr_L__mod___blocks___2_____3___se_act1(x_se_13);  x_se_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_15 = self.getattr_getattr_L__mod___blocks___2_____3___se_conv_expand(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____3___se_gate = self.getattr_getattr_L__mod___blocks___2_____3___se_gate(x_se_15);  x_se_15 = None
    x_139 = x_138 * getattr_getattr_l__mod___blocks___2_____3___se_gate;  x_138 = getattr_getattr_l__mod___blocks___2_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_14 = torch.functional.split(x_139, [168, 168], 1);  x_139 = None
    getitem_31 = split_14[0]
    getitem_32 = split_14[1];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pwl_0(getitem_31);  getitem_31 = None
    getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pwl_1(getitem_32);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_141 = torch.cat([getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0, getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0 = getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___2_____3___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___2_____3___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__20 = getattr_getattr_l__mod___blocks___2_____3___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___2_____3___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_142 = torch.nn.functional.batch_norm(x_141, getattr_getattr_l__mod___blocks___2_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn3_running_var, getattr_getattr_l__mod___blocks___2_____3___bn3_weight, getattr_getattr_l__mod___blocks___2_____3___bn3_bias, True, 0.1, 0.001);  x_141 = getattr_getattr_l__mod___blocks___2_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn3_running_var = getattr_getattr_l__mod___blocks___2_____3___bn3_weight = getattr_getattr_l__mod___blocks___2_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_143 = self.getattr_getattr_L__mod___blocks___2_____3___bn3_drop(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_145 = self.getattr_getattr_L__mod___blocks___2_____3___bn3_act(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____3___drop_path = self.getattr_getattr_L__mod___blocks___2_____3___drop_path(x_145);  x_145 = None
    shortcut_7 = getattr_getattr_l__mod___blocks___2_____3___drop_path + shortcut_6;  getattr_getattr_l__mod___blocks___2_____3___drop_path = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_147 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pw(shortcut_7);  shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__21 = getattr_getattr_l__mod___blocks___3_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_148 = torch.nn.functional.batch_norm(x_147, getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn1_running_var, getattr_getattr_l__mod___blocks___3_____0___bn1_weight, getattr_getattr_l__mod___blocks___3_____0___bn1_bias, True, 0.1, 0.001);  x_147 = getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = getattr_getattr_l__mod___blocks___3_____0___bn1_weight = getattr_getattr_l__mod___blocks___3_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_149 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_drop(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_151 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_act(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_15 = torch.functional.split(x_151, [112, 112, 112], 1);  x_151 = None
    getitem_33 = split_15[0]
    getitem_34 = split_15[1]
    getitem_35 = split_15[2];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___0___weight = self.getattr_getattr_getattr_L__mod___blocks___3_____0___conv_dw___0___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_153 = torch.nn.functional.pad(getitem_33, (0, 1, 0, 1), value = 0);  getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_8 = torch.conv2d(x_153, getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___0___weight, None, (2, 2), (0, 0), (1, 1), 112);  x_153 = getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___0___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___1___weight = self.getattr_getattr_getattr_L__mod___blocks___3_____0___conv_dw___1___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_155 = torch.nn.functional.pad(getitem_34, (1, 2, 1, 2), value = 0);  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_9 = torch.conv2d(x_155, getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___1___weight, None, (2, 2), (0, 0), (1, 1), 112);  x_155 = getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___1___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___2___weight = self.getattr_getattr_getattr_L__mod___blocks___3_____0___conv_dw___2___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_157 = torch.nn.functional.pad(getitem_35, (2, 3, 2, 3), value = 0);  getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_10 = torch.conv2d(x_157, getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___2___weight, None, (2, 2), (0, 0), (1, 1), 112);  x_157 = getattr_getattr_getattr_l__mod___blocks___3_____0___conv_dw___2___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_159 = torch.cat([conv2d_8, conv2d_9, conv2d_10], 1);  conv2d_8 = conv2d_9 = conv2d_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__22 = getattr_getattr_l__mod___blocks___3_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_160 = torch.nn.functional.batch_norm(x_159, getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn2_running_var, getattr_getattr_l__mod___blocks___3_____0___bn2_weight, getattr_getattr_l__mod___blocks___3_____0___bn2_bias, True, 0.1, 0.001);  x_159 = getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = getattr_getattr_l__mod___blocks___3_____0___bn2_weight = getattr_getattr_l__mod___blocks___3_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_161 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_drop(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_163 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_act(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_163.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_17 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_reduce(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_18 = self.getattr_getattr_L__mod___blocks___3_____0___se_act1(x_se_17);  x_se_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_19 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_expand(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____0___se_gate = self.getattr_getattr_L__mod___blocks___3_____0___se_gate(x_se_19);  x_se_19 = None
    x_164 = x_163 * getattr_getattr_l__mod___blocks___3_____0___se_gate;  x_163 = getattr_getattr_l__mod___blocks___3_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_165 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pwl(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__23 = getattr_getattr_l__mod___blocks___3_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_166 = torch.nn.functional.batch_norm(x_165, getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn3_running_var, getattr_getattr_l__mod___blocks___3_____0___bn3_weight, getattr_getattr_l__mod___blocks___3_____0___bn3_bias, True, 0.1, 0.001);  x_165 = getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = getattr_getattr_l__mod___blocks___3_____0___bn3_weight = getattr_getattr_l__mod___blocks___3_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_167 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_drop(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_8 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_act(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_16 = torch.functional.split(shortcut_8, [52, 52], 1)
    getitem_36 = split_16[0]
    getitem_37 = split_16[1];  split_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____1___conv_pw_0 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pw_0(getitem_36);  getitem_36 = None
    getattr_getattr_l__mod___blocks___3_____1___conv_pw_1 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pw_1(getitem_37);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_171 = torch.cat([getattr_getattr_l__mod___blocks___3_____1___conv_pw_0, getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___3_____1___conv_pw_0 = getattr_getattr_l__mod___blocks___3_____1___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____1___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____1___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__24 = getattr_getattr_l__mod___blocks___3_____1___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____1___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_172 = torch.nn.functional.batch_norm(x_171, getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn1_running_var, getattr_getattr_l__mod___blocks___3_____1___bn1_weight, getattr_getattr_l__mod___blocks___3_____1___bn1_bias, True, 0.1, 0.001);  x_171 = getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = getattr_getattr_l__mod___blocks___3_____1___bn1_weight = getattr_getattr_l__mod___blocks___3_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_173 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_drop(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_175 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_act(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_17 = torch.functional.split(x_175, [156, 156, 156, 156], 1);  x_175 = None
    getitem_38 = split_17[0]
    getitem_39 = split_17[1]
    getitem_40 = split_17[2]
    getitem_41 = split_17[3];  split_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____1___conv_dw_0 = self.getattr_getattr_L__mod___blocks___3_____1___conv_dw_0(getitem_38);  getitem_38 = None
    getattr_getattr_l__mod___blocks___3_____1___conv_dw_1 = self.getattr_getattr_L__mod___blocks___3_____1___conv_dw_1(getitem_39);  getitem_39 = None
    getattr_getattr_l__mod___blocks___3_____1___conv_dw_2 = self.getattr_getattr_L__mod___blocks___3_____1___conv_dw_2(getitem_40);  getitem_40 = None
    getattr_getattr_l__mod___blocks___3_____1___conv_dw_3 = self.getattr_getattr_L__mod___blocks___3_____1___conv_dw_3(getitem_41);  getitem_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_177 = torch.cat([getattr_getattr_l__mod___blocks___3_____1___conv_dw_0, getattr_getattr_l__mod___blocks___3_____1___conv_dw_1, getattr_getattr_l__mod___blocks___3_____1___conv_dw_2, getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___3_____1___conv_dw_0 = getattr_getattr_l__mod___blocks___3_____1___conv_dw_1 = getattr_getattr_l__mod___blocks___3_____1___conv_dw_2 = getattr_getattr_l__mod___blocks___3_____1___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____1___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____1___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__25 = getattr_getattr_l__mod___blocks___3_____1___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____1___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_178 = torch.nn.functional.batch_norm(x_177, getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn2_running_var, getattr_getattr_l__mod___blocks___3_____1___bn2_weight, getattr_getattr_l__mod___blocks___3_____1___bn2_bias, True, 0.1, 0.001);  x_177 = getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = getattr_getattr_l__mod___blocks___3_____1___bn2_weight = getattr_getattr_l__mod___blocks___3_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_179 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_drop(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_181 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_act(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_181.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_21 = self.getattr_getattr_L__mod___blocks___3_____1___se_conv_reduce(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_22 = self.getattr_getattr_L__mod___blocks___3_____1___se_act1(x_se_21);  x_se_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_23 = self.getattr_getattr_L__mod___blocks___3_____1___se_conv_expand(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____1___se_gate = self.getattr_getattr_L__mod___blocks___3_____1___se_gate(x_se_23);  x_se_23 = None
    x_182 = x_181 * getattr_getattr_l__mod___blocks___3_____1___se_gate;  x_181 = getattr_getattr_l__mod___blocks___3_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_18 = torch.functional.split(x_182, [312, 312], 1);  x_182 = None
    getitem_42 = split_18[0]
    getitem_43 = split_18[1];  split_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pwl_0(getitem_42);  getitem_42 = None
    getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pwl_1(getitem_43);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_184 = torch.cat([getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____1___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____1___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__26 = getattr_getattr_l__mod___blocks___3_____1___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____1___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_185 = torch.nn.functional.batch_norm(x_184, getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn3_running_var, getattr_getattr_l__mod___blocks___3_____1___bn3_weight, getattr_getattr_l__mod___blocks___3_____1___bn3_bias, True, 0.1, 0.001);  x_184 = getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = getattr_getattr_l__mod___blocks___3_____1___bn3_weight = getattr_getattr_l__mod___blocks___3_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_186 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_drop(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_188 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_act(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____1___drop_path = self.getattr_getattr_L__mod___blocks___3_____1___drop_path(x_188);  x_188 = None
    shortcut_9 = getattr_getattr_l__mod___blocks___3_____1___drop_path + shortcut_8;  getattr_getattr_l__mod___blocks___3_____1___drop_path = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_19 = torch.functional.split(shortcut_9, [52, 52], 1)
    getitem_44 = split_19[0]
    getitem_45 = split_19[1];  split_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____2___conv_pw_0 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pw_0(getitem_44);  getitem_44 = None
    getattr_getattr_l__mod___blocks___3_____2___conv_pw_1 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pw_1(getitem_45);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_191 = torch.cat([getattr_getattr_l__mod___blocks___3_____2___conv_pw_0, getattr_getattr_l__mod___blocks___3_____2___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___3_____2___conv_pw_0 = getattr_getattr_l__mod___blocks___3_____2___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____2___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____2___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__27 = getattr_getattr_l__mod___blocks___3_____2___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____2___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_192 = torch.nn.functional.batch_norm(x_191, getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn1_running_var, getattr_getattr_l__mod___blocks___3_____2___bn1_weight, getattr_getattr_l__mod___blocks___3_____2___bn1_bias, True, 0.1, 0.001);  x_191 = getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = getattr_getattr_l__mod___blocks___3_____2___bn1_weight = getattr_getattr_l__mod___blocks___3_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_193 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_drop(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_195 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_act(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_20 = torch.functional.split(x_195, [156, 156, 156, 156], 1);  x_195 = None
    getitem_46 = split_20[0]
    getitem_47 = split_20[1]
    getitem_48 = split_20[2]
    getitem_49 = split_20[3];  split_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____2___conv_dw_0 = self.getattr_getattr_L__mod___blocks___3_____2___conv_dw_0(getitem_46);  getitem_46 = None
    getattr_getattr_l__mod___blocks___3_____2___conv_dw_1 = self.getattr_getattr_L__mod___blocks___3_____2___conv_dw_1(getitem_47);  getitem_47 = None
    getattr_getattr_l__mod___blocks___3_____2___conv_dw_2 = self.getattr_getattr_L__mod___blocks___3_____2___conv_dw_2(getitem_48);  getitem_48 = None
    getattr_getattr_l__mod___blocks___3_____2___conv_dw_3 = self.getattr_getattr_L__mod___blocks___3_____2___conv_dw_3(getitem_49);  getitem_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_197 = torch.cat([getattr_getattr_l__mod___blocks___3_____2___conv_dw_0, getattr_getattr_l__mod___blocks___3_____2___conv_dw_1, getattr_getattr_l__mod___blocks___3_____2___conv_dw_2, getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___3_____2___conv_dw_0 = getattr_getattr_l__mod___blocks___3_____2___conv_dw_1 = getattr_getattr_l__mod___blocks___3_____2___conv_dw_2 = getattr_getattr_l__mod___blocks___3_____2___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____2___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____2___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__28 = getattr_getattr_l__mod___blocks___3_____2___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____2___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_198 = torch.nn.functional.batch_norm(x_197, getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn2_running_var, getattr_getattr_l__mod___blocks___3_____2___bn2_weight, getattr_getattr_l__mod___blocks___3_____2___bn2_bias, True, 0.1, 0.001);  x_197 = getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = getattr_getattr_l__mod___blocks___3_____2___bn2_weight = getattr_getattr_l__mod___blocks___3_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_199 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_drop(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_201 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_act(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_201.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_25 = self.getattr_getattr_L__mod___blocks___3_____2___se_conv_reduce(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_26 = self.getattr_getattr_L__mod___blocks___3_____2___se_act1(x_se_25);  x_se_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_27 = self.getattr_getattr_L__mod___blocks___3_____2___se_conv_expand(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____2___se_gate = self.getattr_getattr_L__mod___blocks___3_____2___se_gate(x_se_27);  x_se_27 = None
    x_202 = x_201 * getattr_getattr_l__mod___blocks___3_____2___se_gate;  x_201 = getattr_getattr_l__mod___blocks___3_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_21 = torch.functional.split(x_202, [312, 312], 1);  x_202 = None
    getitem_50 = split_21[0]
    getitem_51 = split_21[1];  split_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pwl_0(getitem_50);  getitem_50 = None
    getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pwl_1(getitem_51);  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_204 = torch.cat([getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0, getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0 = getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____2___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____2___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__29 = getattr_getattr_l__mod___blocks___3_____2___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____2___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_205 = torch.nn.functional.batch_norm(x_204, getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn3_running_var, getattr_getattr_l__mod___blocks___3_____2___bn3_weight, getattr_getattr_l__mod___blocks___3_____2___bn3_bias, True, 0.1, 0.001);  x_204 = getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = getattr_getattr_l__mod___blocks___3_____2___bn3_weight = getattr_getattr_l__mod___blocks___3_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_206 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_drop(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_208 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_act(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____2___drop_path = self.getattr_getattr_L__mod___blocks___3_____2___drop_path(x_208);  x_208 = None
    shortcut_10 = getattr_getattr_l__mod___blocks___3_____2___drop_path + shortcut_9;  getattr_getattr_l__mod___blocks___3_____2___drop_path = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_22 = torch.functional.split(shortcut_10, [52, 52], 1)
    getitem_52 = split_22[0]
    getitem_53 = split_22[1];  split_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____3___conv_pw_0 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pw_0(getitem_52);  getitem_52 = None
    getattr_getattr_l__mod___blocks___3_____3___conv_pw_1 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pw_1(getitem_53);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_211 = torch.cat([getattr_getattr_l__mod___blocks___3_____3___conv_pw_0, getattr_getattr_l__mod___blocks___3_____3___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___3_____3___conv_pw_0 = getattr_getattr_l__mod___blocks___3_____3___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____3___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____3___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__30 = getattr_getattr_l__mod___blocks___3_____3___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____3___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_212 = torch.nn.functional.batch_norm(x_211, getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn1_running_var, getattr_getattr_l__mod___blocks___3_____3___bn1_weight, getattr_getattr_l__mod___blocks___3_____3___bn1_bias, True, 0.1, 0.001);  x_211 = getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = getattr_getattr_l__mod___blocks___3_____3___bn1_weight = getattr_getattr_l__mod___blocks___3_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_213 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_drop(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_215 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_act(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_23 = torch.functional.split(x_215, [156, 156, 156, 156], 1);  x_215 = None
    getitem_54 = split_23[0]
    getitem_55 = split_23[1]
    getitem_56 = split_23[2]
    getitem_57 = split_23[3];  split_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____3___conv_dw_0 = self.getattr_getattr_L__mod___blocks___3_____3___conv_dw_0(getitem_54);  getitem_54 = None
    getattr_getattr_l__mod___blocks___3_____3___conv_dw_1 = self.getattr_getattr_L__mod___blocks___3_____3___conv_dw_1(getitem_55);  getitem_55 = None
    getattr_getattr_l__mod___blocks___3_____3___conv_dw_2 = self.getattr_getattr_L__mod___blocks___3_____3___conv_dw_2(getitem_56);  getitem_56 = None
    getattr_getattr_l__mod___blocks___3_____3___conv_dw_3 = self.getattr_getattr_L__mod___blocks___3_____3___conv_dw_3(getitem_57);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_217 = torch.cat([getattr_getattr_l__mod___blocks___3_____3___conv_dw_0, getattr_getattr_l__mod___blocks___3_____3___conv_dw_1, getattr_getattr_l__mod___blocks___3_____3___conv_dw_2, getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___3_____3___conv_dw_0 = getattr_getattr_l__mod___blocks___3_____3___conv_dw_1 = getattr_getattr_l__mod___blocks___3_____3___conv_dw_2 = getattr_getattr_l__mod___blocks___3_____3___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____3___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____3___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__31 = getattr_getattr_l__mod___blocks___3_____3___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____3___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_218 = torch.nn.functional.batch_norm(x_217, getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn2_running_var, getattr_getattr_l__mod___blocks___3_____3___bn2_weight, getattr_getattr_l__mod___blocks___3_____3___bn2_bias, True, 0.1, 0.001);  x_217 = getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = getattr_getattr_l__mod___blocks___3_____3___bn2_weight = getattr_getattr_l__mod___blocks___3_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_219 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_drop(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_221 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_act(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = x_221.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_29 = self.getattr_getattr_L__mod___blocks___3_____3___se_conv_reduce(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_30 = self.getattr_getattr_L__mod___blocks___3_____3___se_act1(x_se_29);  x_se_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_31 = self.getattr_getattr_L__mod___blocks___3_____3___se_conv_expand(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____3___se_gate = self.getattr_getattr_L__mod___blocks___3_____3___se_gate(x_se_31);  x_se_31 = None
    x_222 = x_221 * getattr_getattr_l__mod___blocks___3_____3___se_gate;  x_221 = getattr_getattr_l__mod___blocks___3_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_24 = torch.functional.split(x_222, [312, 312], 1);  x_222 = None
    getitem_58 = split_24[0]
    getitem_59 = split_24[1];  split_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pwl_0(getitem_58);  getitem_58 = None
    getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pwl_1(getitem_59);  getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_224 = torch.cat([getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0, getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0 = getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____3___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____3___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__32 = getattr_getattr_l__mod___blocks___3_____3___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____3___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_225 = torch.nn.functional.batch_norm(x_224, getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn3_running_var, getattr_getattr_l__mod___blocks___3_____3___bn3_weight, getattr_getattr_l__mod___blocks___3_____3___bn3_bias, True, 0.1, 0.001);  x_224 = getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = getattr_getattr_l__mod___blocks___3_____3___bn3_weight = getattr_getattr_l__mod___blocks___3_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_226 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_drop(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_228 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_act(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____3___drop_path = self.getattr_getattr_L__mod___blocks___3_____3___drop_path(x_228);  x_228 = None
    shortcut_11 = getattr_getattr_l__mod___blocks___3_____3___drop_path + shortcut_10;  getattr_getattr_l__mod___blocks___3_____3___drop_path = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_230 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pw(shortcut_11);  shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__33 = getattr_getattr_l__mod___blocks___4_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_231 = torch.nn.functional.batch_norm(x_230, getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn1_running_var, getattr_getattr_l__mod___blocks___4_____0___bn1_weight, getattr_getattr_l__mod___blocks___4_____0___bn1_bias, True, 0.1, 0.001);  x_230 = getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = getattr_getattr_l__mod___blocks___4_____0___bn1_weight = getattr_getattr_l__mod___blocks___4_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_232 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_drop(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_234 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_act(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_235 = self.getattr_getattr_L__mod___blocks___4_____0___conv_dw(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__34 = getattr_getattr_l__mod___blocks___4_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_236 = torch.nn.functional.batch_norm(x_235, getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn2_running_var, getattr_getattr_l__mod___blocks___4_____0___bn2_weight, getattr_getattr_l__mod___blocks___4_____0___bn2_bias, True, 0.1, 0.001);  x_235 = getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = getattr_getattr_l__mod___blocks___4_____0___bn2_weight = getattr_getattr_l__mod___blocks___4_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_237 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_drop(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_239 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_act(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_32 = x_239.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_33 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_reduce(x_se_32);  x_se_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_34 = self.getattr_getattr_L__mod___blocks___4_____0___se_act1(x_se_33);  x_se_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_35 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_expand(x_se_34);  x_se_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____0___se_gate = self.getattr_getattr_L__mod___blocks___4_____0___se_gate(x_se_35);  x_se_35 = None
    x_240 = x_239 * getattr_getattr_l__mod___blocks___4_____0___se_gate;  x_239 = getattr_getattr_l__mod___blocks___4_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_241 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pwl(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__35 = getattr_getattr_l__mod___blocks___4_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_242 = torch.nn.functional.batch_norm(x_241, getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn3_running_var, getattr_getattr_l__mod___blocks___4_____0___bn3_weight, getattr_getattr_l__mod___blocks___4_____0___bn3_bias, True, 0.1, 0.001);  x_241 = getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = getattr_getattr_l__mod___blocks___4_____0___bn3_weight = getattr_getattr_l__mod___blocks___4_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_243 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_drop(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_12 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_act(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_25 = torch.functional.split(shortcut_12, [80, 80], 1)
    getitem_60 = split_25[0]
    getitem_61 = split_25[1];  split_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____1___conv_pw_0 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pw_0(getitem_60);  getitem_60 = None
    getattr_getattr_l__mod___blocks___4_____1___conv_pw_1 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pw_1(getitem_61);  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_247 = torch.cat([getattr_getattr_l__mod___blocks___4_____1___conv_pw_0, getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___4_____1___conv_pw_0 = getattr_getattr_l__mod___blocks___4_____1___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____1___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____1___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__36 = getattr_getattr_l__mod___blocks___4_____1___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____1___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_248 = torch.nn.functional.batch_norm(x_247, getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn1_running_var, getattr_getattr_l__mod___blocks___4_____1___bn1_weight, getattr_getattr_l__mod___blocks___4_____1___bn1_bias, True, 0.1, 0.001);  x_247 = getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = getattr_getattr_l__mod___blocks___4_____1___bn1_weight = getattr_getattr_l__mod___blocks___4_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_249 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_drop(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_251 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_act(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_26 = torch.functional.split(x_251, [120, 120, 120, 120], 1);  x_251 = None
    getitem_62 = split_26[0]
    getitem_63 = split_26[1]
    getitem_64 = split_26[2]
    getitem_65 = split_26[3];  split_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____1___conv_dw_0 = self.getattr_getattr_L__mod___blocks___4_____1___conv_dw_0(getitem_62);  getitem_62 = None
    getattr_getattr_l__mod___blocks___4_____1___conv_dw_1 = self.getattr_getattr_L__mod___blocks___4_____1___conv_dw_1(getitem_63);  getitem_63 = None
    getattr_getattr_l__mod___blocks___4_____1___conv_dw_2 = self.getattr_getattr_L__mod___blocks___4_____1___conv_dw_2(getitem_64);  getitem_64 = None
    getattr_getattr_l__mod___blocks___4_____1___conv_dw_3 = self.getattr_getattr_L__mod___blocks___4_____1___conv_dw_3(getitem_65);  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_253 = torch.cat([getattr_getattr_l__mod___blocks___4_____1___conv_dw_0, getattr_getattr_l__mod___blocks___4_____1___conv_dw_1, getattr_getattr_l__mod___blocks___4_____1___conv_dw_2, getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___4_____1___conv_dw_0 = getattr_getattr_l__mod___blocks___4_____1___conv_dw_1 = getattr_getattr_l__mod___blocks___4_____1___conv_dw_2 = getattr_getattr_l__mod___blocks___4_____1___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____1___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____1___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__37 = getattr_getattr_l__mod___blocks___4_____1___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____1___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_254 = torch.nn.functional.batch_norm(x_253, getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn2_running_var, getattr_getattr_l__mod___blocks___4_____1___bn2_weight, getattr_getattr_l__mod___blocks___4_____1___bn2_bias, True, 0.1, 0.001);  x_253 = getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = getattr_getattr_l__mod___blocks___4_____1___bn2_weight = getattr_getattr_l__mod___blocks___4_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_255 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_drop(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_257 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_act(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_36 = x_257.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_37 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_reduce(x_se_36);  x_se_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_38 = self.getattr_getattr_L__mod___blocks___4_____1___se_act1(x_se_37);  x_se_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_39 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_expand(x_se_38);  x_se_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____1___se_gate = self.getattr_getattr_L__mod___blocks___4_____1___se_gate(x_se_39);  x_se_39 = None
    x_258 = x_257 * getattr_getattr_l__mod___blocks___4_____1___se_gate;  x_257 = getattr_getattr_l__mod___blocks___4_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_27 = torch.functional.split(x_258, [240, 240], 1);  x_258 = None
    getitem_66 = split_27[0]
    getitem_67 = split_27[1];  split_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pwl_0(getitem_66);  getitem_66 = None
    getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pwl_1(getitem_67);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_260 = torch.cat([getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____1___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____1___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__38 = getattr_getattr_l__mod___blocks___4_____1___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____1___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_261 = torch.nn.functional.batch_norm(x_260, getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn3_running_var, getattr_getattr_l__mod___blocks___4_____1___bn3_weight, getattr_getattr_l__mod___blocks___4_____1___bn3_bias, True, 0.1, 0.001);  x_260 = getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = getattr_getattr_l__mod___blocks___4_____1___bn3_weight = getattr_getattr_l__mod___blocks___4_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_262 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_drop(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_264 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_act(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____1___drop_path = self.getattr_getattr_L__mod___blocks___4_____1___drop_path(x_264);  x_264 = None
    shortcut_13 = getattr_getattr_l__mod___blocks___4_____1___drop_path + shortcut_12;  getattr_getattr_l__mod___blocks___4_____1___drop_path = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_28 = torch.functional.split(shortcut_13, [80, 80], 1)
    getitem_68 = split_28[0]
    getitem_69 = split_28[1];  split_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____2___conv_pw_0 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pw_0(getitem_68);  getitem_68 = None
    getattr_getattr_l__mod___blocks___4_____2___conv_pw_1 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pw_1(getitem_69);  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_267 = torch.cat([getattr_getattr_l__mod___blocks___4_____2___conv_pw_0, getattr_getattr_l__mod___blocks___4_____2___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___4_____2___conv_pw_0 = getattr_getattr_l__mod___blocks___4_____2___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____2___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____2___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__39 = getattr_getattr_l__mod___blocks___4_____2___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____2___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_268 = torch.nn.functional.batch_norm(x_267, getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn1_running_var, getattr_getattr_l__mod___blocks___4_____2___bn1_weight, getattr_getattr_l__mod___blocks___4_____2___bn1_bias, True, 0.1, 0.001);  x_267 = getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = getattr_getattr_l__mod___blocks___4_____2___bn1_weight = getattr_getattr_l__mod___blocks___4_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_269 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_drop(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_271 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_act(x_269);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_29 = torch.functional.split(x_271, [120, 120, 120, 120], 1);  x_271 = None
    getitem_70 = split_29[0]
    getitem_71 = split_29[1]
    getitem_72 = split_29[2]
    getitem_73 = split_29[3];  split_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____2___conv_dw_0 = self.getattr_getattr_L__mod___blocks___4_____2___conv_dw_0(getitem_70);  getitem_70 = None
    getattr_getattr_l__mod___blocks___4_____2___conv_dw_1 = self.getattr_getattr_L__mod___blocks___4_____2___conv_dw_1(getitem_71);  getitem_71 = None
    getattr_getattr_l__mod___blocks___4_____2___conv_dw_2 = self.getattr_getattr_L__mod___blocks___4_____2___conv_dw_2(getitem_72);  getitem_72 = None
    getattr_getattr_l__mod___blocks___4_____2___conv_dw_3 = self.getattr_getattr_L__mod___blocks___4_____2___conv_dw_3(getitem_73);  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_273 = torch.cat([getattr_getattr_l__mod___blocks___4_____2___conv_dw_0, getattr_getattr_l__mod___blocks___4_____2___conv_dw_1, getattr_getattr_l__mod___blocks___4_____2___conv_dw_2, getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___4_____2___conv_dw_0 = getattr_getattr_l__mod___blocks___4_____2___conv_dw_1 = getattr_getattr_l__mod___blocks___4_____2___conv_dw_2 = getattr_getattr_l__mod___blocks___4_____2___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____2___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____2___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__40 = getattr_getattr_l__mod___blocks___4_____2___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____2___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_274 = torch.nn.functional.batch_norm(x_273, getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn2_running_var, getattr_getattr_l__mod___blocks___4_____2___bn2_weight, getattr_getattr_l__mod___blocks___4_____2___bn2_bias, True, 0.1, 0.001);  x_273 = getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = getattr_getattr_l__mod___blocks___4_____2___bn2_weight = getattr_getattr_l__mod___blocks___4_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_275 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_drop(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_277 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_act(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_40 = x_277.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_41 = self.getattr_getattr_L__mod___blocks___4_____2___se_conv_reduce(x_se_40);  x_se_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_42 = self.getattr_getattr_L__mod___blocks___4_____2___se_act1(x_se_41);  x_se_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_43 = self.getattr_getattr_L__mod___blocks___4_____2___se_conv_expand(x_se_42);  x_se_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____2___se_gate = self.getattr_getattr_L__mod___blocks___4_____2___se_gate(x_se_43);  x_se_43 = None
    x_278 = x_277 * getattr_getattr_l__mod___blocks___4_____2___se_gate;  x_277 = getattr_getattr_l__mod___blocks___4_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_30 = torch.functional.split(x_278, [240, 240], 1);  x_278 = None
    getitem_74 = split_30[0]
    getitem_75 = split_30[1];  split_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pwl_0(getitem_74);  getitem_74 = None
    getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pwl_1(getitem_75);  getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_280 = torch.cat([getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0, getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0 = getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____2___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____2___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__41 = getattr_getattr_l__mod___blocks___4_____2___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____2___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_281 = torch.nn.functional.batch_norm(x_280, getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn3_running_var, getattr_getattr_l__mod___blocks___4_____2___bn3_weight, getattr_getattr_l__mod___blocks___4_____2___bn3_bias, True, 0.1, 0.001);  x_280 = getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = getattr_getattr_l__mod___blocks___4_____2___bn3_weight = getattr_getattr_l__mod___blocks___4_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_282 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_drop(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_284 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_act(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____2___drop_path = self.getattr_getattr_L__mod___blocks___4_____2___drop_path(x_284);  x_284 = None
    shortcut_14 = getattr_getattr_l__mod___blocks___4_____2___drop_path + shortcut_13;  getattr_getattr_l__mod___blocks___4_____2___drop_path = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_31 = torch.functional.split(shortcut_14, [80, 80], 1)
    getitem_76 = split_31[0]
    getitem_77 = split_31[1];  split_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____3___conv_pw_0 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pw_0(getitem_76);  getitem_76 = None
    getattr_getattr_l__mod___blocks___4_____3___conv_pw_1 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pw_1(getitem_77);  getitem_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_287 = torch.cat([getattr_getattr_l__mod___blocks___4_____3___conv_pw_0, getattr_getattr_l__mod___blocks___4_____3___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___4_____3___conv_pw_0 = getattr_getattr_l__mod___blocks___4_____3___conv_pw_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____3___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____3___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__42 = getattr_getattr_l__mod___blocks___4_____3___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____3___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_288 = torch.nn.functional.batch_norm(x_287, getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn1_running_var, getattr_getattr_l__mod___blocks___4_____3___bn1_weight, getattr_getattr_l__mod___blocks___4_____3___bn1_bias, True, 0.1, 0.001);  x_287 = getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn1_running_var = getattr_getattr_l__mod___blocks___4_____3___bn1_weight = getattr_getattr_l__mod___blocks___4_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_289 = self.getattr_getattr_L__mod___blocks___4_____3___bn1_drop(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_291 = self.getattr_getattr_L__mod___blocks___4_____3___bn1_act(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_32 = torch.functional.split(x_291, [120, 120, 120, 120], 1);  x_291 = None
    getitem_78 = split_32[0]
    getitem_79 = split_32[1]
    getitem_80 = split_32[2]
    getitem_81 = split_32[3];  split_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____3___conv_dw_0 = self.getattr_getattr_L__mod___blocks___4_____3___conv_dw_0(getitem_78);  getitem_78 = None
    getattr_getattr_l__mod___blocks___4_____3___conv_dw_1 = self.getattr_getattr_L__mod___blocks___4_____3___conv_dw_1(getitem_79);  getitem_79 = None
    getattr_getattr_l__mod___blocks___4_____3___conv_dw_2 = self.getattr_getattr_L__mod___blocks___4_____3___conv_dw_2(getitem_80);  getitem_80 = None
    getattr_getattr_l__mod___blocks___4_____3___conv_dw_3 = self.getattr_getattr_L__mod___blocks___4_____3___conv_dw_3(getitem_81);  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_293 = torch.cat([getattr_getattr_l__mod___blocks___4_____3___conv_dw_0, getattr_getattr_l__mod___blocks___4_____3___conv_dw_1, getattr_getattr_l__mod___blocks___4_____3___conv_dw_2, getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___4_____3___conv_dw_0 = getattr_getattr_l__mod___blocks___4_____3___conv_dw_1 = getattr_getattr_l__mod___blocks___4_____3___conv_dw_2 = getattr_getattr_l__mod___blocks___4_____3___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____3___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____3___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__43 = getattr_getattr_l__mod___blocks___4_____3___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____3___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_294 = torch.nn.functional.batch_norm(x_293, getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn2_running_var, getattr_getattr_l__mod___blocks___4_____3___bn2_weight, getattr_getattr_l__mod___blocks___4_____3___bn2_bias, True, 0.1, 0.001);  x_293 = getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn2_running_var = getattr_getattr_l__mod___blocks___4_____3___bn2_weight = getattr_getattr_l__mod___blocks___4_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_295 = self.getattr_getattr_L__mod___blocks___4_____3___bn2_drop(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_297 = self.getattr_getattr_L__mod___blocks___4_____3___bn2_act(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_44 = x_297.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_45 = self.getattr_getattr_L__mod___blocks___4_____3___se_conv_reduce(x_se_44);  x_se_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_46 = self.getattr_getattr_L__mod___blocks___4_____3___se_act1(x_se_45);  x_se_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_47 = self.getattr_getattr_L__mod___blocks___4_____3___se_conv_expand(x_se_46);  x_se_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____3___se_gate = self.getattr_getattr_L__mod___blocks___4_____3___se_gate(x_se_47);  x_se_47 = None
    x_298 = x_297 * getattr_getattr_l__mod___blocks___4_____3___se_gate;  x_297 = getattr_getattr_l__mod___blocks___4_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_33 = torch.functional.split(x_298, [240, 240], 1);  x_298 = None
    getitem_82 = split_33[0]
    getitem_83 = split_33[1];  split_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pwl_0(getitem_82);  getitem_82 = None
    getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pwl_1(getitem_83);  getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_300 = torch.cat([getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0, getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0 = getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____3___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____3___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__44 = getattr_getattr_l__mod___blocks___4_____3___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____3___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_301 = torch.nn.functional.batch_norm(x_300, getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn3_running_var, getattr_getattr_l__mod___blocks___4_____3___bn3_weight, getattr_getattr_l__mod___blocks___4_____3___bn3_bias, True, 0.1, 0.001);  x_300 = getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn3_running_var = getattr_getattr_l__mod___blocks___4_____3___bn3_weight = getattr_getattr_l__mod___blocks___4_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_302 = self.getattr_getattr_L__mod___blocks___4_____3___bn3_drop(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_304 = self.getattr_getattr_L__mod___blocks___4_____3___bn3_act(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____3___drop_path = self.getattr_getattr_L__mod___blocks___4_____3___drop_path(x_304);  x_304 = None
    shortcut_15 = getattr_getattr_l__mod___blocks___4_____3___drop_path + shortcut_14;  getattr_getattr_l__mod___blocks___4_____3___drop_path = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_306 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pw(shortcut_15);  shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__45 = getattr_getattr_l__mod___blocks___5_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_307 = torch.nn.functional.batch_norm(x_306, getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn1_running_var, getattr_getattr_l__mod___blocks___5_____0___bn1_weight, getattr_getattr_l__mod___blocks___5_____0___bn1_bias, True, 0.1, 0.001);  x_306 = getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = getattr_getattr_l__mod___blocks___5_____0___bn1_weight = getattr_getattr_l__mod___blocks___5_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_308 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_drop(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_310 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_act(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_34 = torch.functional.split(x_310, [240, 240, 240, 240], 1);  x_310 = None
    getitem_84 = split_34[0]
    getitem_85 = split_34[1]
    getitem_86 = split_34[2]
    getitem_87 = split_34[3];  split_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___0___weight = self.getattr_getattr_getattr_L__mod___blocks___5_____0___conv_dw___0___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_312 = torch.nn.functional.pad(getitem_84, (0, 1, 0, 1), value = 0);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_11 = torch.conv2d(x_312, getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___0___weight, None, (2, 2), (0, 0), (1, 1), 240);  x_312 = getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___0___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___1___weight = self.getattr_getattr_getattr_L__mod___blocks___5_____0___conv_dw___1___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_314 = torch.nn.functional.pad(getitem_85, (1, 2, 1, 2), value = 0);  getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_12 = torch.conv2d(x_314, getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___1___weight, None, (2, 2), (0, 0), (1, 1), 240);  x_314 = getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___1___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___2___weight = self.getattr_getattr_getattr_L__mod___blocks___5_____0___conv_dw___2___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_316 = torch.nn.functional.pad(getitem_86, (2, 3, 2, 3), value = 0);  getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_13 = torch.conv2d(x_316, getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___2___weight, None, (2, 2), (0, 0), (1, 1), 240);  x_316 = getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___2___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___3___weight = self.getattr_getattr_getattr_L__mod___blocks___5_____0___conv_dw___3___weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_318 = torch.nn.functional.pad(getitem_87, (3, 4, 3, 4), value = 0);  getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    conv2d_14 = torch.conv2d(x_318, getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___3___weight, None, (2, 2), (0, 0), (1, 1), 240);  x_318 = getattr_getattr_getattr_l__mod___blocks___5_____0___conv_dw___3___weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_320 = torch.cat([conv2d_11, conv2d_12, conv2d_13, conv2d_14], 1);  conv2d_11 = conv2d_12 = conv2d_13 = conv2d_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__46 = getattr_getattr_l__mod___blocks___5_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_321 = torch.nn.functional.batch_norm(x_320, getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn2_running_var, getattr_getattr_l__mod___blocks___5_____0___bn2_weight, getattr_getattr_l__mod___blocks___5_____0___bn2_bias, True, 0.1, 0.001);  x_320 = getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = getattr_getattr_l__mod___blocks___5_____0___bn2_weight = getattr_getattr_l__mod___blocks___5_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_322 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_drop(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_324 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_act(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_48 = x_324.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_49 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_reduce(x_se_48);  x_se_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_50 = self.getattr_getattr_L__mod___blocks___5_____0___se_act1(x_se_49);  x_se_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_51 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_expand(x_se_50);  x_se_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____0___se_gate = self.getattr_getattr_L__mod___blocks___5_____0___se_gate(x_se_51);  x_se_51 = None
    x_325 = x_324 * getattr_getattr_l__mod___blocks___5_____0___se_gate;  x_324 = getattr_getattr_l__mod___blocks___5_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_326 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pwl(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__47 = getattr_getattr_l__mod___blocks___5_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_327 = torch.nn.functional.batch_norm(x_326, getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn3_running_var, getattr_getattr_l__mod___blocks___5_____0___bn3_weight, getattr_getattr_l__mod___blocks___5_____0___bn3_bias, True, 0.1, 0.001);  x_326 = getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = getattr_getattr_l__mod___blocks___5_____0___bn3_weight = getattr_getattr_l__mod___blocks___5_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_328 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_drop(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_16 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_act(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_331 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pw(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____1___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____1___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__48 = getattr_getattr_l__mod___blocks___5_____1___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____1___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_332 = torch.nn.functional.batch_norm(x_331, getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn1_running_var, getattr_getattr_l__mod___blocks___5_____1___bn1_weight, getattr_getattr_l__mod___blocks___5_____1___bn1_bias, True, 0.1, 0.001);  x_331 = getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = getattr_getattr_l__mod___blocks___5_____1___bn1_weight = getattr_getattr_l__mod___blocks___5_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_333 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_drop(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_335 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_act(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_35 = torch.functional.split(x_335, [396, 396, 396, 396], 1);  x_335 = None
    getitem_88 = split_35[0]
    getitem_89 = split_35[1]
    getitem_90 = split_35[2]
    getitem_91 = split_35[3];  split_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____1___conv_dw_0 = self.getattr_getattr_L__mod___blocks___5_____1___conv_dw_0(getitem_88);  getitem_88 = None
    getattr_getattr_l__mod___blocks___5_____1___conv_dw_1 = self.getattr_getattr_L__mod___blocks___5_____1___conv_dw_1(getitem_89);  getitem_89 = None
    getattr_getattr_l__mod___blocks___5_____1___conv_dw_2 = self.getattr_getattr_L__mod___blocks___5_____1___conv_dw_2(getitem_90);  getitem_90 = None
    getattr_getattr_l__mod___blocks___5_____1___conv_dw_3 = self.getattr_getattr_L__mod___blocks___5_____1___conv_dw_3(getitem_91);  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_337 = torch.cat([getattr_getattr_l__mod___blocks___5_____1___conv_dw_0, getattr_getattr_l__mod___blocks___5_____1___conv_dw_1, getattr_getattr_l__mod___blocks___5_____1___conv_dw_2, getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___5_____1___conv_dw_0 = getattr_getattr_l__mod___blocks___5_____1___conv_dw_1 = getattr_getattr_l__mod___blocks___5_____1___conv_dw_2 = getattr_getattr_l__mod___blocks___5_____1___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____1___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____1___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__49 = getattr_getattr_l__mod___blocks___5_____1___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____1___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_338 = torch.nn.functional.batch_norm(x_337, getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn2_running_var, getattr_getattr_l__mod___blocks___5_____1___bn2_weight, getattr_getattr_l__mod___blocks___5_____1___bn2_bias, True, 0.1, 0.001);  x_337 = getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = getattr_getattr_l__mod___blocks___5_____1___bn2_weight = getattr_getattr_l__mod___blocks___5_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_339 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_drop(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_341 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_act(x_339);  x_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_52 = x_341.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_53 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_reduce(x_se_52);  x_se_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_54 = self.getattr_getattr_L__mod___blocks___5_____1___se_act1(x_se_53);  x_se_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_55 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_expand(x_se_54);  x_se_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____1___se_gate = self.getattr_getattr_L__mod___blocks___5_____1___se_gate(x_se_55);  x_se_55 = None
    x_342 = x_341 * getattr_getattr_l__mod___blocks___5_____1___se_gate;  x_341 = getattr_getattr_l__mod___blocks___5_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_36 = torch.functional.split(x_342, [792, 792], 1);  x_342 = None
    getitem_92 = split_36[0]
    getitem_93 = split_36[1];  split_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pwl_0(getitem_92);  getitem_92 = None
    getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pwl_1(getitem_93);  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_344 = torch.cat([getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____1___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____1___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__50 = getattr_getattr_l__mod___blocks___5_____1___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____1___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_345 = torch.nn.functional.batch_norm(x_344, getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn3_running_var, getattr_getattr_l__mod___blocks___5_____1___bn3_weight, getattr_getattr_l__mod___blocks___5_____1___bn3_bias, True, 0.1, 0.001);  x_344 = getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = getattr_getattr_l__mod___blocks___5_____1___bn3_weight = getattr_getattr_l__mod___blocks___5_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_346 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_drop(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_348 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_act(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____1___drop_path = self.getattr_getattr_L__mod___blocks___5_____1___drop_path(x_348);  x_348 = None
    shortcut_17 = getattr_getattr_l__mod___blocks___5_____1___drop_path + shortcut_16;  getattr_getattr_l__mod___blocks___5_____1___drop_path = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_350 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pw(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____2___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____2___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__51 = getattr_getattr_l__mod___blocks___5_____2___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____2___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_351 = torch.nn.functional.batch_norm(x_350, getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn1_running_var, getattr_getattr_l__mod___blocks___5_____2___bn1_weight, getattr_getattr_l__mod___blocks___5_____2___bn1_bias, True, 0.1, 0.001);  x_350 = getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = getattr_getattr_l__mod___blocks___5_____2___bn1_weight = getattr_getattr_l__mod___blocks___5_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_352 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_drop(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_354 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_act(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_37 = torch.functional.split(x_354, [396, 396, 396, 396], 1);  x_354 = None
    getitem_94 = split_37[0]
    getitem_95 = split_37[1]
    getitem_96 = split_37[2]
    getitem_97 = split_37[3];  split_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____2___conv_dw_0 = self.getattr_getattr_L__mod___blocks___5_____2___conv_dw_0(getitem_94);  getitem_94 = None
    getattr_getattr_l__mod___blocks___5_____2___conv_dw_1 = self.getattr_getattr_L__mod___blocks___5_____2___conv_dw_1(getitem_95);  getitem_95 = None
    getattr_getattr_l__mod___blocks___5_____2___conv_dw_2 = self.getattr_getattr_L__mod___blocks___5_____2___conv_dw_2(getitem_96);  getitem_96 = None
    getattr_getattr_l__mod___blocks___5_____2___conv_dw_3 = self.getattr_getattr_L__mod___blocks___5_____2___conv_dw_3(getitem_97);  getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_356 = torch.cat([getattr_getattr_l__mod___blocks___5_____2___conv_dw_0, getattr_getattr_l__mod___blocks___5_____2___conv_dw_1, getattr_getattr_l__mod___blocks___5_____2___conv_dw_2, getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___5_____2___conv_dw_0 = getattr_getattr_l__mod___blocks___5_____2___conv_dw_1 = getattr_getattr_l__mod___blocks___5_____2___conv_dw_2 = getattr_getattr_l__mod___blocks___5_____2___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____2___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____2___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__52 = getattr_getattr_l__mod___blocks___5_____2___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____2___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_357 = torch.nn.functional.batch_norm(x_356, getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn2_running_var, getattr_getattr_l__mod___blocks___5_____2___bn2_weight, getattr_getattr_l__mod___blocks___5_____2___bn2_bias, True, 0.1, 0.001);  x_356 = getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = getattr_getattr_l__mod___blocks___5_____2___bn2_weight = getattr_getattr_l__mod___blocks___5_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_358 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_drop(x_357);  x_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_360 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_act(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_56 = x_360.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_57 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_reduce(x_se_56);  x_se_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_58 = self.getattr_getattr_L__mod___blocks___5_____2___se_act1(x_se_57);  x_se_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_59 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_expand(x_se_58);  x_se_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____2___se_gate = self.getattr_getattr_L__mod___blocks___5_____2___se_gate(x_se_59);  x_se_59 = None
    x_361 = x_360 * getattr_getattr_l__mod___blocks___5_____2___se_gate;  x_360 = getattr_getattr_l__mod___blocks___5_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_38 = torch.functional.split(x_361, [792, 792], 1);  x_361 = None
    getitem_98 = split_38[0]
    getitem_99 = split_38[1];  split_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pwl_0(getitem_98);  getitem_98 = None
    getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pwl_1(getitem_99);  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_363 = torch.cat([getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0, getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0 = getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____2___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____2___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__53 = getattr_getattr_l__mod___blocks___5_____2___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____2___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_364 = torch.nn.functional.batch_norm(x_363, getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn3_running_var, getattr_getattr_l__mod___blocks___5_____2___bn3_weight, getattr_getattr_l__mod___blocks___5_____2___bn3_bias, True, 0.1, 0.001);  x_363 = getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = getattr_getattr_l__mod___blocks___5_____2___bn3_weight = getattr_getattr_l__mod___blocks___5_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_365 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_drop(x_364);  x_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_367 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_act(x_365);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____2___drop_path = self.getattr_getattr_L__mod___blocks___5_____2___drop_path(x_367);  x_367 = None
    shortcut_18 = getattr_getattr_l__mod___blocks___5_____2___drop_path + shortcut_17;  getattr_getattr_l__mod___blocks___5_____2___drop_path = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_369 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pw(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____3___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____3___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__54 = getattr_getattr_l__mod___blocks___5_____3___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____3___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_370 = torch.nn.functional.batch_norm(x_369, getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn1_running_var, getattr_getattr_l__mod___blocks___5_____3___bn1_weight, getattr_getattr_l__mod___blocks___5_____3___bn1_bias, True, 0.1, 0.001);  x_369 = getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = getattr_getattr_l__mod___blocks___5_____3___bn1_weight = getattr_getattr_l__mod___blocks___5_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_371 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_drop(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_373 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_act(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_39 = torch.functional.split(x_373, [396, 396, 396, 396], 1);  x_373 = None
    getitem_100 = split_39[0]
    getitem_101 = split_39[1]
    getitem_102 = split_39[2]
    getitem_103 = split_39[3];  split_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____3___conv_dw_0 = self.getattr_getattr_L__mod___blocks___5_____3___conv_dw_0(getitem_100);  getitem_100 = None
    getattr_getattr_l__mod___blocks___5_____3___conv_dw_1 = self.getattr_getattr_L__mod___blocks___5_____3___conv_dw_1(getitem_101);  getitem_101 = None
    getattr_getattr_l__mod___blocks___5_____3___conv_dw_2 = self.getattr_getattr_L__mod___blocks___5_____3___conv_dw_2(getitem_102);  getitem_102 = None
    getattr_getattr_l__mod___blocks___5_____3___conv_dw_3 = self.getattr_getattr_L__mod___blocks___5_____3___conv_dw_3(getitem_103);  getitem_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_375 = torch.cat([getattr_getattr_l__mod___blocks___5_____3___conv_dw_0, getattr_getattr_l__mod___blocks___5_____3___conv_dw_1, getattr_getattr_l__mod___blocks___5_____3___conv_dw_2, getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___5_____3___conv_dw_0 = getattr_getattr_l__mod___blocks___5_____3___conv_dw_1 = getattr_getattr_l__mod___blocks___5_____3___conv_dw_2 = getattr_getattr_l__mod___blocks___5_____3___conv_dw_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____3___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____3___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__55 = getattr_getattr_l__mod___blocks___5_____3___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____3___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_376 = torch.nn.functional.batch_norm(x_375, getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn2_running_var, getattr_getattr_l__mod___blocks___5_____3___bn2_weight, getattr_getattr_l__mod___blocks___5_____3___bn2_bias, True, 0.1, 0.001);  x_375 = getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = getattr_getattr_l__mod___blocks___5_____3___bn2_weight = getattr_getattr_l__mod___blocks___5_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_377 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_drop(x_376);  x_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_379 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_act(x_377);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_60 = x_379.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_61 = self.getattr_getattr_L__mod___blocks___5_____3___se_conv_reduce(x_se_60);  x_se_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_62 = self.getattr_getattr_L__mod___blocks___5_____3___se_act1(x_se_61);  x_se_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_63 = self.getattr_getattr_L__mod___blocks___5_____3___se_conv_expand(x_se_62);  x_se_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____3___se_gate = self.getattr_getattr_L__mod___blocks___5_____3___se_gate(x_se_63);  x_se_63 = None
    x_380 = x_379 * getattr_getattr_l__mod___blocks___5_____3___se_gate;  x_379 = getattr_getattr_l__mod___blocks___5_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_40 = torch.functional.split(x_380, [792, 792], 1);  x_380 = None
    getitem_104 = split_40[0]
    getitem_105 = split_40[1];  split_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pwl_0(getitem_104);  getitem_104 = None
    getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pwl_1(getitem_105);  getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_382 = torch.cat([getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0, getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0 = getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____3___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____3___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__56 = getattr_getattr_l__mod___blocks___5_____3___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____3___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_383 = torch.nn.functional.batch_norm(x_382, getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn3_running_var, getattr_getattr_l__mod___blocks___5_____3___bn3_weight, getattr_getattr_l__mod___blocks___5_____3___bn3_bias, True, 0.1, 0.001);  x_382 = getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = getattr_getattr_l__mod___blocks___5_____3___bn3_weight = getattr_getattr_l__mod___blocks___5_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_384 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_drop(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_386 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_act(x_384);  x_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____3___drop_path = self.getattr_getattr_L__mod___blocks___5_____3___drop_path(x_386);  x_386 = None
    x_388 = getattr_getattr_l__mod___blocks___5_____3___drop_path + shortcut_18;  getattr_getattr_l__mod___blocks___5_____3___drop_path = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    x_389 = self.L__mod___conv_head(x_388);  x_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___bn2_num_batches_tracked = self.L__mod___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__57 = l__mod___bn2_num_batches_tracked.add_(1);  l__mod___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___bn2_running_mean = self.L__mod___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___bn2_running_var = self.L__mod___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___bn2_weight = self.L__mod___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___bn2_bias = self.L__mod___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_390 = torch.nn.functional.batch_norm(x_389, l__mod___bn2_running_mean, l__mod___bn2_running_var, l__mod___bn2_weight, l__mod___bn2_bias, True, 0.1, 0.001);  x_389 = l__mod___bn2_running_mean = l__mod___bn2_running_var = l__mod___bn2_weight = l__mod___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_391 = self.L__mod___bn2_drop(x_390);  x_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_394 = self.L__mod___bn2_act(x_391);  x_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_395 = self.L__mod___global_pool_pool(x_394);  x_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_397 = self.L__mod___global_pool_flatten(x_395);  x_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    pred = self.L__mod___classifier(x_397);  x_397 = None
    return (pred,)
    