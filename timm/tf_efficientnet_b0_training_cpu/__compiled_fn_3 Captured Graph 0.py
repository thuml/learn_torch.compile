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
    x_6 = self.getattr_getattr_L__mod___blocks___0_____0___conv_dw(shortcut);  shortcut = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_10.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_1 = self.getattr_getattr_L__mod___blocks___0_____0___se_conv_reduce(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_2 = self.getattr_getattr_L__mod___blocks___0_____0___se_act1(x_se_1);  x_se_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_3 = self.getattr_getattr_L__mod___blocks___0_____0___se_conv_expand(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___0_____0___se_gate = self.getattr_getattr_L__mod___blocks___0_____0___se_gate(x_se_3);  x_se_3 = None
    x_11 = x_10 * getattr_getattr_l__mod___blocks___0_____0___se_gate;  x_10 = getattr_getattr_l__mod___blocks___0_____0___se_gate = None
    
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
    shortcut_1 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_act(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_17 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pw(shortcut_1);  shortcut_1 = None
    
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
    x_18 = torch.nn.functional.batch_norm(x_17, getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn1_running_var, getattr_getattr_l__mod___blocks___1_____0___bn1_weight, getattr_getattr_l__mod___blocks___1_____0___bn1_bias, True, 0.1, 0.001);  x_17 = getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = getattr_getattr_l__mod___blocks___1_____0___bn1_weight = getattr_getattr_l__mod___blocks___1_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_19 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_drop(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_21 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_act(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_l__mod___blocks___1_____0___conv_dw_weight = self.getattr_getattr_L__mod___blocks___1_____0___conv_dw_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_23 = torch.nn.functional.pad(x_21, (0, 1, 0, 1), value = 0);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_24 = torch.conv2d(x_23, getattr_getattr_l__mod___blocks___1_____0___conv_dw_weight, None, (2, 2), (0, 0), (1, 1), 96);  x_23 = getattr_getattr_l__mod___blocks___1_____0___conv_dw_weight = None
    
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
    x_25 = torch.nn.functional.batch_norm(x_24, getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn2_running_var, getattr_getattr_l__mod___blocks___1_____0___bn2_weight, getattr_getattr_l__mod___blocks___1_____0___bn2_bias, True, 0.1, 0.001);  x_24 = getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = getattr_getattr_l__mod___blocks___1_____0___bn2_weight = getattr_getattr_l__mod___blocks___1_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_26 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_drop(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_28 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_act(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_28.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_5 = self.getattr_getattr_L__mod___blocks___1_____0___se_conv_reduce(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_6 = self.getattr_getattr_L__mod___blocks___1_____0___se_act1(x_se_5);  x_se_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_7 = self.getattr_getattr_L__mod___blocks___1_____0___se_conv_expand(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___1_____0___se_gate = self.getattr_getattr_L__mod___blocks___1_____0___se_gate(x_se_7);  x_se_7 = None
    x_29 = x_28 * getattr_getattr_l__mod___blocks___1_____0___se_gate;  x_28 = getattr_getattr_l__mod___blocks___1_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_30 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pwl(x_29);  x_29 = None
    
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
    x_31 = torch.nn.functional.batch_norm(x_30, getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn3_running_var, getattr_getattr_l__mod___blocks___1_____0___bn3_weight, getattr_getattr_l__mod___blocks___1_____0___bn3_bias, True, 0.1, 0.001);  x_30 = getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = getattr_getattr_l__mod___blocks___1_____0___bn3_weight = getattr_getattr_l__mod___blocks___1_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_32 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_drop(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_act(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_35 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pw(shortcut_2)
    
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
    x_36 = torch.nn.functional.batch_norm(x_35, getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn1_running_var, getattr_getattr_l__mod___blocks___1_____1___bn1_weight, getattr_getattr_l__mod___blocks___1_____1___bn1_bias, True, 0.1, 0.001);  x_35 = getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = getattr_getattr_l__mod___blocks___1_____1___bn1_weight = getattr_getattr_l__mod___blocks___1_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_37 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_drop(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_39 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_act(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_40 = self.getattr_getattr_L__mod___blocks___1_____1___conv_dw(x_39);  x_39 = None
    
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
    x_41 = torch.nn.functional.batch_norm(x_40, getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn2_running_var, getattr_getattr_l__mod___blocks___1_____1___bn2_weight, getattr_getattr_l__mod___blocks___1_____1___bn2_bias, True, 0.1, 0.001);  x_40 = getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = getattr_getattr_l__mod___blocks___1_____1___bn2_weight = getattr_getattr_l__mod___blocks___1_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_42 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_drop(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_44 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_act(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_44.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_9 = self.getattr_getattr_L__mod___blocks___1_____1___se_conv_reduce(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_10 = self.getattr_getattr_L__mod___blocks___1_____1___se_act1(x_se_9);  x_se_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_11 = self.getattr_getattr_L__mod___blocks___1_____1___se_conv_expand(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___1_____1___se_gate = self.getattr_getattr_L__mod___blocks___1_____1___se_gate(x_se_11);  x_se_11 = None
    x_45 = x_44 * getattr_getattr_l__mod___blocks___1_____1___se_gate;  x_44 = getattr_getattr_l__mod___blocks___1_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_46 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pwl(x_45);  x_45 = None
    
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
    x_47 = torch.nn.functional.batch_norm(x_46, getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn3_running_var, getattr_getattr_l__mod___blocks___1_____1___bn3_weight, getattr_getattr_l__mod___blocks___1_____1___bn3_bias, True, 0.1, 0.001);  x_46 = getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = getattr_getattr_l__mod___blocks___1_____1___bn3_weight = getattr_getattr_l__mod___blocks___1_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_48 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_drop(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_50 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_act(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___1_____1___drop_path = self.getattr_getattr_L__mod___blocks___1_____1___drop_path(x_50);  x_50 = None
    shortcut_3 = getattr_getattr_l__mod___blocks___1_____1___drop_path + shortcut_2;  getattr_getattr_l__mod___blocks___1_____1___drop_path = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_52 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pw(shortcut_3);  shortcut_3 = None
    
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
    x_53 = torch.nn.functional.batch_norm(x_52, getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn1_running_var, getattr_getattr_l__mod___blocks___2_____0___bn1_weight, getattr_getattr_l__mod___blocks___2_____0___bn1_bias, True, 0.1, 0.001);  x_52 = getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = getattr_getattr_l__mod___blocks___2_____0___bn1_weight = getattr_getattr_l__mod___blocks___2_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_54 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_drop(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_56 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_act(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_l__mod___blocks___2_____0___conv_dw_weight = self.getattr_getattr_L__mod___blocks___2_____0___conv_dw_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_58 = torch.nn.functional.pad(x_56, (1, 2, 1, 2), value = 0);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_59 = torch.conv2d(x_58, getattr_getattr_l__mod___blocks___2_____0___conv_dw_weight, None, (2, 2), (0, 0), (1, 1), 144);  x_58 = getattr_getattr_l__mod___blocks___2_____0___conv_dw_weight = None
    
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
    x_60 = torch.nn.functional.batch_norm(x_59, getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn2_running_var, getattr_getattr_l__mod___blocks___2_____0___bn2_weight, getattr_getattr_l__mod___blocks___2_____0___bn2_bias, True, 0.1, 0.001);  x_59 = getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = getattr_getattr_l__mod___blocks___2_____0___bn2_weight = getattr_getattr_l__mod___blocks___2_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_61 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_drop(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_63 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_act(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_63.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_13 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_reduce(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_14 = self.getattr_getattr_L__mod___blocks___2_____0___se_act1(x_se_13);  x_se_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_15 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_expand(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____0___se_gate = self.getattr_getattr_L__mod___blocks___2_____0___se_gate(x_se_15);  x_se_15 = None
    x_64 = x_63 * getattr_getattr_l__mod___blocks___2_____0___se_gate;  x_63 = getattr_getattr_l__mod___blocks___2_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_65 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pwl(x_64);  x_64 = None
    
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
    x_66 = torch.nn.functional.batch_norm(x_65, getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn3_running_var, getattr_getattr_l__mod___blocks___2_____0___bn3_weight, getattr_getattr_l__mod___blocks___2_____0___bn3_bias, True, 0.1, 0.001);  x_65 = getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = getattr_getattr_l__mod___blocks___2_____0___bn3_weight = getattr_getattr_l__mod___blocks___2_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_67 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_drop(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_4 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_act(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_70 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pw(shortcut_4)
    
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
    x_71 = torch.nn.functional.batch_norm(x_70, getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn1_running_var, getattr_getattr_l__mod___blocks___2_____1___bn1_weight, getattr_getattr_l__mod___blocks___2_____1___bn1_bias, True, 0.1, 0.001);  x_70 = getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = getattr_getattr_l__mod___blocks___2_____1___bn1_weight = getattr_getattr_l__mod___blocks___2_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_72 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_drop(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_74 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_act(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_75 = self.getattr_getattr_L__mod___blocks___2_____1___conv_dw(x_74);  x_74 = None
    
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
    x_76 = torch.nn.functional.batch_norm(x_75, getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn2_running_var, getattr_getattr_l__mod___blocks___2_____1___bn2_weight, getattr_getattr_l__mod___blocks___2_____1___bn2_bias, True, 0.1, 0.001);  x_75 = getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = getattr_getattr_l__mod___blocks___2_____1___bn2_weight = getattr_getattr_l__mod___blocks___2_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_77 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_drop(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_79 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_act(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_79.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_17 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_reduce(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_18 = self.getattr_getattr_L__mod___blocks___2_____1___se_act1(x_se_17);  x_se_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_19 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_expand(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____1___se_gate = self.getattr_getattr_L__mod___blocks___2_____1___se_gate(x_se_19);  x_se_19 = None
    x_80 = x_79 * getattr_getattr_l__mod___blocks___2_____1___se_gate;  x_79 = getattr_getattr_l__mod___blocks___2_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_81 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pwl(x_80);  x_80 = None
    
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
    x_82 = torch.nn.functional.batch_norm(x_81, getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn3_running_var, getattr_getattr_l__mod___blocks___2_____1___bn3_weight, getattr_getattr_l__mod___blocks___2_____1___bn3_bias, True, 0.1, 0.001);  x_81 = getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = getattr_getattr_l__mod___blocks___2_____1___bn3_weight = getattr_getattr_l__mod___blocks___2_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_83 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_drop(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_85 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_act(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____1___drop_path = self.getattr_getattr_L__mod___blocks___2_____1___drop_path(x_85);  x_85 = None
    shortcut_5 = getattr_getattr_l__mod___blocks___2_____1___drop_path + shortcut_4;  getattr_getattr_l__mod___blocks___2_____1___drop_path = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_87 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pw(shortcut_5);  shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__15 = getattr_getattr_l__mod___blocks___3_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_88 = torch.nn.functional.batch_norm(x_87, getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn1_running_var, getattr_getattr_l__mod___blocks___3_____0___bn1_weight, getattr_getattr_l__mod___blocks___3_____0___bn1_bias, True, 0.1, 0.001);  x_87 = getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = getattr_getattr_l__mod___blocks___3_____0___bn1_weight = getattr_getattr_l__mod___blocks___3_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_89 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_drop(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_91 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_act(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_l__mod___blocks___3_____0___conv_dw_weight = self.getattr_getattr_L__mod___blocks___3_____0___conv_dw_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_93 = torch.nn.functional.pad(x_91, (0, 1, 0, 1), value = 0);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_94 = torch.conv2d(x_93, getattr_getattr_l__mod___blocks___3_____0___conv_dw_weight, None, (2, 2), (0, 0), (1, 1), 240);  x_93 = getattr_getattr_l__mod___blocks___3_____0___conv_dw_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__16 = getattr_getattr_l__mod___blocks___3_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_95 = torch.nn.functional.batch_norm(x_94, getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn2_running_var, getattr_getattr_l__mod___blocks___3_____0___bn2_weight, getattr_getattr_l__mod___blocks___3_____0___bn2_bias, True, 0.1, 0.001);  x_94 = getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = getattr_getattr_l__mod___blocks___3_____0___bn2_weight = getattr_getattr_l__mod___blocks___3_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_96 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_drop(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_98 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_act(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_98.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_21 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_reduce(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_22 = self.getattr_getattr_L__mod___blocks___3_____0___se_act1(x_se_21);  x_se_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_23 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_expand(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____0___se_gate = self.getattr_getattr_L__mod___blocks___3_____0___se_gate(x_se_23);  x_se_23 = None
    x_99 = x_98 * getattr_getattr_l__mod___blocks___3_____0___se_gate;  x_98 = getattr_getattr_l__mod___blocks___3_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_100 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pwl(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__17 = getattr_getattr_l__mod___blocks___3_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_101 = torch.nn.functional.batch_norm(x_100, getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn3_running_var, getattr_getattr_l__mod___blocks___3_____0___bn3_weight, getattr_getattr_l__mod___blocks___3_____0___bn3_bias, True, 0.1, 0.001);  x_100 = getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = getattr_getattr_l__mod___blocks___3_____0___bn3_weight = getattr_getattr_l__mod___blocks___3_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_102 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_drop(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_6 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_act(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_105 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pw(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____1___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____1___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__18 = getattr_getattr_l__mod___blocks___3_____1___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____1___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_106 = torch.nn.functional.batch_norm(x_105, getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn1_running_var, getattr_getattr_l__mod___blocks___3_____1___bn1_weight, getattr_getattr_l__mod___blocks___3_____1___bn1_bias, True, 0.1, 0.001);  x_105 = getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = getattr_getattr_l__mod___blocks___3_____1___bn1_weight = getattr_getattr_l__mod___blocks___3_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_107 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_drop(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_109 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_act(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_110 = self.getattr_getattr_L__mod___blocks___3_____1___conv_dw(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____1___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____1___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__19 = getattr_getattr_l__mod___blocks___3_____1___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____1___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_111 = torch.nn.functional.batch_norm(x_110, getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn2_running_var, getattr_getattr_l__mod___blocks___3_____1___bn2_weight, getattr_getattr_l__mod___blocks___3_____1___bn2_bias, True, 0.1, 0.001);  x_110 = getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = getattr_getattr_l__mod___blocks___3_____1___bn2_weight = getattr_getattr_l__mod___blocks___3_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_112 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_drop(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_114 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_act(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_114.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_25 = self.getattr_getattr_L__mod___blocks___3_____1___se_conv_reduce(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_26 = self.getattr_getattr_L__mod___blocks___3_____1___se_act1(x_se_25);  x_se_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_27 = self.getattr_getattr_L__mod___blocks___3_____1___se_conv_expand(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____1___se_gate = self.getattr_getattr_L__mod___blocks___3_____1___se_gate(x_se_27);  x_se_27 = None
    x_115 = x_114 * getattr_getattr_l__mod___blocks___3_____1___se_gate;  x_114 = getattr_getattr_l__mod___blocks___3_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_116 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pwl(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____1___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____1___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__20 = getattr_getattr_l__mod___blocks___3_____1___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____1___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_117 = torch.nn.functional.batch_norm(x_116, getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn3_running_var, getattr_getattr_l__mod___blocks___3_____1___bn3_weight, getattr_getattr_l__mod___blocks___3_____1___bn3_bias, True, 0.1, 0.001);  x_116 = getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = getattr_getattr_l__mod___blocks___3_____1___bn3_weight = getattr_getattr_l__mod___blocks___3_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_118 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_drop(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_120 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_act(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____1___drop_path = self.getattr_getattr_L__mod___blocks___3_____1___drop_path(x_120);  x_120 = None
    shortcut_7 = getattr_getattr_l__mod___blocks___3_____1___drop_path + shortcut_6;  getattr_getattr_l__mod___blocks___3_____1___drop_path = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_122 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pw(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____2___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____2___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__21 = getattr_getattr_l__mod___blocks___3_____2___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____2___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_123 = torch.nn.functional.batch_norm(x_122, getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn1_running_var, getattr_getattr_l__mod___blocks___3_____2___bn1_weight, getattr_getattr_l__mod___blocks___3_____2___bn1_bias, True, 0.1, 0.001);  x_122 = getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = getattr_getattr_l__mod___blocks___3_____2___bn1_weight = getattr_getattr_l__mod___blocks___3_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_124 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_drop(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_126 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_act(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_127 = self.getattr_getattr_L__mod___blocks___3_____2___conv_dw(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____2___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____2___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__22 = getattr_getattr_l__mod___blocks___3_____2___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____2___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_128 = torch.nn.functional.batch_norm(x_127, getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn2_running_var, getattr_getattr_l__mod___blocks___3_____2___bn2_weight, getattr_getattr_l__mod___blocks___3_____2___bn2_bias, True, 0.1, 0.001);  x_127 = getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = getattr_getattr_l__mod___blocks___3_____2___bn2_weight = getattr_getattr_l__mod___blocks___3_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_129 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_drop(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_131 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_act(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = x_131.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_29 = self.getattr_getattr_L__mod___blocks___3_____2___se_conv_reduce(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_30 = self.getattr_getattr_L__mod___blocks___3_____2___se_act1(x_se_29);  x_se_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_31 = self.getattr_getattr_L__mod___blocks___3_____2___se_conv_expand(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____2___se_gate = self.getattr_getattr_L__mod___blocks___3_____2___se_gate(x_se_31);  x_se_31 = None
    x_132 = x_131 * getattr_getattr_l__mod___blocks___3_____2___se_gate;  x_131 = getattr_getattr_l__mod___blocks___3_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_133 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pwl(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___3_____2___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___3_____2___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__23 = getattr_getattr_l__mod___blocks___3_____2___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___3_____2___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_134 = torch.nn.functional.batch_norm(x_133, getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn3_running_var, getattr_getattr_l__mod___blocks___3_____2___bn3_weight, getattr_getattr_l__mod___blocks___3_____2___bn3_bias, True, 0.1, 0.001);  x_133 = getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = getattr_getattr_l__mod___blocks___3_____2___bn3_weight = getattr_getattr_l__mod___blocks___3_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_135 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_137 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____2___drop_path = self.getattr_getattr_L__mod___blocks___3_____2___drop_path(x_137);  x_137 = None
    shortcut_8 = getattr_getattr_l__mod___blocks___3_____2___drop_path + shortcut_7;  getattr_getattr_l__mod___blocks___3_____2___drop_path = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_139 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pw(shortcut_8);  shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__24 = getattr_getattr_l__mod___blocks___4_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_140 = torch.nn.functional.batch_norm(x_139, getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn1_running_var, getattr_getattr_l__mod___blocks___4_____0___bn1_weight, getattr_getattr_l__mod___blocks___4_____0___bn1_bias, True, 0.1, 0.001);  x_139 = getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = getattr_getattr_l__mod___blocks___4_____0___bn1_weight = getattr_getattr_l__mod___blocks___4_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_141 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_drop(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_143 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_act(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_144 = self.getattr_getattr_L__mod___blocks___4_____0___conv_dw(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__25 = getattr_getattr_l__mod___blocks___4_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_145 = torch.nn.functional.batch_norm(x_144, getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn2_running_var, getattr_getattr_l__mod___blocks___4_____0___bn2_weight, getattr_getattr_l__mod___blocks___4_____0___bn2_bias, True, 0.1, 0.001);  x_144 = getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = getattr_getattr_l__mod___blocks___4_____0___bn2_weight = getattr_getattr_l__mod___blocks___4_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_146 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_drop(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_148 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_act(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_32 = x_148.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_33 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_reduce(x_se_32);  x_se_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_34 = self.getattr_getattr_L__mod___blocks___4_____0___se_act1(x_se_33);  x_se_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_35 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_expand(x_se_34);  x_se_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____0___se_gate = self.getattr_getattr_L__mod___blocks___4_____0___se_gate(x_se_35);  x_se_35 = None
    x_149 = x_148 * getattr_getattr_l__mod___blocks___4_____0___se_gate;  x_148 = getattr_getattr_l__mod___blocks___4_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_150 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pwl(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__26 = getattr_getattr_l__mod___blocks___4_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_151 = torch.nn.functional.batch_norm(x_150, getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn3_running_var, getattr_getattr_l__mod___blocks___4_____0___bn3_weight, getattr_getattr_l__mod___blocks___4_____0___bn3_bias, True, 0.1, 0.001);  x_150 = getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = getattr_getattr_l__mod___blocks___4_____0___bn3_weight = getattr_getattr_l__mod___blocks___4_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_152 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_drop(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_9 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_act(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_155 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pw(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____1___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____1___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__27 = getattr_getattr_l__mod___blocks___4_____1___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____1___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_156 = torch.nn.functional.batch_norm(x_155, getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn1_running_var, getattr_getattr_l__mod___blocks___4_____1___bn1_weight, getattr_getattr_l__mod___blocks___4_____1___bn1_bias, True, 0.1, 0.001);  x_155 = getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = getattr_getattr_l__mod___blocks___4_____1___bn1_weight = getattr_getattr_l__mod___blocks___4_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_157 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_drop(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_159 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_act(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_160 = self.getattr_getattr_L__mod___blocks___4_____1___conv_dw(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____1___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____1___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__28 = getattr_getattr_l__mod___blocks___4_____1___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____1___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_161 = torch.nn.functional.batch_norm(x_160, getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn2_running_var, getattr_getattr_l__mod___blocks___4_____1___bn2_weight, getattr_getattr_l__mod___blocks___4_____1___bn2_bias, True, 0.1, 0.001);  x_160 = getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = getattr_getattr_l__mod___blocks___4_____1___bn2_weight = getattr_getattr_l__mod___blocks___4_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_162 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_drop(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_164 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_act(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_36 = x_164.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_37 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_reduce(x_se_36);  x_se_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_38 = self.getattr_getattr_L__mod___blocks___4_____1___se_act1(x_se_37);  x_se_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_39 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_expand(x_se_38);  x_se_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____1___se_gate = self.getattr_getattr_L__mod___blocks___4_____1___se_gate(x_se_39);  x_se_39 = None
    x_165 = x_164 * getattr_getattr_l__mod___blocks___4_____1___se_gate;  x_164 = getattr_getattr_l__mod___blocks___4_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_166 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pwl(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____1___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____1___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__29 = getattr_getattr_l__mod___blocks___4_____1___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____1___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_167 = torch.nn.functional.batch_norm(x_166, getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn3_running_var, getattr_getattr_l__mod___blocks___4_____1___bn3_weight, getattr_getattr_l__mod___blocks___4_____1___bn3_bias, True, 0.1, 0.001);  x_166 = getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = getattr_getattr_l__mod___blocks___4_____1___bn3_weight = getattr_getattr_l__mod___blocks___4_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_168 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_drop(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_170 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_act(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____1___drop_path = self.getattr_getattr_L__mod___blocks___4_____1___drop_path(x_170);  x_170 = None
    shortcut_10 = getattr_getattr_l__mod___blocks___4_____1___drop_path + shortcut_9;  getattr_getattr_l__mod___blocks___4_____1___drop_path = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_172 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pw(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____2___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____2___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__30 = getattr_getattr_l__mod___blocks___4_____2___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____2___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_173 = torch.nn.functional.batch_norm(x_172, getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn1_running_var, getattr_getattr_l__mod___blocks___4_____2___bn1_weight, getattr_getattr_l__mod___blocks___4_____2___bn1_bias, True, 0.1, 0.001);  x_172 = getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = getattr_getattr_l__mod___blocks___4_____2___bn1_weight = getattr_getattr_l__mod___blocks___4_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_174 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_drop(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_176 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_act(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_177 = self.getattr_getattr_L__mod___blocks___4_____2___conv_dw(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____2___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____2___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__31 = getattr_getattr_l__mod___blocks___4_____2___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____2___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_178 = torch.nn.functional.batch_norm(x_177, getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn2_running_var, getattr_getattr_l__mod___blocks___4_____2___bn2_weight, getattr_getattr_l__mod___blocks___4_____2___bn2_bias, True, 0.1, 0.001);  x_177 = getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = getattr_getattr_l__mod___blocks___4_____2___bn2_weight = getattr_getattr_l__mod___blocks___4_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_179 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_drop(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_181 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_act(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_40 = x_181.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_41 = self.getattr_getattr_L__mod___blocks___4_____2___se_conv_reduce(x_se_40);  x_se_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_42 = self.getattr_getattr_L__mod___blocks___4_____2___se_act1(x_se_41);  x_se_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_43 = self.getattr_getattr_L__mod___blocks___4_____2___se_conv_expand(x_se_42);  x_se_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____2___se_gate = self.getattr_getattr_L__mod___blocks___4_____2___se_gate(x_se_43);  x_se_43 = None
    x_182 = x_181 * getattr_getattr_l__mod___blocks___4_____2___se_gate;  x_181 = getattr_getattr_l__mod___blocks___4_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_183 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pwl(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___4_____2___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___4_____2___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__32 = getattr_getattr_l__mod___blocks___4_____2___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___4_____2___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_184 = torch.nn.functional.batch_norm(x_183, getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn3_running_var, getattr_getattr_l__mod___blocks___4_____2___bn3_weight, getattr_getattr_l__mod___blocks___4_____2___bn3_bias, True, 0.1, 0.001);  x_183 = getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = getattr_getattr_l__mod___blocks___4_____2___bn3_weight = getattr_getattr_l__mod___blocks___4_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_185 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_drop(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_187 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_act(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____2___drop_path = self.getattr_getattr_L__mod___blocks___4_____2___drop_path(x_187);  x_187 = None
    shortcut_11 = getattr_getattr_l__mod___blocks___4_____2___drop_path + shortcut_10;  getattr_getattr_l__mod___blocks___4_____2___drop_path = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_189 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pw(shortcut_11);  shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__33 = getattr_getattr_l__mod___blocks___5_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_190 = torch.nn.functional.batch_norm(x_189, getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn1_running_var, getattr_getattr_l__mod___blocks___5_____0___bn1_weight, getattr_getattr_l__mod___blocks___5_____0___bn1_bias, True, 0.1, 0.001);  x_189 = getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = getattr_getattr_l__mod___blocks___5_____0___bn1_weight = getattr_getattr_l__mod___blocks___5_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_191 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_drop(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_193 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_act(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    getattr_getattr_l__mod___blocks___5_____0___conv_dw_weight = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_195 = torch.nn.functional.pad(x_193, (1, 2, 1, 2), value = 0);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_196 = torch.conv2d(x_195, getattr_getattr_l__mod___blocks___5_____0___conv_dw_weight, None, (2, 2), (0, 0), (1, 1), 672);  x_195 = getattr_getattr_l__mod___blocks___5_____0___conv_dw_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__34 = getattr_getattr_l__mod___blocks___5_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_197 = torch.nn.functional.batch_norm(x_196, getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn2_running_var, getattr_getattr_l__mod___blocks___5_____0___bn2_weight, getattr_getattr_l__mod___blocks___5_____0___bn2_bias, True, 0.1, 0.001);  x_196 = getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = getattr_getattr_l__mod___blocks___5_____0___bn2_weight = getattr_getattr_l__mod___blocks___5_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_198 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_drop(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_200 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_act(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_44 = x_200.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_45 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_reduce(x_se_44);  x_se_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_46 = self.getattr_getattr_L__mod___blocks___5_____0___se_act1(x_se_45);  x_se_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_47 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_expand(x_se_46);  x_se_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____0___se_gate = self.getattr_getattr_L__mod___blocks___5_____0___se_gate(x_se_47);  x_se_47 = None
    x_201 = x_200 * getattr_getattr_l__mod___blocks___5_____0___se_gate;  x_200 = getattr_getattr_l__mod___blocks___5_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_202 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pwl(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__35 = getattr_getattr_l__mod___blocks___5_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_203 = torch.nn.functional.batch_norm(x_202, getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn3_running_var, getattr_getattr_l__mod___blocks___5_____0___bn3_weight, getattr_getattr_l__mod___blocks___5_____0___bn3_bias, True, 0.1, 0.001);  x_202 = getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = getattr_getattr_l__mod___blocks___5_____0___bn3_weight = getattr_getattr_l__mod___blocks___5_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_204 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_drop(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_12 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_act(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_207 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pw(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____1___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____1___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__36 = getattr_getattr_l__mod___blocks___5_____1___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____1___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_208 = torch.nn.functional.batch_norm(x_207, getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn1_running_var, getattr_getattr_l__mod___blocks___5_____1___bn1_weight, getattr_getattr_l__mod___blocks___5_____1___bn1_bias, True, 0.1, 0.001);  x_207 = getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = getattr_getattr_l__mod___blocks___5_____1___bn1_weight = getattr_getattr_l__mod___blocks___5_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_209 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_drop(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_211 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_act(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_212 = self.getattr_getattr_L__mod___blocks___5_____1___conv_dw(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____1___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____1___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__37 = getattr_getattr_l__mod___blocks___5_____1___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____1___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_213 = torch.nn.functional.batch_norm(x_212, getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn2_running_var, getattr_getattr_l__mod___blocks___5_____1___bn2_weight, getattr_getattr_l__mod___blocks___5_____1___bn2_bias, True, 0.1, 0.001);  x_212 = getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = getattr_getattr_l__mod___blocks___5_____1___bn2_weight = getattr_getattr_l__mod___blocks___5_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_214 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_drop(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_216 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_act(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_48 = x_216.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_49 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_reduce(x_se_48);  x_se_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_50 = self.getattr_getattr_L__mod___blocks___5_____1___se_act1(x_se_49);  x_se_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_51 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_expand(x_se_50);  x_se_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____1___se_gate = self.getattr_getattr_L__mod___blocks___5_____1___se_gate(x_se_51);  x_se_51 = None
    x_217 = x_216 * getattr_getattr_l__mod___blocks___5_____1___se_gate;  x_216 = getattr_getattr_l__mod___blocks___5_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_218 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pwl(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____1___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____1___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__38 = getattr_getattr_l__mod___blocks___5_____1___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____1___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_219 = torch.nn.functional.batch_norm(x_218, getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn3_running_var, getattr_getattr_l__mod___blocks___5_____1___bn3_weight, getattr_getattr_l__mod___blocks___5_____1___bn3_bias, True, 0.1, 0.001);  x_218 = getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = getattr_getattr_l__mod___blocks___5_____1___bn3_weight = getattr_getattr_l__mod___blocks___5_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_220 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_drop(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_222 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_act(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____1___drop_path = self.getattr_getattr_L__mod___blocks___5_____1___drop_path(x_222);  x_222 = None
    shortcut_13 = getattr_getattr_l__mod___blocks___5_____1___drop_path + shortcut_12;  getattr_getattr_l__mod___blocks___5_____1___drop_path = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_224 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pw(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____2___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____2___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__39 = getattr_getattr_l__mod___blocks___5_____2___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____2___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_225 = torch.nn.functional.batch_norm(x_224, getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn1_running_var, getattr_getattr_l__mod___blocks___5_____2___bn1_weight, getattr_getattr_l__mod___blocks___5_____2___bn1_bias, True, 0.1, 0.001);  x_224 = getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = getattr_getattr_l__mod___blocks___5_____2___bn1_weight = getattr_getattr_l__mod___blocks___5_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_226 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_drop(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_228 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_act(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_229 = self.getattr_getattr_L__mod___blocks___5_____2___conv_dw(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____2___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____2___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__40 = getattr_getattr_l__mod___blocks___5_____2___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____2___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_230 = torch.nn.functional.batch_norm(x_229, getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn2_running_var, getattr_getattr_l__mod___blocks___5_____2___bn2_weight, getattr_getattr_l__mod___blocks___5_____2___bn2_bias, True, 0.1, 0.001);  x_229 = getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = getattr_getattr_l__mod___blocks___5_____2___bn2_weight = getattr_getattr_l__mod___blocks___5_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_231 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_drop(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_233 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_act(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_52 = x_233.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_53 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_reduce(x_se_52);  x_se_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_54 = self.getattr_getattr_L__mod___blocks___5_____2___se_act1(x_se_53);  x_se_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_55 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_expand(x_se_54);  x_se_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____2___se_gate = self.getattr_getattr_L__mod___blocks___5_____2___se_gate(x_se_55);  x_se_55 = None
    x_234 = x_233 * getattr_getattr_l__mod___blocks___5_____2___se_gate;  x_233 = getattr_getattr_l__mod___blocks___5_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_235 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pwl(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____2___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____2___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__41 = getattr_getattr_l__mod___blocks___5_____2___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____2___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_236 = torch.nn.functional.batch_norm(x_235, getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn3_running_var, getattr_getattr_l__mod___blocks___5_____2___bn3_weight, getattr_getattr_l__mod___blocks___5_____2___bn3_bias, True, 0.1, 0.001);  x_235 = getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = getattr_getattr_l__mod___blocks___5_____2___bn3_weight = getattr_getattr_l__mod___blocks___5_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_237 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_drop(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_239 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_act(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____2___drop_path = self.getattr_getattr_L__mod___blocks___5_____2___drop_path(x_239);  x_239 = None
    shortcut_14 = getattr_getattr_l__mod___blocks___5_____2___drop_path + shortcut_13;  getattr_getattr_l__mod___blocks___5_____2___drop_path = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_241 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pw(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____3___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____3___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__42 = getattr_getattr_l__mod___blocks___5_____3___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____3___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_242 = torch.nn.functional.batch_norm(x_241, getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn1_running_var, getattr_getattr_l__mod___blocks___5_____3___bn1_weight, getattr_getattr_l__mod___blocks___5_____3___bn1_bias, True, 0.1, 0.001);  x_241 = getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = getattr_getattr_l__mod___blocks___5_____3___bn1_weight = getattr_getattr_l__mod___blocks___5_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_243 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_drop(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_245 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_act(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_246 = self.getattr_getattr_L__mod___blocks___5_____3___conv_dw(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____3___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____3___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__43 = getattr_getattr_l__mod___blocks___5_____3___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____3___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_247 = torch.nn.functional.batch_norm(x_246, getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn2_running_var, getattr_getattr_l__mod___blocks___5_____3___bn2_weight, getattr_getattr_l__mod___blocks___5_____3___bn2_bias, True, 0.1, 0.001);  x_246 = getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = getattr_getattr_l__mod___blocks___5_____3___bn2_weight = getattr_getattr_l__mod___blocks___5_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_248 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_drop(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_250 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_act(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_56 = x_250.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_57 = self.getattr_getattr_L__mod___blocks___5_____3___se_conv_reduce(x_se_56);  x_se_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_58 = self.getattr_getattr_L__mod___blocks___5_____3___se_act1(x_se_57);  x_se_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_59 = self.getattr_getattr_L__mod___blocks___5_____3___se_conv_expand(x_se_58);  x_se_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____3___se_gate = self.getattr_getattr_L__mod___blocks___5_____3___se_gate(x_se_59);  x_se_59 = None
    x_251 = x_250 * getattr_getattr_l__mod___blocks___5_____3___se_gate;  x_250 = getattr_getattr_l__mod___blocks___5_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_252 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pwl(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___5_____3___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___5_____3___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__44 = getattr_getattr_l__mod___blocks___5_____3___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___5_____3___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_253 = torch.nn.functional.batch_norm(x_252, getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn3_running_var, getattr_getattr_l__mod___blocks___5_____3___bn3_weight, getattr_getattr_l__mod___blocks___5_____3___bn3_bias, True, 0.1, 0.001);  x_252 = getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = getattr_getattr_l__mod___blocks___5_____3___bn3_weight = getattr_getattr_l__mod___blocks___5_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_254 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_drop(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_256 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_act(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____3___drop_path = self.getattr_getattr_L__mod___blocks___5_____3___drop_path(x_256);  x_256 = None
    shortcut_15 = getattr_getattr_l__mod___blocks___5_____3___drop_path + shortcut_14;  getattr_getattr_l__mod___blocks___5_____3___drop_path = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_258 = self.getattr_getattr_L__mod___blocks___6_____0___conv_pw(shortcut_15);  shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___6_____0___bn1_num_batches_tracked = self.getattr_getattr_L__mod___blocks___6_____0___bn1_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__45 = getattr_getattr_l__mod___blocks___6_____0___bn1_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___6_____0___bn1_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_259 = torch.nn.functional.batch_norm(x_258, getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn1_running_var, getattr_getattr_l__mod___blocks___6_____0___bn1_weight, getattr_getattr_l__mod___blocks___6_____0___bn1_bias, True, 0.1, 0.001);  x_258 = getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn1_running_var = getattr_getattr_l__mod___blocks___6_____0___bn1_weight = getattr_getattr_l__mod___blocks___6_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_260 = self.getattr_getattr_L__mod___blocks___6_____0___bn1_drop(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_262 = self.getattr_getattr_L__mod___blocks___6_____0___bn1_act(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_263 = self.getattr_getattr_L__mod___blocks___6_____0___conv_dw(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___6_____0___bn2_num_batches_tracked = self.getattr_getattr_L__mod___blocks___6_____0___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__46 = getattr_getattr_l__mod___blocks___6_____0___bn2_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___6_____0___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_264 = torch.nn.functional.batch_norm(x_263, getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn2_running_var, getattr_getattr_l__mod___blocks___6_____0___bn2_weight, getattr_getattr_l__mod___blocks___6_____0___bn2_bias, True, 0.1, 0.001);  x_263 = getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn2_running_var = getattr_getattr_l__mod___blocks___6_____0___bn2_weight = getattr_getattr_l__mod___blocks___6_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_265 = self.getattr_getattr_L__mod___blocks___6_____0___bn2_drop(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_267 = self.getattr_getattr_L__mod___blocks___6_____0___bn2_act(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_60 = x_267.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_61 = self.getattr_getattr_L__mod___blocks___6_____0___se_conv_reduce(x_se_60);  x_se_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_62 = self.getattr_getattr_L__mod___blocks___6_____0___se_act1(x_se_61);  x_se_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_63 = self.getattr_getattr_L__mod___blocks___6_____0___se_conv_expand(x_se_62);  x_se_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___6_____0___se_gate = self.getattr_getattr_L__mod___blocks___6_____0___se_gate(x_se_63);  x_se_63 = None
    x_268 = x_267 * getattr_getattr_l__mod___blocks___6_____0___se_gate;  x_267 = getattr_getattr_l__mod___blocks___6_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_269 = self.getattr_getattr_L__mod___blocks___6_____0___conv_pwl(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___blocks___6_____0___bn3_num_batches_tracked = self.getattr_getattr_L__mod___blocks___6_____0___bn3_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__47 = getattr_getattr_l__mod___blocks___6_____0___bn3_num_batches_tracked.add_(1);  getattr_getattr_l__mod___blocks___6_____0___bn3_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_270 = torch.nn.functional.batch_norm(x_269, getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn3_running_var, getattr_getattr_l__mod___blocks___6_____0___bn3_weight, getattr_getattr_l__mod___blocks___6_____0___bn3_bias, True, 0.1, 0.001);  x_269 = getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn3_running_var = getattr_getattr_l__mod___blocks___6_____0___bn3_weight = getattr_getattr_l__mod___blocks___6_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_271 = self.getattr_getattr_L__mod___blocks___6_____0___bn3_drop(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_274 = self.getattr_getattr_L__mod___blocks___6_____0___bn3_act(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    x_275 = self.L__mod___conv_head(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___bn2_num_batches_tracked = self.L__mod___bn2_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__48 = l__mod___bn2_num_batches_tracked.add_(1);  l__mod___bn2_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___bn2_running_mean = self.L__mod___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___bn2_running_var = self.L__mod___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___bn2_weight = self.L__mod___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___bn2_bias = self.L__mod___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_276 = torch.nn.functional.batch_norm(x_275, l__mod___bn2_running_mean, l__mod___bn2_running_var, l__mod___bn2_weight, l__mod___bn2_bias, True, 0.1, 0.001);  x_275 = l__mod___bn2_running_mean = l__mod___bn2_running_var = l__mod___bn2_weight = l__mod___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_277 = self.L__mod___bn2_drop(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_280 = self.L__mod___bn2_act(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_281 = self.L__mod___global_pool_pool(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_283 = self.L__mod___global_pool_flatten(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    pred = self.L__mod___classifier(x_283);  x_283 = None
    return (pred,)
    