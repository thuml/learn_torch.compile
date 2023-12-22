from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    x = self.L__mod___conv_stem(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
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
    x_1 = torch.nn.functional.batch_norm(x, l__mod___bn1_running_mean, l__mod___bn1_running_var, l__mod___bn1_weight, l__mod___bn1_bias, True, 0.1, 1e-05);  x = l__mod___bn1_running_mean = l__mod___bn1_running_var = l__mod___bn1_weight = l__mod___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___bn1_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___bn1_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    x_5 = self.getattr_getattr_L__mod___blocks___0_____0___conv_dw(shortcut)
    
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
    x_6 = torch.nn.functional.batch_norm(x_5, getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn1_running_var, getattr_getattr_l__mod___blocks___0_____0___bn1_weight, getattr_getattr_l__mod___blocks___0_____0___bn1_bias, True, 0.1, 1e-05);  x_5 = getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn1_running_var = getattr_getattr_l__mod___blocks___0_____0___bn1_weight = getattr_getattr_l__mod___blocks___0_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_7 = self.getattr_getattr_L__mod___blocks___0_____0___bn1_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_9 = self.getattr_getattr_L__mod___blocks___0_____0___bn1_act(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:125, code: x = self.se(x)
    x_10 = self.getattr_getattr_L__mod___blocks___0_____0___se(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    x_11 = self.getattr_getattr_L__mod___blocks___0_____0___conv_pw(x_10);  x_10 = None
    
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
    x_12 = torch.nn.functional.batch_norm(x_11, getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn2_running_var, getattr_getattr_l__mod___blocks___0_____0___bn2_weight, getattr_getattr_l__mod___blocks___0_____0___bn2_bias, True, 0.1, 1e-05);  x_11 = getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn2_running_var = getattr_getattr_l__mod___blocks___0_____0___bn2_weight = getattr_getattr_l__mod___blocks___0_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_13 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_drop(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_15 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_act(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___0_____0___drop_path = self.getattr_getattr_L__mod___blocks___0_____0___drop_path(x_15);  x_15 = None
    shortcut_1 = getattr_getattr_l__mod___blocks___0_____0___drop_path + shortcut;  getattr_getattr_l__mod___blocks___0_____0___drop_path = shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split = torch.functional.split(shortcut_1, [16, 16], 1);  shortcut_1 = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____0___conv_pw_0 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pw_0(getitem);  getitem = None
    getattr_getattr_l__mod___blocks___1_____0___conv_pw_1 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pw_1(getitem_1);  getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_18 = torch.cat([getattr_getattr_l__mod___blocks___1_____0___conv_pw_0, getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___1_____0___conv_pw_0 = getattr_getattr_l__mod___blocks___1_____0___conv_pw_1 = None
    
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
    x_19 = torch.nn.functional.batch_norm(x_18, getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn1_running_var, getattr_getattr_l__mod___blocks___1_____0___bn1_weight, getattr_getattr_l__mod___blocks___1_____0___bn1_bias, True, 0.1, 1e-05);  x_18 = getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = getattr_getattr_l__mod___blocks___1_____0___bn1_weight = getattr_getattr_l__mod___blocks___1_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_20 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_drop(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_22 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_act(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_1 = torch.functional.split(x_22, [64, 64, 64], 1);  x_22 = None
    getitem_2 = split_1[0]
    getitem_3 = split_1[1]
    getitem_4 = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____0___conv_dw_0 = self.getattr_getattr_L__mod___blocks___1_____0___conv_dw_0(getitem_2);  getitem_2 = None
    getattr_getattr_l__mod___blocks___1_____0___conv_dw_1 = self.getattr_getattr_L__mod___blocks___1_____0___conv_dw_1(getitem_3);  getitem_3 = None
    getattr_getattr_l__mod___blocks___1_____0___conv_dw_2 = self.getattr_getattr_L__mod___blocks___1_____0___conv_dw_2(getitem_4);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_24 = torch.cat([getattr_getattr_l__mod___blocks___1_____0___conv_dw_0, getattr_getattr_l__mod___blocks___1_____0___conv_dw_1, getattr_getattr_l__mod___blocks___1_____0___conv_dw_2], 1);  getattr_getattr_l__mod___blocks___1_____0___conv_dw_0 = getattr_getattr_l__mod___blocks___1_____0___conv_dw_1 = getattr_getattr_l__mod___blocks___1_____0___conv_dw_2 = None
    
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
    x_25 = torch.nn.functional.batch_norm(x_24, getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn2_running_var, getattr_getattr_l__mod___blocks___1_____0___bn2_weight, getattr_getattr_l__mod___blocks___1_____0___bn2_bias, True, 0.1, 1e-05);  x_24 = getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = getattr_getattr_l__mod___blocks___1_____0___bn2_weight = getattr_getattr_l__mod___blocks___1_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_26 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_drop(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_28 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_act(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_29 = self.getattr_getattr_L__mod___blocks___1_____0___se(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_2 = torch.functional.split(x_29, [96, 96], 1);  x_29 = None
    getitem_5 = split_2[0]
    getitem_6 = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pwl_0(getitem_5);  getitem_5 = None
    getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pwl_1(getitem_6);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_31 = torch.cat([getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0, getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0 = getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1 = None
    
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
    x_32 = torch.nn.functional.batch_norm(x_31, getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn3_running_var, getattr_getattr_l__mod___blocks___1_____0___bn3_weight, getattr_getattr_l__mod___blocks___1_____0___bn3_bias, True, 0.1, 1e-05);  x_31 = getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = getattr_getattr_l__mod___blocks___1_____0___bn3_weight = getattr_getattr_l__mod___blocks___1_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_33 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_drop(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_act(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_3 = torch.functional.split(shortcut_2, [20, 20], 1)
    getitem_7 = split_3[0]
    getitem_8 = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____1___conv_pw_0 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pw_0(getitem_7);  getitem_7 = None
    getattr_getattr_l__mod___blocks___1_____1___conv_pw_1 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pw_1(getitem_8);  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_37 = torch.cat([getattr_getattr_l__mod___blocks___1_____1___conv_pw_0, getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___1_____1___conv_pw_0 = getattr_getattr_l__mod___blocks___1_____1___conv_pw_1 = None
    
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
    x_38 = torch.nn.functional.batch_norm(x_37, getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn1_running_var, getattr_getattr_l__mod___blocks___1_____1___bn1_weight, getattr_getattr_l__mod___blocks___1_____1___bn1_bias, True, 0.1, 1e-05);  x_37 = getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = getattr_getattr_l__mod___blocks___1_____1___bn1_weight = getattr_getattr_l__mod___blocks___1_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_39 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_drop(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_41 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_act(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_42 = self.getattr_getattr_L__mod___blocks___1_____1___conv_dw(x_41);  x_41 = None
    
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
    x_43 = torch.nn.functional.batch_norm(x_42, getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn2_running_var, getattr_getattr_l__mod___blocks___1_____1___bn2_weight, getattr_getattr_l__mod___blocks___1_____1___bn2_bias, True, 0.1, 1e-05);  x_42 = getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = getattr_getattr_l__mod___blocks___1_____1___bn2_weight = getattr_getattr_l__mod___blocks___1_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_44 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_drop(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_46 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_act(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:184, code: x = self.se(x)
    x_47 = self.getattr_getattr_L__mod___blocks___1_____1___se(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_4 = torch.functional.split(x_47, [60, 60], 1);  x_47 = None
    getitem_9 = split_4[0]
    getitem_10 = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pwl_0(getitem_9);  getitem_9 = None
    getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pwl_1(getitem_10);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_49 = torch.cat([getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1 = None
    
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
    x_50 = torch.nn.functional.batch_norm(x_49, getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn3_running_var, getattr_getattr_l__mod___blocks___1_____1___bn3_weight, getattr_getattr_l__mod___blocks___1_____1___bn3_bias, True, 0.1, 1e-05);  x_49 = getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = getattr_getattr_l__mod___blocks___1_____1___bn3_weight = getattr_getattr_l__mod___blocks___1_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_51 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_drop(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_53 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_act(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___1_____1___drop_path = self.getattr_getattr_L__mod___blocks___1_____1___drop_path(x_53);  x_53 = None
    shortcut_3 = getattr_getattr_l__mod___blocks___1_____1___drop_path + shortcut_2;  getattr_getattr_l__mod___blocks___1_____1___drop_path = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_55 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pw(shortcut_3);  shortcut_3 = None
    
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
    x_56 = torch.nn.functional.batch_norm(x_55, getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn1_running_var, getattr_getattr_l__mod___blocks___2_____0___bn1_weight, getattr_getattr_l__mod___blocks___2_____0___bn1_bias, True, 0.1, 1e-05);  x_55 = getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = getattr_getattr_l__mod___blocks___2_____0___bn1_weight = getattr_getattr_l__mod___blocks___2_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_57 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_drop(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_59 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_act(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_5 = torch.functional.split(x_59, [60, 60, 60, 60], 1);  x_59 = None
    getitem_11 = split_5[0]
    getitem_12 = split_5[1]
    getitem_13 = split_5[2]
    getitem_14 = split_5[3];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____0___conv_dw_0 = self.getattr_getattr_L__mod___blocks___2_____0___conv_dw_0(getitem_11);  getitem_11 = None
    getattr_getattr_l__mod___blocks___2_____0___conv_dw_1 = self.getattr_getattr_L__mod___blocks___2_____0___conv_dw_1(getitem_12);  getitem_12 = None
    getattr_getattr_l__mod___blocks___2_____0___conv_dw_2 = self.getattr_getattr_L__mod___blocks___2_____0___conv_dw_2(getitem_13);  getitem_13 = None
    getattr_getattr_l__mod___blocks___2_____0___conv_dw_3 = self.getattr_getattr_L__mod___blocks___2_____0___conv_dw_3(getitem_14);  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_61 = torch.cat([getattr_getattr_l__mod___blocks___2_____0___conv_dw_0, getattr_getattr_l__mod___blocks___2_____0___conv_dw_1, getattr_getattr_l__mod___blocks___2_____0___conv_dw_2, getattr_getattr_l__mod___blocks___2_____0___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___2_____0___conv_dw_0 = getattr_getattr_l__mod___blocks___2_____0___conv_dw_1 = getattr_getattr_l__mod___blocks___2_____0___conv_dw_2 = getattr_getattr_l__mod___blocks___2_____0___conv_dw_3 = None
    
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
    x_62 = torch.nn.functional.batch_norm(x_61, getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn2_running_var, getattr_getattr_l__mod___blocks___2_____0___bn2_weight, getattr_getattr_l__mod___blocks___2_____0___bn2_bias, True, 0.1, 1e-05);  x_61 = getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = getattr_getattr_l__mod___blocks___2_____0___bn2_weight = getattr_getattr_l__mod___blocks___2_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_63 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_drop(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_65 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_act(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_65.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_1 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_reduce(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_2 = self.getattr_getattr_L__mod___blocks___2_____0___se_act1(x_se_1);  x_se_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_3 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_expand(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____0___se_gate = self.getattr_getattr_L__mod___blocks___2_____0___se_gate(x_se_3);  x_se_3 = None
    x_66 = x_65 * getattr_getattr_l__mod___blocks___2_____0___se_gate;  x_65 = getattr_getattr_l__mod___blocks___2_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_67 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pwl(x_66);  x_66 = None
    
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
    x_68 = torch.nn.functional.batch_norm(x_67, getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn3_running_var, getattr_getattr_l__mod___blocks___2_____0___bn3_weight, getattr_getattr_l__mod___blocks___2_____0___bn3_bias, True, 0.1, 1e-05);  x_67 = getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = getattr_getattr_l__mod___blocks___2_____0___bn3_weight = getattr_getattr_l__mod___blocks___2_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_69 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_drop(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_4 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_act(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_6 = torch.functional.split(shortcut_4, [28, 28], 1)
    getitem_15 = split_6[0]
    getitem_16 = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____1___conv_pw_0 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pw_0(getitem_15);  getitem_15 = None
    getattr_getattr_l__mod___blocks___2_____1___conv_pw_1 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pw_1(getitem_16);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_73 = torch.cat([getattr_getattr_l__mod___blocks___2_____1___conv_pw_0, getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___2_____1___conv_pw_0 = getattr_getattr_l__mod___blocks___2_____1___conv_pw_1 = None
    
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
    x_74 = torch.nn.functional.batch_norm(x_73, getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn1_running_var, getattr_getattr_l__mod___blocks___2_____1___bn1_weight, getattr_getattr_l__mod___blocks___2_____1___bn1_bias, True, 0.1, 1e-05);  x_73 = getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = getattr_getattr_l__mod___blocks___2_____1___bn1_weight = getattr_getattr_l__mod___blocks___2_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_75 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_drop(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_77 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_act(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_7 = torch.functional.split(x_77, [168, 168], 1);  x_77 = None
    getitem_17 = split_7[0]
    getitem_18 = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____1___conv_dw_0 = self.getattr_getattr_L__mod___blocks___2_____1___conv_dw_0(getitem_17);  getitem_17 = None
    getattr_getattr_l__mod___blocks___2_____1___conv_dw_1 = self.getattr_getattr_L__mod___blocks___2_____1___conv_dw_1(getitem_18);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_79 = torch.cat([getattr_getattr_l__mod___blocks___2_____1___conv_dw_0, getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], 1);  getattr_getattr_l__mod___blocks___2_____1___conv_dw_0 = getattr_getattr_l__mod___blocks___2_____1___conv_dw_1 = None
    
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
    x_80 = torch.nn.functional.batch_norm(x_79, getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn2_running_var, getattr_getattr_l__mod___blocks___2_____1___bn2_weight, getattr_getattr_l__mod___blocks___2_____1___bn2_bias, True, 0.1, 1e-05);  x_79 = getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = getattr_getattr_l__mod___blocks___2_____1___bn2_weight = getattr_getattr_l__mod___blocks___2_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_81 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_drop(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_83 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_act(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_83.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_5 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_reduce(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_6 = self.getattr_getattr_L__mod___blocks___2_____1___se_act1(x_se_5);  x_se_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_7 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_expand(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____1___se_gate = self.getattr_getattr_L__mod___blocks___2_____1___se_gate(x_se_7);  x_se_7 = None
    x_84 = x_83 * getattr_getattr_l__mod___blocks___2_____1___se_gate;  x_83 = getattr_getattr_l__mod___blocks___2_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_8 = torch.functional.split(x_84, [168, 168], 1);  x_84 = None
    getitem_19 = split_8[0]
    getitem_20 = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pwl_0(getitem_19);  getitem_19 = None
    getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pwl_1(getitem_20);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_86 = torch.cat([getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1 = None
    
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
    x_87 = torch.nn.functional.batch_norm(x_86, getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn3_running_var, getattr_getattr_l__mod___blocks___2_____1___bn3_weight, getattr_getattr_l__mod___blocks___2_____1___bn3_bias, True, 0.1, 1e-05);  x_86 = getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = getattr_getattr_l__mod___blocks___2_____1___bn3_weight = getattr_getattr_l__mod___blocks___2_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_88 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_drop(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_90 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_act(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____1___drop_path = self.getattr_getattr_L__mod___blocks___2_____1___drop_path(x_90);  x_90 = None
    shortcut_5 = getattr_getattr_l__mod___blocks___2_____1___drop_path + shortcut_4;  getattr_getattr_l__mod___blocks___2_____1___drop_path = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_9 = torch.functional.split(shortcut_5, [28, 28], 1)
    getitem_21 = split_9[0]
    getitem_22 = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____2___conv_pw_0 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pw_0(getitem_21);  getitem_21 = None
    getattr_getattr_l__mod___blocks___2_____2___conv_pw_1 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pw_1(getitem_22);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_93 = torch.cat([getattr_getattr_l__mod___blocks___2_____2___conv_pw_0, getattr_getattr_l__mod___blocks___2_____2___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___2_____2___conv_pw_0 = getattr_getattr_l__mod___blocks___2_____2___conv_pw_1 = None
    
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
    x_94 = torch.nn.functional.batch_norm(x_93, getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn1_running_var, getattr_getattr_l__mod___blocks___2_____2___bn1_weight, getattr_getattr_l__mod___blocks___2_____2___bn1_bias, True, 0.1, 1e-05);  x_93 = getattr_getattr_l__mod___blocks___2_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn1_running_var = getattr_getattr_l__mod___blocks___2_____2___bn1_weight = getattr_getattr_l__mod___blocks___2_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_95 = self.getattr_getattr_L__mod___blocks___2_____2___bn1_drop(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_97 = self.getattr_getattr_L__mod___blocks___2_____2___bn1_act(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_10 = torch.functional.split(x_97, [168, 168], 1);  x_97 = None
    getitem_23 = split_10[0]
    getitem_24 = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____2___conv_dw_0 = self.getattr_getattr_L__mod___blocks___2_____2___conv_dw_0(getitem_23);  getitem_23 = None
    getattr_getattr_l__mod___blocks___2_____2___conv_dw_1 = self.getattr_getattr_L__mod___blocks___2_____2___conv_dw_1(getitem_24);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_99 = torch.cat([getattr_getattr_l__mod___blocks___2_____2___conv_dw_0, getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], 1);  getattr_getattr_l__mod___blocks___2_____2___conv_dw_0 = getattr_getattr_l__mod___blocks___2_____2___conv_dw_1 = None
    
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
    x_100 = torch.nn.functional.batch_norm(x_99, getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn2_running_var, getattr_getattr_l__mod___blocks___2_____2___bn2_weight, getattr_getattr_l__mod___blocks___2_____2___bn2_bias, True, 0.1, 1e-05);  x_99 = getattr_getattr_l__mod___blocks___2_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn2_running_var = getattr_getattr_l__mod___blocks___2_____2___bn2_weight = getattr_getattr_l__mod___blocks___2_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_101 = self.getattr_getattr_L__mod___blocks___2_____2___bn2_drop(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_103 = self.getattr_getattr_L__mod___blocks___2_____2___bn2_act(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_103.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_9 = self.getattr_getattr_L__mod___blocks___2_____2___se_conv_reduce(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_10 = self.getattr_getattr_L__mod___blocks___2_____2___se_act1(x_se_9);  x_se_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_11 = self.getattr_getattr_L__mod___blocks___2_____2___se_conv_expand(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____2___se_gate = self.getattr_getattr_L__mod___blocks___2_____2___se_gate(x_se_11);  x_se_11 = None
    x_104 = x_103 * getattr_getattr_l__mod___blocks___2_____2___se_gate;  x_103 = getattr_getattr_l__mod___blocks___2_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_11 = torch.functional.split(x_104, [168, 168], 1);  x_104 = None
    getitem_25 = split_11[0]
    getitem_26 = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pwl_0(getitem_25);  getitem_25 = None
    getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___2_____2___conv_pwl_1(getitem_26);  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_106 = torch.cat([getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0, getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0 = getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1 = None
    
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
    x_107 = torch.nn.functional.batch_norm(x_106, getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____2___bn3_running_var, getattr_getattr_l__mod___blocks___2_____2___bn3_weight, getattr_getattr_l__mod___blocks___2_____2___bn3_bias, True, 0.1, 1e-05);  x_106 = getattr_getattr_l__mod___blocks___2_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____2___bn3_running_var = getattr_getattr_l__mod___blocks___2_____2___bn3_weight = getattr_getattr_l__mod___blocks___2_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_108 = self.getattr_getattr_L__mod___blocks___2_____2___bn3_drop(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_110 = self.getattr_getattr_L__mod___blocks___2_____2___bn3_act(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____2___drop_path = self.getattr_getattr_L__mod___blocks___2_____2___drop_path(x_110);  x_110 = None
    shortcut_6 = getattr_getattr_l__mod___blocks___2_____2___drop_path + shortcut_5;  getattr_getattr_l__mod___blocks___2_____2___drop_path = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_12 = torch.functional.split(shortcut_6, [28, 28], 1)
    getitem_27 = split_12[0]
    getitem_28 = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____3___conv_pw_0 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pw_0(getitem_27);  getitem_27 = None
    getattr_getattr_l__mod___blocks___2_____3___conv_pw_1 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pw_1(getitem_28);  getitem_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_113 = torch.cat([getattr_getattr_l__mod___blocks___2_____3___conv_pw_0, getattr_getattr_l__mod___blocks___2_____3___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___2_____3___conv_pw_0 = getattr_getattr_l__mod___blocks___2_____3___conv_pw_1 = None
    
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
    x_114 = torch.nn.functional.batch_norm(x_113, getattr_getattr_l__mod___blocks___2_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn1_running_var, getattr_getattr_l__mod___blocks___2_____3___bn1_weight, getattr_getattr_l__mod___blocks___2_____3___bn1_bias, True, 0.1, 1e-05);  x_113 = getattr_getattr_l__mod___blocks___2_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn1_running_var = getattr_getattr_l__mod___blocks___2_____3___bn1_weight = getattr_getattr_l__mod___blocks___2_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_115 = self.getattr_getattr_L__mod___blocks___2_____3___bn1_drop(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_117 = self.getattr_getattr_L__mod___blocks___2_____3___bn1_act(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_13 = torch.functional.split(x_117, [168, 168], 1);  x_117 = None
    getitem_29 = split_13[0]
    getitem_30 = split_13[1];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____3___conv_dw_0 = self.getattr_getattr_L__mod___blocks___2_____3___conv_dw_0(getitem_29);  getitem_29 = None
    getattr_getattr_l__mod___blocks___2_____3___conv_dw_1 = self.getattr_getattr_L__mod___blocks___2_____3___conv_dw_1(getitem_30);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_119 = torch.cat([getattr_getattr_l__mod___blocks___2_____3___conv_dw_0, getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], 1);  getattr_getattr_l__mod___blocks___2_____3___conv_dw_0 = getattr_getattr_l__mod___blocks___2_____3___conv_dw_1 = None
    
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
    x_120 = torch.nn.functional.batch_norm(x_119, getattr_getattr_l__mod___blocks___2_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn2_running_var, getattr_getattr_l__mod___blocks___2_____3___bn2_weight, getattr_getattr_l__mod___blocks___2_____3___bn2_bias, True, 0.1, 1e-05);  x_119 = getattr_getattr_l__mod___blocks___2_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn2_running_var = getattr_getattr_l__mod___blocks___2_____3___bn2_weight = getattr_getattr_l__mod___blocks___2_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_121 = self.getattr_getattr_L__mod___blocks___2_____3___bn2_drop(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_123 = self.getattr_getattr_L__mod___blocks___2_____3___bn2_act(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_123.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_13 = self.getattr_getattr_L__mod___blocks___2_____3___se_conv_reduce(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_14 = self.getattr_getattr_L__mod___blocks___2_____3___se_act1(x_se_13);  x_se_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_15 = self.getattr_getattr_L__mod___blocks___2_____3___se_conv_expand(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____3___se_gate = self.getattr_getattr_L__mod___blocks___2_____3___se_gate(x_se_15);  x_se_15 = None
    x_124 = x_123 * getattr_getattr_l__mod___blocks___2_____3___se_gate;  x_123 = getattr_getattr_l__mod___blocks___2_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_14 = torch.functional.split(x_124, [168, 168], 1);  x_124 = None
    getitem_31 = split_14[0]
    getitem_32 = split_14[1];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pwl_0(getitem_31);  getitem_31 = None
    getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___2_____3___conv_pwl_1(getitem_32);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_126 = torch.cat([getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0, getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0 = getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1 = None
    
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
    x_127 = torch.nn.functional.batch_norm(x_126, getattr_getattr_l__mod___blocks___2_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____3___bn3_running_var, getattr_getattr_l__mod___blocks___2_____3___bn3_weight, getattr_getattr_l__mod___blocks___2_____3___bn3_bias, True, 0.1, 1e-05);  x_126 = getattr_getattr_l__mod___blocks___2_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____3___bn3_running_var = getattr_getattr_l__mod___blocks___2_____3___bn3_weight = getattr_getattr_l__mod___blocks___2_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_128 = self.getattr_getattr_L__mod___blocks___2_____3___bn3_drop(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_130 = self.getattr_getattr_L__mod___blocks___2_____3___bn3_act(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____3___drop_path = self.getattr_getattr_L__mod___blocks___2_____3___drop_path(x_130);  x_130 = None
    shortcut_7 = getattr_getattr_l__mod___blocks___2_____3___drop_path + shortcut_6;  getattr_getattr_l__mod___blocks___2_____3___drop_path = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_132 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pw(shortcut_7);  shortcut_7 = None
    
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
    x_133 = torch.nn.functional.batch_norm(x_132, getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn1_running_var, getattr_getattr_l__mod___blocks___3_____0___bn1_weight, getattr_getattr_l__mod___blocks___3_____0___bn1_bias, True, 0.1, 1e-05);  x_132 = getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = getattr_getattr_l__mod___blocks___3_____0___bn1_weight = getattr_getattr_l__mod___blocks___3_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_134 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_drop(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_136 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_act(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_15 = torch.functional.split(x_136, [112, 112, 112], 1);  x_136 = None
    getitem_33 = split_15[0]
    getitem_34 = split_15[1]
    getitem_35 = split_15[2];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____0___conv_dw_0 = self.getattr_getattr_L__mod___blocks___3_____0___conv_dw_0(getitem_33);  getitem_33 = None
    getattr_getattr_l__mod___blocks___3_____0___conv_dw_1 = self.getattr_getattr_L__mod___blocks___3_____0___conv_dw_1(getitem_34);  getitem_34 = None
    getattr_getattr_l__mod___blocks___3_____0___conv_dw_2 = self.getattr_getattr_L__mod___blocks___3_____0___conv_dw_2(getitem_35);  getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_138 = torch.cat([getattr_getattr_l__mod___blocks___3_____0___conv_dw_0, getattr_getattr_l__mod___blocks___3_____0___conv_dw_1, getattr_getattr_l__mod___blocks___3_____0___conv_dw_2], 1);  getattr_getattr_l__mod___blocks___3_____0___conv_dw_0 = getattr_getattr_l__mod___blocks___3_____0___conv_dw_1 = getattr_getattr_l__mod___blocks___3_____0___conv_dw_2 = None
    
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
    x_139 = torch.nn.functional.batch_norm(x_138, getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn2_running_var, getattr_getattr_l__mod___blocks___3_____0___bn2_weight, getattr_getattr_l__mod___blocks___3_____0___bn2_bias, True, 0.1, 1e-05);  x_138 = getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = getattr_getattr_l__mod___blocks___3_____0___bn2_weight = getattr_getattr_l__mod___blocks___3_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_140 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_drop(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_142 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_act(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_142.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_17 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_reduce(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_18 = self.getattr_getattr_L__mod___blocks___3_____0___se_act1(x_se_17);  x_se_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_19 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_expand(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____0___se_gate = self.getattr_getattr_L__mod___blocks___3_____0___se_gate(x_se_19);  x_se_19 = None
    x_143 = x_142 * getattr_getattr_l__mod___blocks___3_____0___se_gate;  x_142 = getattr_getattr_l__mod___blocks___3_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_144 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pwl(x_143);  x_143 = None
    
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
    x_145 = torch.nn.functional.batch_norm(x_144, getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn3_running_var, getattr_getattr_l__mod___blocks___3_____0___bn3_weight, getattr_getattr_l__mod___blocks___3_____0___bn3_bias, True, 0.1, 1e-05);  x_144 = getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = getattr_getattr_l__mod___blocks___3_____0___bn3_weight = getattr_getattr_l__mod___blocks___3_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_146 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_drop(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_8 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_act(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_16 = torch.functional.split(shortcut_8, [52, 52], 1)
    getitem_36 = split_16[0]
    getitem_37 = split_16[1];  split_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____1___conv_pw_0 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pw_0(getitem_36);  getitem_36 = None
    getattr_getattr_l__mod___blocks___3_____1___conv_pw_1 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pw_1(getitem_37);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_150 = torch.cat([getattr_getattr_l__mod___blocks___3_____1___conv_pw_0, getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___3_____1___conv_pw_0 = getattr_getattr_l__mod___blocks___3_____1___conv_pw_1 = None
    
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
    x_151 = torch.nn.functional.batch_norm(x_150, getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn1_running_var, getattr_getattr_l__mod___blocks___3_____1___bn1_weight, getattr_getattr_l__mod___blocks___3_____1___bn1_bias, True, 0.1, 1e-05);  x_150 = getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = getattr_getattr_l__mod___blocks___3_____1___bn1_weight = getattr_getattr_l__mod___blocks___3_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_152 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_drop(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_154 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_act(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_17 = torch.functional.split(x_154, [156, 156, 156, 156], 1);  x_154 = None
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
    x_156 = torch.cat([getattr_getattr_l__mod___blocks___3_____1___conv_dw_0, getattr_getattr_l__mod___blocks___3_____1___conv_dw_1, getattr_getattr_l__mod___blocks___3_____1___conv_dw_2, getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___3_____1___conv_dw_0 = getattr_getattr_l__mod___blocks___3_____1___conv_dw_1 = getattr_getattr_l__mod___blocks___3_____1___conv_dw_2 = getattr_getattr_l__mod___blocks___3_____1___conv_dw_3 = None
    
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
    x_157 = torch.nn.functional.batch_norm(x_156, getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn2_running_var, getattr_getattr_l__mod___blocks___3_____1___bn2_weight, getattr_getattr_l__mod___blocks___3_____1___bn2_bias, True, 0.1, 1e-05);  x_156 = getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = getattr_getattr_l__mod___blocks___3_____1___bn2_weight = getattr_getattr_l__mod___blocks___3_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_158 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_drop(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_160 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_act(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_160.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_21 = self.getattr_getattr_L__mod___blocks___3_____1___se_conv_reduce(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_22 = self.getattr_getattr_L__mod___blocks___3_____1___se_act1(x_se_21);  x_se_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_23 = self.getattr_getattr_L__mod___blocks___3_____1___se_conv_expand(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____1___se_gate = self.getattr_getattr_L__mod___blocks___3_____1___se_gate(x_se_23);  x_se_23 = None
    x_161 = x_160 * getattr_getattr_l__mod___blocks___3_____1___se_gate;  x_160 = getattr_getattr_l__mod___blocks___3_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_18 = torch.functional.split(x_161, [312, 312], 1);  x_161 = None
    getitem_42 = split_18[0]
    getitem_43 = split_18[1];  split_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pwl_0(getitem_42);  getitem_42 = None
    getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pwl_1(getitem_43);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_163 = torch.cat([getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1 = None
    
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
    x_164 = torch.nn.functional.batch_norm(x_163, getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn3_running_var, getattr_getattr_l__mod___blocks___3_____1___bn3_weight, getattr_getattr_l__mod___blocks___3_____1___bn3_bias, True, 0.1, 1e-05);  x_163 = getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = getattr_getattr_l__mod___blocks___3_____1___bn3_weight = getattr_getattr_l__mod___blocks___3_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_165 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_drop(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_167 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_act(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____1___drop_path = self.getattr_getattr_L__mod___blocks___3_____1___drop_path(x_167);  x_167 = None
    shortcut_9 = getattr_getattr_l__mod___blocks___3_____1___drop_path + shortcut_8;  getattr_getattr_l__mod___blocks___3_____1___drop_path = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_19 = torch.functional.split(shortcut_9, [52, 52], 1)
    getitem_44 = split_19[0]
    getitem_45 = split_19[1];  split_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____2___conv_pw_0 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pw_0(getitem_44);  getitem_44 = None
    getattr_getattr_l__mod___blocks___3_____2___conv_pw_1 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pw_1(getitem_45);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_170 = torch.cat([getattr_getattr_l__mod___blocks___3_____2___conv_pw_0, getattr_getattr_l__mod___blocks___3_____2___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___3_____2___conv_pw_0 = getattr_getattr_l__mod___blocks___3_____2___conv_pw_1 = None
    
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
    x_171 = torch.nn.functional.batch_norm(x_170, getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn1_running_var, getattr_getattr_l__mod___blocks___3_____2___bn1_weight, getattr_getattr_l__mod___blocks___3_____2___bn1_bias, True, 0.1, 1e-05);  x_170 = getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = getattr_getattr_l__mod___blocks___3_____2___bn1_weight = getattr_getattr_l__mod___blocks___3_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_172 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_drop(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_174 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_act(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_20 = torch.functional.split(x_174, [156, 156, 156, 156], 1);  x_174 = None
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
    x_176 = torch.cat([getattr_getattr_l__mod___blocks___3_____2___conv_dw_0, getattr_getattr_l__mod___blocks___3_____2___conv_dw_1, getattr_getattr_l__mod___blocks___3_____2___conv_dw_2, getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___3_____2___conv_dw_0 = getattr_getattr_l__mod___blocks___3_____2___conv_dw_1 = getattr_getattr_l__mod___blocks___3_____2___conv_dw_2 = getattr_getattr_l__mod___blocks___3_____2___conv_dw_3 = None
    
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
    x_177 = torch.nn.functional.batch_norm(x_176, getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn2_running_var, getattr_getattr_l__mod___blocks___3_____2___bn2_weight, getattr_getattr_l__mod___blocks___3_____2___bn2_bias, True, 0.1, 1e-05);  x_176 = getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = getattr_getattr_l__mod___blocks___3_____2___bn2_weight = getattr_getattr_l__mod___blocks___3_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_178 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_drop(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_180 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_act(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_180.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_25 = self.getattr_getattr_L__mod___blocks___3_____2___se_conv_reduce(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_26 = self.getattr_getattr_L__mod___blocks___3_____2___se_act1(x_se_25);  x_se_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_27 = self.getattr_getattr_L__mod___blocks___3_____2___se_conv_expand(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____2___se_gate = self.getattr_getattr_L__mod___blocks___3_____2___se_gate(x_se_27);  x_se_27 = None
    x_181 = x_180 * getattr_getattr_l__mod___blocks___3_____2___se_gate;  x_180 = getattr_getattr_l__mod___blocks___3_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_21 = torch.functional.split(x_181, [312, 312], 1);  x_181 = None
    getitem_50 = split_21[0]
    getitem_51 = split_21[1];  split_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pwl_0(getitem_50);  getitem_50 = None
    getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pwl_1(getitem_51);  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_183 = torch.cat([getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0, getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0 = getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1 = None
    
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
    x_184 = torch.nn.functional.batch_norm(x_183, getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn3_running_var, getattr_getattr_l__mod___blocks___3_____2___bn3_weight, getattr_getattr_l__mod___blocks___3_____2___bn3_bias, True, 0.1, 1e-05);  x_183 = getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = getattr_getattr_l__mod___blocks___3_____2___bn3_weight = getattr_getattr_l__mod___blocks___3_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_185 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_drop(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_187 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_act(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____2___drop_path = self.getattr_getattr_L__mod___blocks___3_____2___drop_path(x_187);  x_187 = None
    shortcut_10 = getattr_getattr_l__mod___blocks___3_____2___drop_path + shortcut_9;  getattr_getattr_l__mod___blocks___3_____2___drop_path = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_22 = torch.functional.split(shortcut_10, [52, 52], 1)
    getitem_52 = split_22[0]
    getitem_53 = split_22[1];  split_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____3___conv_pw_0 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pw_0(getitem_52);  getitem_52 = None
    getattr_getattr_l__mod___blocks___3_____3___conv_pw_1 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pw_1(getitem_53);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_190 = torch.cat([getattr_getattr_l__mod___blocks___3_____3___conv_pw_0, getattr_getattr_l__mod___blocks___3_____3___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___3_____3___conv_pw_0 = getattr_getattr_l__mod___blocks___3_____3___conv_pw_1 = None
    
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
    x_191 = torch.nn.functional.batch_norm(x_190, getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn1_running_var, getattr_getattr_l__mod___blocks___3_____3___bn1_weight, getattr_getattr_l__mod___blocks___3_____3___bn1_bias, True, 0.1, 1e-05);  x_190 = getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = getattr_getattr_l__mod___blocks___3_____3___bn1_weight = getattr_getattr_l__mod___blocks___3_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_192 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_drop(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_194 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_act(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_23 = torch.functional.split(x_194, [156, 156, 156, 156], 1);  x_194 = None
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
    x_196 = torch.cat([getattr_getattr_l__mod___blocks___3_____3___conv_dw_0, getattr_getattr_l__mod___blocks___3_____3___conv_dw_1, getattr_getattr_l__mod___blocks___3_____3___conv_dw_2, getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___3_____3___conv_dw_0 = getattr_getattr_l__mod___blocks___3_____3___conv_dw_1 = getattr_getattr_l__mod___blocks___3_____3___conv_dw_2 = getattr_getattr_l__mod___blocks___3_____3___conv_dw_3 = None
    
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
    x_197 = torch.nn.functional.batch_norm(x_196, getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn2_running_var, getattr_getattr_l__mod___blocks___3_____3___bn2_weight, getattr_getattr_l__mod___blocks___3_____3___bn2_bias, True, 0.1, 1e-05);  x_196 = getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = getattr_getattr_l__mod___blocks___3_____3___bn2_weight = getattr_getattr_l__mod___blocks___3_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_198 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_drop(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_200 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_act(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = x_200.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_29 = self.getattr_getattr_L__mod___blocks___3_____3___se_conv_reduce(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_30 = self.getattr_getattr_L__mod___blocks___3_____3___se_act1(x_se_29);  x_se_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_31 = self.getattr_getattr_L__mod___blocks___3_____3___se_conv_expand(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____3___se_gate = self.getattr_getattr_L__mod___blocks___3_____3___se_gate(x_se_31);  x_se_31 = None
    x_201 = x_200 * getattr_getattr_l__mod___blocks___3_____3___se_gate;  x_200 = getattr_getattr_l__mod___blocks___3_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_24 = torch.functional.split(x_201, [312, 312], 1);  x_201 = None
    getitem_58 = split_24[0]
    getitem_59 = split_24[1];  split_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pwl_0(getitem_58);  getitem_58 = None
    getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pwl_1(getitem_59);  getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_203 = torch.cat([getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0, getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0 = getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1 = None
    
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
    x_204 = torch.nn.functional.batch_norm(x_203, getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn3_running_var, getattr_getattr_l__mod___blocks___3_____3___bn3_weight, getattr_getattr_l__mod___blocks___3_____3___bn3_bias, True, 0.1, 1e-05);  x_203 = getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = getattr_getattr_l__mod___blocks___3_____3___bn3_weight = getattr_getattr_l__mod___blocks___3_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_205 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_drop(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_207 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_act(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____3___drop_path = self.getattr_getattr_L__mod___blocks___3_____3___drop_path(x_207);  x_207 = None
    shortcut_11 = getattr_getattr_l__mod___blocks___3_____3___drop_path + shortcut_10;  getattr_getattr_l__mod___blocks___3_____3___drop_path = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_209 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pw(shortcut_11);  shortcut_11 = None
    
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
    x_210 = torch.nn.functional.batch_norm(x_209, getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn1_running_var, getattr_getattr_l__mod___blocks___4_____0___bn1_weight, getattr_getattr_l__mod___blocks___4_____0___bn1_bias, True, 0.1, 1e-05);  x_209 = getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = getattr_getattr_l__mod___blocks___4_____0___bn1_weight = getattr_getattr_l__mod___blocks___4_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_211 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_drop(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_213 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_act(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_214 = self.getattr_getattr_L__mod___blocks___4_____0___conv_dw(x_213);  x_213 = None
    
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
    x_215 = torch.nn.functional.batch_norm(x_214, getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn2_running_var, getattr_getattr_l__mod___blocks___4_____0___bn2_weight, getattr_getattr_l__mod___blocks___4_____0___bn2_bias, True, 0.1, 1e-05);  x_214 = getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = getattr_getattr_l__mod___blocks___4_____0___bn2_weight = getattr_getattr_l__mod___blocks___4_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_216 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_drop(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_218 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_act(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_32 = x_218.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_33 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_reduce(x_se_32);  x_se_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_34 = self.getattr_getattr_L__mod___blocks___4_____0___se_act1(x_se_33);  x_se_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_35 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_expand(x_se_34);  x_se_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____0___se_gate = self.getattr_getattr_L__mod___blocks___4_____0___se_gate(x_se_35);  x_se_35 = None
    x_219 = x_218 * getattr_getattr_l__mod___blocks___4_____0___se_gate;  x_218 = getattr_getattr_l__mod___blocks___4_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_220 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pwl(x_219);  x_219 = None
    
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
    x_221 = torch.nn.functional.batch_norm(x_220, getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn3_running_var, getattr_getattr_l__mod___blocks___4_____0___bn3_weight, getattr_getattr_l__mod___blocks___4_____0___bn3_bias, True, 0.1, 1e-05);  x_220 = getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = getattr_getattr_l__mod___blocks___4_____0___bn3_weight = getattr_getattr_l__mod___blocks___4_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_222 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_drop(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_12 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_act(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_25 = torch.functional.split(shortcut_12, [80, 80], 1)
    getitem_60 = split_25[0]
    getitem_61 = split_25[1];  split_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____1___conv_pw_0 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pw_0(getitem_60);  getitem_60 = None
    getattr_getattr_l__mod___blocks___4_____1___conv_pw_1 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pw_1(getitem_61);  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_226 = torch.cat([getattr_getattr_l__mod___blocks___4_____1___conv_pw_0, getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___4_____1___conv_pw_0 = getattr_getattr_l__mod___blocks___4_____1___conv_pw_1 = None
    
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
    x_227 = torch.nn.functional.batch_norm(x_226, getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn1_running_var, getattr_getattr_l__mod___blocks___4_____1___bn1_weight, getattr_getattr_l__mod___blocks___4_____1___bn1_bias, True, 0.1, 1e-05);  x_226 = getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = getattr_getattr_l__mod___blocks___4_____1___bn1_weight = getattr_getattr_l__mod___blocks___4_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_228 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_drop(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_230 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_act(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_26 = torch.functional.split(x_230, [120, 120, 120, 120], 1);  x_230 = None
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
    x_232 = torch.cat([getattr_getattr_l__mod___blocks___4_____1___conv_dw_0, getattr_getattr_l__mod___blocks___4_____1___conv_dw_1, getattr_getattr_l__mod___blocks___4_____1___conv_dw_2, getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___4_____1___conv_dw_0 = getattr_getattr_l__mod___blocks___4_____1___conv_dw_1 = getattr_getattr_l__mod___blocks___4_____1___conv_dw_2 = getattr_getattr_l__mod___blocks___4_____1___conv_dw_3 = None
    
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
    x_233 = torch.nn.functional.batch_norm(x_232, getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn2_running_var, getattr_getattr_l__mod___blocks___4_____1___bn2_weight, getattr_getattr_l__mod___blocks___4_____1___bn2_bias, True, 0.1, 1e-05);  x_232 = getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = getattr_getattr_l__mod___blocks___4_____1___bn2_weight = getattr_getattr_l__mod___blocks___4_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_234 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_drop(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_236 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_act(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_36 = x_236.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_37 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_reduce(x_se_36);  x_se_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_38 = self.getattr_getattr_L__mod___blocks___4_____1___se_act1(x_se_37);  x_se_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_39 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_expand(x_se_38);  x_se_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____1___se_gate = self.getattr_getattr_L__mod___blocks___4_____1___se_gate(x_se_39);  x_se_39 = None
    x_237 = x_236 * getattr_getattr_l__mod___blocks___4_____1___se_gate;  x_236 = getattr_getattr_l__mod___blocks___4_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_27 = torch.functional.split(x_237, [240, 240], 1);  x_237 = None
    getitem_66 = split_27[0]
    getitem_67 = split_27[1];  split_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pwl_0(getitem_66);  getitem_66 = None
    getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pwl_1(getitem_67);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_239 = torch.cat([getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1 = None
    
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
    x_240 = torch.nn.functional.batch_norm(x_239, getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn3_running_var, getattr_getattr_l__mod___blocks___4_____1___bn3_weight, getattr_getattr_l__mod___blocks___4_____1___bn3_bias, True, 0.1, 1e-05);  x_239 = getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = getattr_getattr_l__mod___blocks___4_____1___bn3_weight = getattr_getattr_l__mod___blocks___4_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_241 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_drop(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_243 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_act(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____1___drop_path = self.getattr_getattr_L__mod___blocks___4_____1___drop_path(x_243);  x_243 = None
    shortcut_13 = getattr_getattr_l__mod___blocks___4_____1___drop_path + shortcut_12;  getattr_getattr_l__mod___blocks___4_____1___drop_path = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_28 = torch.functional.split(shortcut_13, [80, 80], 1)
    getitem_68 = split_28[0]
    getitem_69 = split_28[1];  split_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____2___conv_pw_0 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pw_0(getitem_68);  getitem_68 = None
    getattr_getattr_l__mod___blocks___4_____2___conv_pw_1 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pw_1(getitem_69);  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_246 = torch.cat([getattr_getattr_l__mod___blocks___4_____2___conv_pw_0, getattr_getattr_l__mod___blocks___4_____2___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___4_____2___conv_pw_0 = getattr_getattr_l__mod___blocks___4_____2___conv_pw_1 = None
    
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
    x_247 = torch.nn.functional.batch_norm(x_246, getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn1_running_var, getattr_getattr_l__mod___blocks___4_____2___bn1_weight, getattr_getattr_l__mod___blocks___4_____2___bn1_bias, True, 0.1, 1e-05);  x_246 = getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = getattr_getattr_l__mod___blocks___4_____2___bn1_weight = getattr_getattr_l__mod___blocks___4_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_248 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_drop(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_250 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_act(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_29 = torch.functional.split(x_250, [120, 120, 120, 120], 1);  x_250 = None
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
    x_252 = torch.cat([getattr_getattr_l__mod___blocks___4_____2___conv_dw_0, getattr_getattr_l__mod___blocks___4_____2___conv_dw_1, getattr_getattr_l__mod___blocks___4_____2___conv_dw_2, getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___4_____2___conv_dw_0 = getattr_getattr_l__mod___blocks___4_____2___conv_dw_1 = getattr_getattr_l__mod___blocks___4_____2___conv_dw_2 = getattr_getattr_l__mod___blocks___4_____2___conv_dw_3 = None
    
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
    x_253 = torch.nn.functional.batch_norm(x_252, getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn2_running_var, getattr_getattr_l__mod___blocks___4_____2___bn2_weight, getattr_getattr_l__mod___blocks___4_____2___bn2_bias, True, 0.1, 1e-05);  x_252 = getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = getattr_getattr_l__mod___blocks___4_____2___bn2_weight = getattr_getattr_l__mod___blocks___4_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_254 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_drop(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_256 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_act(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_40 = x_256.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_41 = self.getattr_getattr_L__mod___blocks___4_____2___se_conv_reduce(x_se_40);  x_se_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_42 = self.getattr_getattr_L__mod___blocks___4_____2___se_act1(x_se_41);  x_se_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_43 = self.getattr_getattr_L__mod___blocks___4_____2___se_conv_expand(x_se_42);  x_se_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____2___se_gate = self.getattr_getattr_L__mod___blocks___4_____2___se_gate(x_se_43);  x_se_43 = None
    x_257 = x_256 * getattr_getattr_l__mod___blocks___4_____2___se_gate;  x_256 = getattr_getattr_l__mod___blocks___4_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_30 = torch.functional.split(x_257, [240, 240], 1);  x_257 = None
    getitem_74 = split_30[0]
    getitem_75 = split_30[1];  split_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pwl_0(getitem_74);  getitem_74 = None
    getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pwl_1(getitem_75);  getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_259 = torch.cat([getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0, getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0 = getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1 = None
    
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
    x_260 = torch.nn.functional.batch_norm(x_259, getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn3_running_var, getattr_getattr_l__mod___blocks___4_____2___bn3_weight, getattr_getattr_l__mod___blocks___4_____2___bn3_bias, True, 0.1, 1e-05);  x_259 = getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = getattr_getattr_l__mod___blocks___4_____2___bn3_weight = getattr_getattr_l__mod___blocks___4_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_261 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_drop(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_263 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_act(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____2___drop_path = self.getattr_getattr_L__mod___blocks___4_____2___drop_path(x_263);  x_263 = None
    shortcut_14 = getattr_getattr_l__mod___blocks___4_____2___drop_path + shortcut_13;  getattr_getattr_l__mod___blocks___4_____2___drop_path = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_31 = torch.functional.split(shortcut_14, [80, 80], 1)
    getitem_76 = split_31[0]
    getitem_77 = split_31[1];  split_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____3___conv_pw_0 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pw_0(getitem_76);  getitem_76 = None
    getattr_getattr_l__mod___blocks___4_____3___conv_pw_1 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pw_1(getitem_77);  getitem_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_266 = torch.cat([getattr_getattr_l__mod___blocks___4_____3___conv_pw_0, getattr_getattr_l__mod___blocks___4_____3___conv_pw_1], 1);  getattr_getattr_l__mod___blocks___4_____3___conv_pw_0 = getattr_getattr_l__mod___blocks___4_____3___conv_pw_1 = None
    
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
    x_267 = torch.nn.functional.batch_norm(x_266, getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn1_running_var, getattr_getattr_l__mod___blocks___4_____3___bn1_weight, getattr_getattr_l__mod___blocks___4_____3___bn1_bias, True, 0.1, 1e-05);  x_266 = getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn1_running_var = getattr_getattr_l__mod___blocks___4_____3___bn1_weight = getattr_getattr_l__mod___blocks___4_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_268 = self.getattr_getattr_L__mod___blocks___4_____3___bn1_drop(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_270 = self.getattr_getattr_L__mod___blocks___4_____3___bn1_act(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_32 = torch.functional.split(x_270, [120, 120, 120, 120], 1);  x_270 = None
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
    x_272 = torch.cat([getattr_getattr_l__mod___blocks___4_____3___conv_dw_0, getattr_getattr_l__mod___blocks___4_____3___conv_dw_1, getattr_getattr_l__mod___blocks___4_____3___conv_dw_2, getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___4_____3___conv_dw_0 = getattr_getattr_l__mod___blocks___4_____3___conv_dw_1 = getattr_getattr_l__mod___blocks___4_____3___conv_dw_2 = getattr_getattr_l__mod___blocks___4_____3___conv_dw_3 = None
    
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
    x_273 = torch.nn.functional.batch_norm(x_272, getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn2_running_var, getattr_getattr_l__mod___blocks___4_____3___bn2_weight, getattr_getattr_l__mod___blocks___4_____3___bn2_bias, True, 0.1, 1e-05);  x_272 = getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn2_running_var = getattr_getattr_l__mod___blocks___4_____3___bn2_weight = getattr_getattr_l__mod___blocks___4_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_274 = self.getattr_getattr_L__mod___blocks___4_____3___bn2_drop(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_276 = self.getattr_getattr_L__mod___blocks___4_____3___bn2_act(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_44 = x_276.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_45 = self.getattr_getattr_L__mod___blocks___4_____3___se_conv_reduce(x_se_44);  x_se_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_46 = self.getattr_getattr_L__mod___blocks___4_____3___se_act1(x_se_45);  x_se_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_47 = self.getattr_getattr_L__mod___blocks___4_____3___se_conv_expand(x_se_46);  x_se_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____3___se_gate = self.getattr_getattr_L__mod___blocks___4_____3___se_gate(x_se_47);  x_se_47 = None
    x_277 = x_276 * getattr_getattr_l__mod___blocks___4_____3___se_gate;  x_276 = getattr_getattr_l__mod___blocks___4_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_33 = torch.functional.split(x_277, [240, 240], 1);  x_277 = None
    getitem_82 = split_33[0]
    getitem_83 = split_33[1];  split_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pwl_0(getitem_82);  getitem_82 = None
    getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pwl_1(getitem_83);  getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_279 = torch.cat([getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0, getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0 = getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1 = None
    
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
    x_280 = torch.nn.functional.batch_norm(x_279, getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn3_running_var, getattr_getattr_l__mod___blocks___4_____3___bn3_weight, getattr_getattr_l__mod___blocks___4_____3___bn3_bias, True, 0.1, 1e-05);  x_279 = getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn3_running_var = getattr_getattr_l__mod___blocks___4_____3___bn3_weight = getattr_getattr_l__mod___blocks___4_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_281 = self.getattr_getattr_L__mod___blocks___4_____3___bn3_drop(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_283 = self.getattr_getattr_L__mod___blocks___4_____3___bn3_act(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____3___drop_path = self.getattr_getattr_L__mod___blocks___4_____3___drop_path(x_283);  x_283 = None
    shortcut_15 = getattr_getattr_l__mod___blocks___4_____3___drop_path + shortcut_14;  getattr_getattr_l__mod___blocks___4_____3___drop_path = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_285 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pw(shortcut_15);  shortcut_15 = None
    
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
    x_286 = torch.nn.functional.batch_norm(x_285, getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn1_running_var, getattr_getattr_l__mod___blocks___5_____0___bn1_weight, getattr_getattr_l__mod___blocks___5_____0___bn1_bias, True, 0.1, 1e-05);  x_285 = getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = getattr_getattr_l__mod___blocks___5_____0___bn1_weight = getattr_getattr_l__mod___blocks___5_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_287 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_drop(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_289 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_act(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_34 = torch.functional.split(x_289, [240, 240, 240, 240], 1);  x_289 = None
    getitem_84 = split_34[0]
    getitem_85 = split_34[1]
    getitem_86 = split_34[2]
    getitem_87 = split_34[3];  split_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____0___conv_dw_0 = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw_0(getitem_84);  getitem_84 = None
    getattr_getattr_l__mod___blocks___5_____0___conv_dw_1 = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw_1(getitem_85);  getitem_85 = None
    getattr_getattr_l__mod___blocks___5_____0___conv_dw_2 = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw_2(getitem_86);  getitem_86 = None
    getattr_getattr_l__mod___blocks___5_____0___conv_dw_3 = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw_3(getitem_87);  getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_291 = torch.cat([getattr_getattr_l__mod___blocks___5_____0___conv_dw_0, getattr_getattr_l__mod___blocks___5_____0___conv_dw_1, getattr_getattr_l__mod___blocks___5_____0___conv_dw_2, getattr_getattr_l__mod___blocks___5_____0___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___5_____0___conv_dw_0 = getattr_getattr_l__mod___blocks___5_____0___conv_dw_1 = getattr_getattr_l__mod___blocks___5_____0___conv_dw_2 = getattr_getattr_l__mod___blocks___5_____0___conv_dw_3 = None
    
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
    x_292 = torch.nn.functional.batch_norm(x_291, getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn2_running_var, getattr_getattr_l__mod___blocks___5_____0___bn2_weight, getattr_getattr_l__mod___blocks___5_____0___bn2_bias, True, 0.1, 1e-05);  x_291 = getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = getattr_getattr_l__mod___blocks___5_____0___bn2_weight = getattr_getattr_l__mod___blocks___5_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_293 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_drop(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_295 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_act(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_48 = x_295.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_49 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_reduce(x_se_48);  x_se_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_50 = self.getattr_getattr_L__mod___blocks___5_____0___se_act1(x_se_49);  x_se_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_51 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_expand(x_se_50);  x_se_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____0___se_gate = self.getattr_getattr_L__mod___blocks___5_____0___se_gate(x_se_51);  x_se_51 = None
    x_296 = x_295 * getattr_getattr_l__mod___blocks___5_____0___se_gate;  x_295 = getattr_getattr_l__mod___blocks___5_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_297 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pwl(x_296);  x_296 = None
    
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
    x_298 = torch.nn.functional.batch_norm(x_297, getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn3_running_var, getattr_getattr_l__mod___blocks___5_____0___bn3_weight, getattr_getattr_l__mod___blocks___5_____0___bn3_bias, True, 0.1, 1e-05);  x_297 = getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = getattr_getattr_l__mod___blocks___5_____0___bn3_weight = getattr_getattr_l__mod___blocks___5_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_299 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_drop(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_16 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_act(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_302 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pw(shortcut_16)
    
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
    x_303 = torch.nn.functional.batch_norm(x_302, getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn1_running_var, getattr_getattr_l__mod___blocks___5_____1___bn1_weight, getattr_getattr_l__mod___blocks___5_____1___bn1_bias, True, 0.1, 1e-05);  x_302 = getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = getattr_getattr_l__mod___blocks___5_____1___bn1_weight = getattr_getattr_l__mod___blocks___5_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_304 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_drop(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_306 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_act(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_35 = torch.functional.split(x_306, [396, 396, 396, 396], 1);  x_306 = None
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
    x_308 = torch.cat([getattr_getattr_l__mod___blocks___5_____1___conv_dw_0, getattr_getattr_l__mod___blocks___5_____1___conv_dw_1, getattr_getattr_l__mod___blocks___5_____1___conv_dw_2, getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___5_____1___conv_dw_0 = getattr_getattr_l__mod___blocks___5_____1___conv_dw_1 = getattr_getattr_l__mod___blocks___5_____1___conv_dw_2 = getattr_getattr_l__mod___blocks___5_____1___conv_dw_3 = None
    
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
    x_309 = torch.nn.functional.batch_norm(x_308, getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn2_running_var, getattr_getattr_l__mod___blocks___5_____1___bn2_weight, getattr_getattr_l__mod___blocks___5_____1___bn2_bias, True, 0.1, 1e-05);  x_308 = getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = getattr_getattr_l__mod___blocks___5_____1___bn2_weight = getattr_getattr_l__mod___blocks___5_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_310 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_drop(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_312 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_act(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_52 = x_312.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_53 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_reduce(x_se_52);  x_se_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_54 = self.getattr_getattr_L__mod___blocks___5_____1___se_act1(x_se_53);  x_se_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_55 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_expand(x_se_54);  x_se_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____1___se_gate = self.getattr_getattr_L__mod___blocks___5_____1___se_gate(x_se_55);  x_se_55 = None
    x_313 = x_312 * getattr_getattr_l__mod___blocks___5_____1___se_gate;  x_312 = getattr_getattr_l__mod___blocks___5_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_36 = torch.functional.split(x_313, [792, 792], 1);  x_313 = None
    getitem_92 = split_36[0]
    getitem_93 = split_36[1];  split_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pwl_0(getitem_92);  getitem_92 = None
    getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pwl_1(getitem_93);  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_315 = torch.cat([getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0, getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0 = getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1 = None
    
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
    x_316 = torch.nn.functional.batch_norm(x_315, getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn3_running_var, getattr_getattr_l__mod___blocks___5_____1___bn3_weight, getattr_getattr_l__mod___blocks___5_____1___bn3_bias, True, 0.1, 1e-05);  x_315 = getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = getattr_getattr_l__mod___blocks___5_____1___bn3_weight = getattr_getattr_l__mod___blocks___5_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_317 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_drop(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_319 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_act(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____1___drop_path = self.getattr_getattr_L__mod___blocks___5_____1___drop_path(x_319);  x_319 = None
    shortcut_17 = getattr_getattr_l__mod___blocks___5_____1___drop_path + shortcut_16;  getattr_getattr_l__mod___blocks___5_____1___drop_path = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_321 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pw(shortcut_17)
    
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
    x_322 = torch.nn.functional.batch_norm(x_321, getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn1_running_var, getattr_getattr_l__mod___blocks___5_____2___bn1_weight, getattr_getattr_l__mod___blocks___5_____2___bn1_bias, True, 0.1, 1e-05);  x_321 = getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = getattr_getattr_l__mod___blocks___5_____2___bn1_weight = getattr_getattr_l__mod___blocks___5_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_323 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_drop(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_325 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_act(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_37 = torch.functional.split(x_325, [396, 396, 396, 396], 1);  x_325 = None
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
    x_327 = torch.cat([getattr_getattr_l__mod___blocks___5_____2___conv_dw_0, getattr_getattr_l__mod___blocks___5_____2___conv_dw_1, getattr_getattr_l__mod___blocks___5_____2___conv_dw_2, getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___5_____2___conv_dw_0 = getattr_getattr_l__mod___blocks___5_____2___conv_dw_1 = getattr_getattr_l__mod___blocks___5_____2___conv_dw_2 = getattr_getattr_l__mod___blocks___5_____2___conv_dw_3 = None
    
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
    x_328 = torch.nn.functional.batch_norm(x_327, getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn2_running_var, getattr_getattr_l__mod___blocks___5_____2___bn2_weight, getattr_getattr_l__mod___blocks___5_____2___bn2_bias, True, 0.1, 1e-05);  x_327 = getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = getattr_getattr_l__mod___blocks___5_____2___bn2_weight = getattr_getattr_l__mod___blocks___5_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_329 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_drop(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_331 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_act(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_56 = x_331.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_57 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_reduce(x_se_56);  x_se_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_58 = self.getattr_getattr_L__mod___blocks___5_____2___se_act1(x_se_57);  x_se_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_59 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_expand(x_se_58);  x_se_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____2___se_gate = self.getattr_getattr_L__mod___blocks___5_____2___se_gate(x_se_59);  x_se_59 = None
    x_332 = x_331 * getattr_getattr_l__mod___blocks___5_____2___se_gate;  x_331 = getattr_getattr_l__mod___blocks___5_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_38 = torch.functional.split(x_332, [792, 792], 1);  x_332 = None
    getitem_98 = split_38[0]
    getitem_99 = split_38[1];  split_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pwl_0(getitem_98);  getitem_98 = None
    getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pwl_1(getitem_99);  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_334 = torch.cat([getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0, getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0 = getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1 = None
    
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
    x_335 = torch.nn.functional.batch_norm(x_334, getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn3_running_var, getattr_getattr_l__mod___blocks___5_____2___bn3_weight, getattr_getattr_l__mod___blocks___5_____2___bn3_bias, True, 0.1, 1e-05);  x_334 = getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = getattr_getattr_l__mod___blocks___5_____2___bn3_weight = getattr_getattr_l__mod___blocks___5_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_336 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_drop(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_338 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_act(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____2___drop_path = self.getattr_getattr_L__mod___blocks___5_____2___drop_path(x_338);  x_338 = None
    shortcut_18 = getattr_getattr_l__mod___blocks___5_____2___drop_path + shortcut_17;  getattr_getattr_l__mod___blocks___5_____2___drop_path = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_340 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pw(shortcut_18)
    
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
    x_341 = torch.nn.functional.batch_norm(x_340, getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn1_running_var, getattr_getattr_l__mod___blocks___5_____3___bn1_weight, getattr_getattr_l__mod___blocks___5_____3___bn1_bias, True, 0.1, 1e-05);  x_340 = getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = getattr_getattr_l__mod___blocks___5_____3___bn1_weight = getattr_getattr_l__mod___blocks___5_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_342 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_drop(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_344 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_act(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_39 = torch.functional.split(x_344, [396, 396, 396, 396], 1);  x_344 = None
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
    x_346 = torch.cat([getattr_getattr_l__mod___blocks___5_____3___conv_dw_0, getattr_getattr_l__mod___blocks___5_____3___conv_dw_1, getattr_getattr_l__mod___blocks___5_____3___conv_dw_2, getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], 1);  getattr_getattr_l__mod___blocks___5_____3___conv_dw_0 = getattr_getattr_l__mod___blocks___5_____3___conv_dw_1 = getattr_getattr_l__mod___blocks___5_____3___conv_dw_2 = getattr_getattr_l__mod___blocks___5_____3___conv_dw_3 = None
    
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
    x_347 = torch.nn.functional.batch_norm(x_346, getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn2_running_var, getattr_getattr_l__mod___blocks___5_____3___bn2_weight, getattr_getattr_l__mod___blocks___5_____3___bn2_bias, True, 0.1, 1e-05);  x_346 = getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = getattr_getattr_l__mod___blocks___5_____3___bn2_weight = getattr_getattr_l__mod___blocks___5_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_348 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_drop(x_347);  x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_350 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_act(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_60 = x_350.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_61 = self.getattr_getattr_L__mod___blocks___5_____3___se_conv_reduce(x_se_60);  x_se_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_62 = self.getattr_getattr_L__mod___blocks___5_____3___se_act1(x_se_61);  x_se_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_63 = self.getattr_getattr_L__mod___blocks___5_____3___se_conv_expand(x_se_62);  x_se_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____3___se_gate = self.getattr_getattr_L__mod___blocks___5_____3___se_gate(x_se_63);  x_se_63 = None
    x_351 = x_350 * getattr_getattr_l__mod___blocks___5_____3___se_gate;  x_350 = getattr_getattr_l__mod___blocks___5_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_40 = torch.functional.split(x_351, [792, 792], 1);  x_351 = None
    getitem_104 = split_40[0]
    getitem_105 = split_40[1];  split_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pwl_0(getitem_104);  getitem_104 = None
    getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pwl_1(getitem_105);  getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    x_353 = torch.cat([getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0, getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], 1);  getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0 = getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1 = None
    
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
    x_354 = torch.nn.functional.batch_norm(x_353, getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn3_running_var, getattr_getattr_l__mod___blocks___5_____3___bn3_weight, getattr_getattr_l__mod___blocks___5_____3___bn3_bias, True, 0.1, 1e-05);  x_353 = getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = getattr_getattr_l__mod___blocks___5_____3___bn3_weight = getattr_getattr_l__mod___blocks___5_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_355 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_drop(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_357 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_act(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____3___drop_path = self.getattr_getattr_L__mod___blocks___5_____3___drop_path(x_357);  x_357 = None
    x_359 = getattr_getattr_l__mod___blocks___5_____3___drop_path + shortcut_18;  getattr_getattr_l__mod___blocks___5_____3___drop_path = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    x_360 = self.L__mod___conv_head(x_359);  x_359 = None
    
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
    x_361 = torch.nn.functional.batch_norm(x_360, l__mod___bn2_running_mean, l__mod___bn2_running_var, l__mod___bn2_weight, l__mod___bn2_bias, True, 0.1, 1e-05);  x_360 = l__mod___bn2_running_mean = l__mod___bn2_running_var = l__mod___bn2_weight = l__mod___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_362 = self.L__mod___bn2_drop(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_365 = self.L__mod___bn2_act(x_362);  x_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_366 = self.L__mod___global_pool_pool(x_365);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_368 = self.L__mod___global_pool_flatten(x_366);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    pred = self.L__mod___classifier(x_368);  x_368 = None
    return (pred,)
    