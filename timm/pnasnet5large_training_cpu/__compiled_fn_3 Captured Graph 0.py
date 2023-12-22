from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___conv_0_conv(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___conv_0_bn_num_batches_tracked = self.L__mod___conv_0_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_ = l__mod___conv_0_bn_num_batches_tracked.add_(1);  l__mod___conv_0_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___conv_0_bn_running_mean = self.L__mod___conv_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___conv_0_bn_running_var = self.L__mod___conv_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___conv_0_bn_weight = self.L__mod___conv_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___conv_0_bn_bias = self.L__mod___conv_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___conv_0_bn_running_mean, l__mod___conv_0_bn_running_var, l__mod___conv_0_bn_weight, l__mod___conv_0_bn_bias, True, 0.1, 0.001);  x = l__mod___conv_0_bn_running_mean = l__mod___conv_0_bn_running_var = l__mod___conv_0_bn_weight = l__mod___conv_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___conv_0_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_conv_0 = self.L__mod___conv_0_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_5 = self.L__mod___cell_stem_0_conv_1x1_act(x_conv_0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_6 = self.L__mod___cell_stem_0_conv_1x1_conv(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_right = self.L__mod___cell_stem_0_conv_1x1_bn(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_8 = self.L__mod___cell_stem_0_comb_iter_0_left_act_1(x_conv_0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_0_comb_iter_0_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_0_comb_iter_0_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_10 = torch.nn.functional.pad(x_8, (2, 2, 2, 2), value = 0);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_11 = torch.conv2d(x_10, l__mod___cell_stem_0_comb_iter_0_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 96);  x_10 = l__mod___cell_stem_0_comb_iter_0_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_13 = self.L__mod___cell_stem_0_comb_iter_0_left_separable_1_pointwise_conv2d(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_14 = self.L__mod___cell_stem_0_comb_iter_0_left_bn_sep_1(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_15 = self.L__mod___cell_stem_0_comb_iter_0_left_act_2(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_16 = self.L__mod___cell_stem_0_comb_iter_0_left_separable_2_depthwise_conv2d(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_18 = self.L__mod___cell_stem_0_comb_iter_0_left_separable_2_pointwise_conv2d(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left = self.L__mod___cell_stem_0_comb_iter_0_left_bn_sep_2(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_21 = torch.nn.functional.pad(x_conv_0, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d = torch.nn.functional.max_pool2d(x_21, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    l__mod___cell_stem_0_comb_iter_0_right_conv = self.L__mod___cell_stem_0_comb_iter_0_right_conv(max_pool2d);  max_pool2d = None
    x_comb_iter_0_right = self.L__mod___cell_stem_0_comb_iter_0_right_bn(l__mod___cell_stem_0_comb_iter_0_right_conv);  l__mod___cell_stem_0_comb_iter_0_right_conv = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right;  x_comb_iter_0_left = x_comb_iter_0_right = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_22 = self.L__mod___cell_stem_0_comb_iter_1_left_act_1(x_right)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_0_comb_iter_1_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_0_comb_iter_1_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_24 = torch.nn.functional.pad(x_22, (3, 3, 3, 3), value = 0);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_25 = torch.conv2d(x_24, l__mod___cell_stem_0_comb_iter_1_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 54);  x_24 = l__mod___cell_stem_0_comb_iter_1_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_27 = self.L__mod___cell_stem_0_comb_iter_1_left_separable_1_pointwise_conv2d(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_28 = self.L__mod___cell_stem_0_comb_iter_1_left_bn_sep_1(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_29 = self.L__mod___cell_stem_0_comb_iter_1_left_act_2(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_30 = self.L__mod___cell_stem_0_comb_iter_1_left_separable_2_depthwise_conv2d(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_32 = self.L__mod___cell_stem_0_comb_iter_1_left_separable_2_pointwise_conv2d(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left = self.L__mod___cell_stem_0_comb_iter_1_left_bn_sep_2(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_35 = torch.nn.functional.pad(x_right, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_1_right = torch.nn.functional.max_pool2d(x_35, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right;  x_comb_iter_1_left = x_comb_iter_1_right = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_36 = self.L__mod___cell_stem_0_comb_iter_2_left_act_1(x_right)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_0_comb_iter_2_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_0_comb_iter_2_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_38 = torch.nn.functional.pad(x_36, (2, 2, 2, 2), value = 0);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_39 = torch.conv2d(x_38, l__mod___cell_stem_0_comb_iter_2_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 54);  x_38 = l__mod___cell_stem_0_comb_iter_2_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_41 = self.L__mod___cell_stem_0_comb_iter_2_left_separable_1_pointwise_conv2d(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_42 = self.L__mod___cell_stem_0_comb_iter_2_left_bn_sep_1(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_43 = self.L__mod___cell_stem_0_comb_iter_2_left_act_2(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_44 = self.L__mod___cell_stem_0_comb_iter_2_left_separable_2_depthwise_conv2d(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_46 = self.L__mod___cell_stem_0_comb_iter_2_left_separable_2_pointwise_conv2d(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left = self.L__mod___cell_stem_0_comb_iter_2_left_bn_sep_2(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_48 = self.L__mod___cell_stem_0_comb_iter_2_right_act_1(x_right)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_0_comb_iter_2_right_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_0_comb_iter_2_right_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_50 = torch.nn.functional.pad(x_48, (1, 1, 1, 1), value = 0);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_51 = torch.conv2d(x_50, l__mod___cell_stem_0_comb_iter_2_right_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 54);  x_50 = l__mod___cell_stem_0_comb_iter_2_right_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_53 = self.L__mod___cell_stem_0_comb_iter_2_right_separable_1_pointwise_conv2d(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_54 = self.L__mod___cell_stem_0_comb_iter_2_right_bn_sep_1(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_55 = self.L__mod___cell_stem_0_comb_iter_2_right_act_2(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_56 = self.L__mod___cell_stem_0_comb_iter_2_right_separable_2_depthwise_conv2d(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_58 = self.L__mod___cell_stem_0_comb_iter_2_right_separable_2_pointwise_conv2d(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right = self.L__mod___cell_stem_0_comb_iter_2_right_bn_sep_2(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right;  x_comb_iter_2_left = x_comb_iter_2_right = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_60 = self.L__mod___cell_stem_0_comb_iter_3_left_act_1(x_comb_iter_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_61 = self.L__mod___cell_stem_0_comb_iter_3_left_separable_1_depthwise_conv2d(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_63 = self.L__mod___cell_stem_0_comb_iter_3_left_separable_1_pointwise_conv2d(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_64 = self.L__mod___cell_stem_0_comb_iter_3_left_bn_sep_1(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_65 = self.L__mod___cell_stem_0_comb_iter_3_left_act_2(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_66 = self.L__mod___cell_stem_0_comb_iter_3_left_separable_2_depthwise_conv2d(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_68 = self.L__mod___cell_stem_0_comb_iter_3_left_separable_2_pointwise_conv2d(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left = self.L__mod___cell_stem_0_comb_iter_3_left_bn_sep_2(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_71 = torch.nn.functional.pad(x_right, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_3_right = torch.nn.functional.max_pool2d(x_71, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right;  x_comb_iter_3_left = x_comb_iter_3_right = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_72 = self.L__mod___cell_stem_0_comb_iter_4_left_act_1(x_conv_0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_0_comb_iter_4_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_0_comb_iter_4_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_74 = torch.nn.functional.pad(x_72, (1, 1, 1, 1), value = 0);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_75 = torch.conv2d(x_74, l__mod___cell_stem_0_comb_iter_4_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 96);  x_74 = l__mod___cell_stem_0_comb_iter_4_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_77 = self.L__mod___cell_stem_0_comb_iter_4_left_separable_1_pointwise_conv2d(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_78 = self.L__mod___cell_stem_0_comb_iter_4_left_bn_sep_1(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_79 = self.L__mod___cell_stem_0_comb_iter_4_left_act_2(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_80 = self.L__mod___cell_stem_0_comb_iter_4_left_separable_2_depthwise_conv2d(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_82 = self.L__mod___cell_stem_0_comb_iter_4_left_separable_2_pointwise_conv2d(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left = self.L__mod___cell_stem_0_comb_iter_4_left_bn_sep_2(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_84 = self.L__mod___cell_stem_0_comb_iter_4_right_act(x_right);  x_right = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_0_comb_iter_4_right_conv_weight = self.L__mod___cell_stem_0_comb_iter_4_right_conv_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_86 = torch.nn.functional.pad(x_84, (0, 0, 0, 0), value = 0);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_87 = torch.conv2d(x_86, l__mod___cell_stem_0_comb_iter_4_right_conv_weight, None, (2, 2), (0, 0), (1, 1), 1);  x_86 = l__mod___cell_stem_0_comb_iter_4_right_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right = self.L__mod___cell_stem_0_comb_iter_4_right_bn(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right;  x_comb_iter_4_left = x_comb_iter_4_right = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_stem_0 = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1);  x_comb_iter_0 = x_comb_iter_1 = x_comb_iter_2 = x_comb_iter_3 = x_comb_iter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:95, code: x = self.act(x)
    x_89 = self.L__mod___cell_stem_1_conv_prev_1x1_act(x_conv_0);  x_conv_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    l__mod___cell_stem_1_conv_prev_1x1_path_1_avgpool = self.L__mod___cell_stem_1_conv_prev_1x1_path_1_avgpool(x_89)
    x_path1 = self.L__mod___cell_stem_1_conv_prev_1x1_path_1_conv(l__mod___cell_stem_1_conv_prev_1x1_path_1_avgpool);  l__mod___cell_stem_1_conv_prev_1x1_path_1_avgpool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    l__mod___cell_stem_1_conv_prev_1x1_path_2_pad = self.L__mod___cell_stem_1_conv_prev_1x1_path_2_pad(x_89);  x_89 = None
    l__mod___cell_stem_1_conv_prev_1x1_path_2_avgpool = self.L__mod___cell_stem_1_conv_prev_1x1_path_2_avgpool(l__mod___cell_stem_1_conv_prev_1x1_path_2_pad);  l__mod___cell_stem_1_conv_prev_1x1_path_2_pad = None
    x_path2 = self.L__mod___cell_stem_1_conv_prev_1x1_path_2_conv(l__mod___cell_stem_1_conv_prev_1x1_path_2_avgpool);  l__mod___cell_stem_1_conv_prev_1x1_path_2_avgpool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    cat_1 = torch.cat([x_path1, x_path2], 1);  x_path1 = x_path2 = None
    x_left = self.L__mod___cell_stem_1_conv_prev_1x1_final_path_bn(cat_1);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_90 = self.L__mod___cell_stem_1_conv_1x1_act(x_stem_0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_91 = self.L__mod___cell_stem_1_conv_1x1_conv(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_right_1 = self.L__mod___cell_stem_1_conv_1x1_bn(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_93 = self.L__mod___cell_stem_1_comb_iter_0_left_act_1(x_left)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_1_comb_iter_0_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_1_comb_iter_0_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_95 = torch.nn.functional.pad(x_93, (2, 2, 2, 2), value = 0);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_96 = torch.conv2d(x_95, l__mod___cell_stem_1_comb_iter_0_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 108);  x_95 = l__mod___cell_stem_1_comb_iter_0_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_98 = self.L__mod___cell_stem_1_comb_iter_0_left_separable_1_pointwise_conv2d(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_99 = self.L__mod___cell_stem_1_comb_iter_0_left_bn_sep_1(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_100 = self.L__mod___cell_stem_1_comb_iter_0_left_act_2(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_101 = self.L__mod___cell_stem_1_comb_iter_0_left_separable_2_depthwise_conv2d(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_103 = self.L__mod___cell_stem_1_comb_iter_0_left_separable_2_pointwise_conv2d(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_1 = self.L__mod___cell_stem_1_comb_iter_0_left_bn_sep_2(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_106 = torch.nn.functional.pad(x_left, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_0_right_1 = torch.nn.functional.max_pool2d(x_106, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_5 = x_comb_iter_0_left_1 + x_comb_iter_0_right_1;  x_comb_iter_0_left_1 = x_comb_iter_0_right_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_107 = self.L__mod___cell_stem_1_comb_iter_1_left_act_1(x_right_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_1_comb_iter_1_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_1_comb_iter_1_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_109 = torch.nn.functional.pad(x_107, (3, 3, 3, 3), value = 0);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_110 = torch.conv2d(x_109, l__mod___cell_stem_1_comb_iter_1_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 108);  x_109 = l__mod___cell_stem_1_comb_iter_1_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_112 = self.L__mod___cell_stem_1_comb_iter_1_left_separable_1_pointwise_conv2d(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_113 = self.L__mod___cell_stem_1_comb_iter_1_left_bn_sep_1(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_114 = self.L__mod___cell_stem_1_comb_iter_1_left_act_2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_115 = self.L__mod___cell_stem_1_comb_iter_1_left_separable_2_depthwise_conv2d(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_117 = self.L__mod___cell_stem_1_comb_iter_1_left_separable_2_pointwise_conv2d(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_1 = self.L__mod___cell_stem_1_comb_iter_1_left_bn_sep_2(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_120 = torch.nn.functional.pad(x_right_1, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_1_right_1 = torch.nn.functional.max_pool2d(x_120, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_6 = x_comb_iter_1_left_1 + x_comb_iter_1_right_1;  x_comb_iter_1_left_1 = x_comb_iter_1_right_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_121 = self.L__mod___cell_stem_1_comb_iter_2_left_act_1(x_right_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_1_comb_iter_2_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_1_comb_iter_2_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_123 = torch.nn.functional.pad(x_121, (2, 2, 2, 2), value = 0);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_124 = torch.conv2d(x_123, l__mod___cell_stem_1_comb_iter_2_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 108);  x_123 = l__mod___cell_stem_1_comb_iter_2_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_126 = self.L__mod___cell_stem_1_comb_iter_2_left_separable_1_pointwise_conv2d(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_127 = self.L__mod___cell_stem_1_comb_iter_2_left_bn_sep_1(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_128 = self.L__mod___cell_stem_1_comb_iter_2_left_act_2(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_129 = self.L__mod___cell_stem_1_comb_iter_2_left_separable_2_depthwise_conv2d(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_131 = self.L__mod___cell_stem_1_comb_iter_2_left_separable_2_pointwise_conv2d(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_1 = self.L__mod___cell_stem_1_comb_iter_2_left_bn_sep_2(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_133 = self.L__mod___cell_stem_1_comb_iter_2_right_act_1(x_right_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_1_comb_iter_2_right_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_1_comb_iter_2_right_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_135 = torch.nn.functional.pad(x_133, (1, 1, 1, 1), value = 0);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_136 = torch.conv2d(x_135, l__mod___cell_stem_1_comb_iter_2_right_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 108);  x_135 = l__mod___cell_stem_1_comb_iter_2_right_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_138 = self.L__mod___cell_stem_1_comb_iter_2_right_separable_1_pointwise_conv2d(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_139 = self.L__mod___cell_stem_1_comb_iter_2_right_bn_sep_1(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_140 = self.L__mod___cell_stem_1_comb_iter_2_right_act_2(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_141 = self.L__mod___cell_stem_1_comb_iter_2_right_separable_2_depthwise_conv2d(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_143 = self.L__mod___cell_stem_1_comb_iter_2_right_separable_2_pointwise_conv2d(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_1 = self.L__mod___cell_stem_1_comb_iter_2_right_bn_sep_2(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_7 = x_comb_iter_2_left_1 + x_comb_iter_2_right_1;  x_comb_iter_2_left_1 = x_comb_iter_2_right_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_145 = self.L__mod___cell_stem_1_comb_iter_3_left_act_1(x_comb_iter_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_146 = self.L__mod___cell_stem_1_comb_iter_3_left_separable_1_depthwise_conv2d(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_148 = self.L__mod___cell_stem_1_comb_iter_3_left_separable_1_pointwise_conv2d(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_149 = self.L__mod___cell_stem_1_comb_iter_3_left_bn_sep_1(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_150 = self.L__mod___cell_stem_1_comb_iter_3_left_act_2(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_151 = self.L__mod___cell_stem_1_comb_iter_3_left_separable_2_depthwise_conv2d(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_153 = self.L__mod___cell_stem_1_comb_iter_3_left_separable_2_pointwise_conv2d(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_1 = self.L__mod___cell_stem_1_comb_iter_3_left_bn_sep_2(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_156 = torch.nn.functional.pad(x_right_1, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_3_right_1 = torch.nn.functional.max_pool2d(x_156, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_8 = x_comb_iter_3_left_1 + x_comb_iter_3_right_1;  x_comb_iter_3_left_1 = x_comb_iter_3_right_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_157 = self.L__mod___cell_stem_1_comb_iter_4_left_act_1(x_left);  x_left = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_1_comb_iter_4_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_stem_1_comb_iter_4_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_159 = torch.nn.functional.pad(x_157, (1, 1, 1, 1), value = 0);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_160 = torch.conv2d(x_159, l__mod___cell_stem_1_comb_iter_4_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 108);  x_159 = l__mod___cell_stem_1_comb_iter_4_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_162 = self.L__mod___cell_stem_1_comb_iter_4_left_separable_1_pointwise_conv2d(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_163 = self.L__mod___cell_stem_1_comb_iter_4_left_bn_sep_1(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_164 = self.L__mod___cell_stem_1_comb_iter_4_left_act_2(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_165 = self.L__mod___cell_stem_1_comb_iter_4_left_separable_2_depthwise_conv2d(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_167 = self.L__mod___cell_stem_1_comb_iter_4_left_separable_2_pointwise_conv2d(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_1 = self.L__mod___cell_stem_1_comb_iter_4_left_bn_sep_2(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_169 = self.L__mod___cell_stem_1_comb_iter_4_right_act(x_right_1);  x_right_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_stem_1_comb_iter_4_right_conv_weight = self.L__mod___cell_stem_1_comb_iter_4_right_conv_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_171 = torch.nn.functional.pad(x_169, (0, 0, 0, 0), value = 0);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_172 = torch.conv2d(x_171, l__mod___cell_stem_1_comb_iter_4_right_conv_weight, None, (2, 2), (0, 0), (1, 1), 1);  x_171 = l__mod___cell_stem_1_comb_iter_4_right_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_1 = self.L__mod___cell_stem_1_comb_iter_4_right_bn(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_9 = x_comb_iter_4_left_1 + x_comb_iter_4_right_1;  x_comb_iter_4_left_1 = x_comb_iter_4_right_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_stem_1 = torch.cat([x_comb_iter_5, x_comb_iter_6, x_comb_iter_7, x_comb_iter_8, x_comb_iter_9], 1);  x_comb_iter_5 = x_comb_iter_6 = x_comb_iter_7 = x_comb_iter_8 = x_comb_iter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:95, code: x = self.act(x)
    x_174 = self.L__mod___cell_0_conv_prev_1x1_act(x_stem_0);  x_stem_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    l__mod___cell_0_conv_prev_1x1_path_1_avgpool = self.L__mod___cell_0_conv_prev_1x1_path_1_avgpool(x_174)
    x_path1_1 = self.L__mod___cell_0_conv_prev_1x1_path_1_conv(l__mod___cell_0_conv_prev_1x1_path_1_avgpool);  l__mod___cell_0_conv_prev_1x1_path_1_avgpool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    l__mod___cell_0_conv_prev_1x1_path_2_pad = self.L__mod___cell_0_conv_prev_1x1_path_2_pad(x_174);  x_174 = None
    l__mod___cell_0_conv_prev_1x1_path_2_avgpool = self.L__mod___cell_0_conv_prev_1x1_path_2_avgpool(l__mod___cell_0_conv_prev_1x1_path_2_pad);  l__mod___cell_0_conv_prev_1x1_path_2_pad = None
    x_path2_1 = self.L__mod___cell_0_conv_prev_1x1_path_2_conv(l__mod___cell_0_conv_prev_1x1_path_2_avgpool);  l__mod___cell_0_conv_prev_1x1_path_2_avgpool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    cat_3 = torch.cat([x_path1_1, x_path2_1], 1);  x_path1_1 = x_path2_1 = None
    x_left_1 = self.L__mod___cell_0_conv_prev_1x1_final_path_bn(cat_3);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_175 = self.L__mod___cell_0_conv_1x1_act(x_stem_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_176 = self.L__mod___cell_0_conv_1x1_conv(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_2 = self.L__mod___cell_0_conv_1x1_bn(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_178 = self.L__mod___cell_0_comb_iter_0_left_act_1(x_left_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_179 = self.L__mod___cell_0_comb_iter_0_left_separable_1_depthwise_conv2d(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_181 = self.L__mod___cell_0_comb_iter_0_left_separable_1_pointwise_conv2d(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_182 = self.L__mod___cell_0_comb_iter_0_left_bn_sep_1(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_183 = self.L__mod___cell_0_comb_iter_0_left_act_2(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_184 = self.L__mod___cell_0_comb_iter_0_left_separable_2_depthwise_conv2d(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_186 = self.L__mod___cell_0_comb_iter_0_left_separable_2_pointwise_conv2d(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_2 = self.L__mod___cell_0_comb_iter_0_left_bn_sep_2(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_2 = self.L__mod___cell_0_comb_iter_0_right(x_left_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_10 = x_comb_iter_0_left_2 + x_comb_iter_0_right_2;  x_comb_iter_0_left_2 = x_comb_iter_0_right_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_188 = self.L__mod___cell_0_comb_iter_1_left_act_1(x_comb_iter_4_right_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_189 = self.L__mod___cell_0_comb_iter_1_left_separable_1_depthwise_conv2d(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_191 = self.L__mod___cell_0_comb_iter_1_left_separable_1_pointwise_conv2d(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_192 = self.L__mod___cell_0_comb_iter_1_left_bn_sep_1(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_193 = self.L__mod___cell_0_comb_iter_1_left_act_2(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_194 = self.L__mod___cell_0_comb_iter_1_left_separable_2_depthwise_conv2d(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_196 = self.L__mod___cell_0_comb_iter_1_left_separable_2_pointwise_conv2d(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_2 = self.L__mod___cell_0_comb_iter_1_left_bn_sep_2(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_2 = self.L__mod___cell_0_comb_iter_1_right(x_comb_iter_4_right_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_11 = x_comb_iter_1_left_2 + x_comb_iter_1_right_2;  x_comb_iter_1_left_2 = x_comb_iter_1_right_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_198 = self.L__mod___cell_0_comb_iter_2_left_act_1(x_comb_iter_4_right_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_199 = self.L__mod___cell_0_comb_iter_2_left_separable_1_depthwise_conv2d(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_201 = self.L__mod___cell_0_comb_iter_2_left_separable_1_pointwise_conv2d(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_202 = self.L__mod___cell_0_comb_iter_2_left_bn_sep_1(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_203 = self.L__mod___cell_0_comb_iter_2_left_act_2(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_204 = self.L__mod___cell_0_comb_iter_2_left_separable_2_depthwise_conv2d(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_206 = self.L__mod___cell_0_comb_iter_2_left_separable_2_pointwise_conv2d(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_2 = self.L__mod___cell_0_comb_iter_2_left_bn_sep_2(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_208 = self.L__mod___cell_0_comb_iter_2_right_act_1(x_comb_iter_4_right_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_209 = self.L__mod___cell_0_comb_iter_2_right_separable_1_depthwise_conv2d(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_211 = self.L__mod___cell_0_comb_iter_2_right_separable_1_pointwise_conv2d(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_212 = self.L__mod___cell_0_comb_iter_2_right_bn_sep_1(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_213 = self.L__mod___cell_0_comb_iter_2_right_act_2(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_214 = self.L__mod___cell_0_comb_iter_2_right_separable_2_depthwise_conv2d(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_216 = self.L__mod___cell_0_comb_iter_2_right_separable_2_pointwise_conv2d(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_2 = self.L__mod___cell_0_comb_iter_2_right_bn_sep_2(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_12 = x_comb_iter_2_left_2 + x_comb_iter_2_right_2;  x_comb_iter_2_left_2 = x_comb_iter_2_right_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_218 = self.L__mod___cell_0_comb_iter_3_left_act_1(x_comb_iter_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_219 = self.L__mod___cell_0_comb_iter_3_left_separable_1_depthwise_conv2d(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_221 = self.L__mod___cell_0_comb_iter_3_left_separable_1_pointwise_conv2d(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_222 = self.L__mod___cell_0_comb_iter_3_left_bn_sep_1(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_223 = self.L__mod___cell_0_comb_iter_3_left_act_2(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_224 = self.L__mod___cell_0_comb_iter_3_left_separable_2_depthwise_conv2d(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_226 = self.L__mod___cell_0_comb_iter_3_left_separable_2_pointwise_conv2d(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_2 = self.L__mod___cell_0_comb_iter_3_left_bn_sep_2(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_2 = self.L__mod___cell_0_comb_iter_3_right(x_comb_iter_4_right_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_13 = x_comb_iter_3_left_2 + x_comb_iter_3_right_2;  x_comb_iter_3_left_2 = x_comb_iter_3_right_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_228 = self.L__mod___cell_0_comb_iter_4_left_act_1(x_left_1);  x_left_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_229 = self.L__mod___cell_0_comb_iter_4_left_separable_1_depthwise_conv2d(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_231 = self.L__mod___cell_0_comb_iter_4_left_separable_1_pointwise_conv2d(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_232 = self.L__mod___cell_0_comb_iter_4_left_bn_sep_1(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_233 = self.L__mod___cell_0_comb_iter_4_left_act_2(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_234 = self.L__mod___cell_0_comb_iter_4_left_separable_2_depthwise_conv2d(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_236 = self.L__mod___cell_0_comb_iter_4_left_separable_2_pointwise_conv2d(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_2 = self.L__mod___cell_0_comb_iter_4_left_bn_sep_2(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_14 = x_comb_iter_4_left_2 + x_comb_iter_4_right_2;  x_comb_iter_4_left_2 = x_comb_iter_4_right_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_0 = torch.cat([x_comb_iter_10, x_comb_iter_11, x_comb_iter_12, x_comb_iter_13, x_comb_iter_14], 1);  x_comb_iter_10 = x_comb_iter_11 = x_comb_iter_12 = x_comb_iter_13 = x_comb_iter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_238 = self.L__mod___cell_1_conv_prev_1x1_act(x_stem_1);  x_stem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_239 = self.L__mod___cell_1_conv_prev_1x1_conv(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_2 = self.L__mod___cell_1_conv_prev_1x1_bn(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_241 = self.L__mod___cell_1_conv_1x1_act(x_cell_0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_242 = self.L__mod___cell_1_conv_1x1_conv(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_3 = self.L__mod___cell_1_conv_1x1_bn(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_244 = self.L__mod___cell_1_comb_iter_0_left_act_1(x_left_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_245 = self.L__mod___cell_1_comb_iter_0_left_separable_1_depthwise_conv2d(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_247 = self.L__mod___cell_1_comb_iter_0_left_separable_1_pointwise_conv2d(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_248 = self.L__mod___cell_1_comb_iter_0_left_bn_sep_1(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_249 = self.L__mod___cell_1_comb_iter_0_left_act_2(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_250 = self.L__mod___cell_1_comb_iter_0_left_separable_2_depthwise_conv2d(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_252 = self.L__mod___cell_1_comb_iter_0_left_separable_2_pointwise_conv2d(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_3 = self.L__mod___cell_1_comb_iter_0_left_bn_sep_2(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_3 = self.L__mod___cell_1_comb_iter_0_right(x_left_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_15 = x_comb_iter_0_left_3 + x_comb_iter_0_right_3;  x_comb_iter_0_left_3 = x_comb_iter_0_right_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_254 = self.L__mod___cell_1_comb_iter_1_left_act_1(x_comb_iter_4_right_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_255 = self.L__mod___cell_1_comb_iter_1_left_separable_1_depthwise_conv2d(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_257 = self.L__mod___cell_1_comb_iter_1_left_separable_1_pointwise_conv2d(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_258 = self.L__mod___cell_1_comb_iter_1_left_bn_sep_1(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_259 = self.L__mod___cell_1_comb_iter_1_left_act_2(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_260 = self.L__mod___cell_1_comb_iter_1_left_separable_2_depthwise_conv2d(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_262 = self.L__mod___cell_1_comb_iter_1_left_separable_2_pointwise_conv2d(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_3 = self.L__mod___cell_1_comb_iter_1_left_bn_sep_2(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_3 = self.L__mod___cell_1_comb_iter_1_right(x_comb_iter_4_right_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_16 = x_comb_iter_1_left_3 + x_comb_iter_1_right_3;  x_comb_iter_1_left_3 = x_comb_iter_1_right_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_264 = self.L__mod___cell_1_comb_iter_2_left_act_1(x_comb_iter_4_right_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_265 = self.L__mod___cell_1_comb_iter_2_left_separable_1_depthwise_conv2d(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_267 = self.L__mod___cell_1_comb_iter_2_left_separable_1_pointwise_conv2d(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_268 = self.L__mod___cell_1_comb_iter_2_left_bn_sep_1(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_269 = self.L__mod___cell_1_comb_iter_2_left_act_2(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_270 = self.L__mod___cell_1_comb_iter_2_left_separable_2_depthwise_conv2d(x_269);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_272 = self.L__mod___cell_1_comb_iter_2_left_separable_2_pointwise_conv2d(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_3 = self.L__mod___cell_1_comb_iter_2_left_bn_sep_2(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_274 = self.L__mod___cell_1_comb_iter_2_right_act_1(x_comb_iter_4_right_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_275 = self.L__mod___cell_1_comb_iter_2_right_separable_1_depthwise_conv2d(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_277 = self.L__mod___cell_1_comb_iter_2_right_separable_1_pointwise_conv2d(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_278 = self.L__mod___cell_1_comb_iter_2_right_bn_sep_1(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_279 = self.L__mod___cell_1_comb_iter_2_right_act_2(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_280 = self.L__mod___cell_1_comb_iter_2_right_separable_2_depthwise_conv2d(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_282 = self.L__mod___cell_1_comb_iter_2_right_separable_2_pointwise_conv2d(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_3 = self.L__mod___cell_1_comb_iter_2_right_bn_sep_2(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_17 = x_comb_iter_2_left_3 + x_comb_iter_2_right_3;  x_comb_iter_2_left_3 = x_comb_iter_2_right_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_284 = self.L__mod___cell_1_comb_iter_3_left_act_1(x_comb_iter_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_285 = self.L__mod___cell_1_comb_iter_3_left_separable_1_depthwise_conv2d(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_287 = self.L__mod___cell_1_comb_iter_3_left_separable_1_pointwise_conv2d(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_288 = self.L__mod___cell_1_comb_iter_3_left_bn_sep_1(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_289 = self.L__mod___cell_1_comb_iter_3_left_act_2(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_290 = self.L__mod___cell_1_comb_iter_3_left_separable_2_depthwise_conv2d(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_292 = self.L__mod___cell_1_comb_iter_3_left_separable_2_pointwise_conv2d(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_3 = self.L__mod___cell_1_comb_iter_3_left_bn_sep_2(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_3 = self.L__mod___cell_1_comb_iter_3_right(x_comb_iter_4_right_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_18 = x_comb_iter_3_left_3 + x_comb_iter_3_right_3;  x_comb_iter_3_left_3 = x_comb_iter_3_right_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_294 = self.L__mod___cell_1_comb_iter_4_left_act_1(x_left_2);  x_left_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_295 = self.L__mod___cell_1_comb_iter_4_left_separable_1_depthwise_conv2d(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_297 = self.L__mod___cell_1_comb_iter_4_left_separable_1_pointwise_conv2d(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_298 = self.L__mod___cell_1_comb_iter_4_left_bn_sep_1(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_299 = self.L__mod___cell_1_comb_iter_4_left_act_2(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_300 = self.L__mod___cell_1_comb_iter_4_left_separable_2_depthwise_conv2d(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_302 = self.L__mod___cell_1_comb_iter_4_left_separable_2_pointwise_conv2d(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_3 = self.L__mod___cell_1_comb_iter_4_left_bn_sep_2(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_19 = x_comb_iter_4_left_3 + x_comb_iter_4_right_3;  x_comb_iter_4_left_3 = x_comb_iter_4_right_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_1 = torch.cat([x_comb_iter_15, x_comb_iter_16, x_comb_iter_17, x_comb_iter_18, x_comb_iter_19], 1);  x_comb_iter_15 = x_comb_iter_16 = x_comb_iter_17 = x_comb_iter_18 = x_comb_iter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_304 = self.L__mod___cell_2_conv_prev_1x1_act(x_cell_0);  x_cell_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_305 = self.L__mod___cell_2_conv_prev_1x1_conv(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_3 = self.L__mod___cell_2_conv_prev_1x1_bn(x_305);  x_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_307 = self.L__mod___cell_2_conv_1x1_act(x_cell_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_308 = self.L__mod___cell_2_conv_1x1_conv(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_4 = self.L__mod___cell_2_conv_1x1_bn(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_310 = self.L__mod___cell_2_comb_iter_0_left_act_1(x_left_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_311 = self.L__mod___cell_2_comb_iter_0_left_separable_1_depthwise_conv2d(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_313 = self.L__mod___cell_2_comb_iter_0_left_separable_1_pointwise_conv2d(x_311);  x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_314 = self.L__mod___cell_2_comb_iter_0_left_bn_sep_1(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_315 = self.L__mod___cell_2_comb_iter_0_left_act_2(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_316 = self.L__mod___cell_2_comb_iter_0_left_separable_2_depthwise_conv2d(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_318 = self.L__mod___cell_2_comb_iter_0_left_separable_2_pointwise_conv2d(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_4 = self.L__mod___cell_2_comb_iter_0_left_bn_sep_2(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_4 = self.L__mod___cell_2_comb_iter_0_right(x_left_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_20 = x_comb_iter_0_left_4 + x_comb_iter_0_right_4;  x_comb_iter_0_left_4 = x_comb_iter_0_right_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_320 = self.L__mod___cell_2_comb_iter_1_left_act_1(x_comb_iter_4_right_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_321 = self.L__mod___cell_2_comb_iter_1_left_separable_1_depthwise_conv2d(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_323 = self.L__mod___cell_2_comb_iter_1_left_separable_1_pointwise_conv2d(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_324 = self.L__mod___cell_2_comb_iter_1_left_bn_sep_1(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_325 = self.L__mod___cell_2_comb_iter_1_left_act_2(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_326 = self.L__mod___cell_2_comb_iter_1_left_separable_2_depthwise_conv2d(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_328 = self.L__mod___cell_2_comb_iter_1_left_separable_2_pointwise_conv2d(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_4 = self.L__mod___cell_2_comb_iter_1_left_bn_sep_2(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_4 = self.L__mod___cell_2_comb_iter_1_right(x_comb_iter_4_right_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_21 = x_comb_iter_1_left_4 + x_comb_iter_1_right_4;  x_comb_iter_1_left_4 = x_comb_iter_1_right_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_330 = self.L__mod___cell_2_comb_iter_2_left_act_1(x_comb_iter_4_right_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_331 = self.L__mod___cell_2_comb_iter_2_left_separable_1_depthwise_conv2d(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_333 = self.L__mod___cell_2_comb_iter_2_left_separable_1_pointwise_conv2d(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_334 = self.L__mod___cell_2_comb_iter_2_left_bn_sep_1(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_335 = self.L__mod___cell_2_comb_iter_2_left_act_2(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_336 = self.L__mod___cell_2_comb_iter_2_left_separable_2_depthwise_conv2d(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_338 = self.L__mod___cell_2_comb_iter_2_left_separable_2_pointwise_conv2d(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_4 = self.L__mod___cell_2_comb_iter_2_left_bn_sep_2(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_340 = self.L__mod___cell_2_comb_iter_2_right_act_1(x_comb_iter_4_right_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_341 = self.L__mod___cell_2_comb_iter_2_right_separable_1_depthwise_conv2d(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_343 = self.L__mod___cell_2_comb_iter_2_right_separable_1_pointwise_conv2d(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_344 = self.L__mod___cell_2_comb_iter_2_right_bn_sep_1(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_345 = self.L__mod___cell_2_comb_iter_2_right_act_2(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_346 = self.L__mod___cell_2_comb_iter_2_right_separable_2_depthwise_conv2d(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_348 = self.L__mod___cell_2_comb_iter_2_right_separable_2_pointwise_conv2d(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_4 = self.L__mod___cell_2_comb_iter_2_right_bn_sep_2(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_22 = x_comb_iter_2_left_4 + x_comb_iter_2_right_4;  x_comb_iter_2_left_4 = x_comb_iter_2_right_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_350 = self.L__mod___cell_2_comb_iter_3_left_act_1(x_comb_iter_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_351 = self.L__mod___cell_2_comb_iter_3_left_separable_1_depthwise_conv2d(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_353 = self.L__mod___cell_2_comb_iter_3_left_separable_1_pointwise_conv2d(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_354 = self.L__mod___cell_2_comb_iter_3_left_bn_sep_1(x_353);  x_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_355 = self.L__mod___cell_2_comb_iter_3_left_act_2(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_356 = self.L__mod___cell_2_comb_iter_3_left_separable_2_depthwise_conv2d(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_358 = self.L__mod___cell_2_comb_iter_3_left_separable_2_pointwise_conv2d(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_4 = self.L__mod___cell_2_comb_iter_3_left_bn_sep_2(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_4 = self.L__mod___cell_2_comb_iter_3_right(x_comb_iter_4_right_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_23 = x_comb_iter_3_left_4 + x_comb_iter_3_right_4;  x_comb_iter_3_left_4 = x_comb_iter_3_right_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_360 = self.L__mod___cell_2_comb_iter_4_left_act_1(x_left_3);  x_left_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_361 = self.L__mod___cell_2_comb_iter_4_left_separable_1_depthwise_conv2d(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_363 = self.L__mod___cell_2_comb_iter_4_left_separable_1_pointwise_conv2d(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_364 = self.L__mod___cell_2_comb_iter_4_left_bn_sep_1(x_363);  x_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_365 = self.L__mod___cell_2_comb_iter_4_left_act_2(x_364);  x_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_366 = self.L__mod___cell_2_comb_iter_4_left_separable_2_depthwise_conv2d(x_365);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_368 = self.L__mod___cell_2_comb_iter_4_left_separable_2_pointwise_conv2d(x_366);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_4 = self.L__mod___cell_2_comb_iter_4_left_bn_sep_2(x_368);  x_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_24 = x_comb_iter_4_left_4 + x_comb_iter_4_right_4;  x_comb_iter_4_left_4 = x_comb_iter_4_right_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_2 = torch.cat([x_comb_iter_20, x_comb_iter_21, x_comb_iter_22, x_comb_iter_23, x_comb_iter_24], 1);  x_comb_iter_20 = x_comb_iter_21 = x_comb_iter_22 = x_comb_iter_23 = x_comb_iter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_370 = self.L__mod___cell_3_conv_prev_1x1_act(x_cell_1);  x_cell_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_371 = self.L__mod___cell_3_conv_prev_1x1_conv(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_4 = self.L__mod___cell_3_conv_prev_1x1_bn(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_373 = self.L__mod___cell_3_conv_1x1_act(x_cell_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_374 = self.L__mod___cell_3_conv_1x1_conv(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_5 = self.L__mod___cell_3_conv_1x1_bn(x_374);  x_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_376 = self.L__mod___cell_3_comb_iter_0_left_act_1(x_left_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_377 = self.L__mod___cell_3_comb_iter_0_left_separable_1_depthwise_conv2d(x_376);  x_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_379 = self.L__mod___cell_3_comb_iter_0_left_separable_1_pointwise_conv2d(x_377);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_380 = self.L__mod___cell_3_comb_iter_0_left_bn_sep_1(x_379);  x_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_381 = self.L__mod___cell_3_comb_iter_0_left_act_2(x_380);  x_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_382 = self.L__mod___cell_3_comb_iter_0_left_separable_2_depthwise_conv2d(x_381);  x_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_384 = self.L__mod___cell_3_comb_iter_0_left_separable_2_pointwise_conv2d(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_5 = self.L__mod___cell_3_comb_iter_0_left_bn_sep_2(x_384);  x_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_5 = self.L__mod___cell_3_comb_iter_0_right(x_left_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_25 = x_comb_iter_0_left_5 + x_comb_iter_0_right_5;  x_comb_iter_0_left_5 = x_comb_iter_0_right_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_386 = self.L__mod___cell_3_comb_iter_1_left_act_1(x_comb_iter_4_right_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_387 = self.L__mod___cell_3_comb_iter_1_left_separable_1_depthwise_conv2d(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_389 = self.L__mod___cell_3_comb_iter_1_left_separable_1_pointwise_conv2d(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_390 = self.L__mod___cell_3_comb_iter_1_left_bn_sep_1(x_389);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_391 = self.L__mod___cell_3_comb_iter_1_left_act_2(x_390);  x_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_392 = self.L__mod___cell_3_comb_iter_1_left_separable_2_depthwise_conv2d(x_391);  x_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_394 = self.L__mod___cell_3_comb_iter_1_left_separable_2_pointwise_conv2d(x_392);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_5 = self.L__mod___cell_3_comb_iter_1_left_bn_sep_2(x_394);  x_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_5 = self.L__mod___cell_3_comb_iter_1_right(x_comb_iter_4_right_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_26 = x_comb_iter_1_left_5 + x_comb_iter_1_right_5;  x_comb_iter_1_left_5 = x_comb_iter_1_right_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_396 = self.L__mod___cell_3_comb_iter_2_left_act_1(x_comb_iter_4_right_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_397 = self.L__mod___cell_3_comb_iter_2_left_separable_1_depthwise_conv2d(x_396);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_399 = self.L__mod___cell_3_comb_iter_2_left_separable_1_pointwise_conv2d(x_397);  x_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_400 = self.L__mod___cell_3_comb_iter_2_left_bn_sep_1(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_401 = self.L__mod___cell_3_comb_iter_2_left_act_2(x_400);  x_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_402 = self.L__mod___cell_3_comb_iter_2_left_separable_2_depthwise_conv2d(x_401);  x_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_404 = self.L__mod___cell_3_comb_iter_2_left_separable_2_pointwise_conv2d(x_402);  x_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_5 = self.L__mod___cell_3_comb_iter_2_left_bn_sep_2(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_406 = self.L__mod___cell_3_comb_iter_2_right_act_1(x_comb_iter_4_right_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_407 = self.L__mod___cell_3_comb_iter_2_right_separable_1_depthwise_conv2d(x_406);  x_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_409 = self.L__mod___cell_3_comb_iter_2_right_separable_1_pointwise_conv2d(x_407);  x_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_410 = self.L__mod___cell_3_comb_iter_2_right_bn_sep_1(x_409);  x_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_411 = self.L__mod___cell_3_comb_iter_2_right_act_2(x_410);  x_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_412 = self.L__mod___cell_3_comb_iter_2_right_separable_2_depthwise_conv2d(x_411);  x_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_414 = self.L__mod___cell_3_comb_iter_2_right_separable_2_pointwise_conv2d(x_412);  x_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_5 = self.L__mod___cell_3_comb_iter_2_right_bn_sep_2(x_414);  x_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_27 = x_comb_iter_2_left_5 + x_comb_iter_2_right_5;  x_comb_iter_2_left_5 = x_comb_iter_2_right_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_416 = self.L__mod___cell_3_comb_iter_3_left_act_1(x_comb_iter_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_417 = self.L__mod___cell_3_comb_iter_3_left_separable_1_depthwise_conv2d(x_416);  x_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_419 = self.L__mod___cell_3_comb_iter_3_left_separable_1_pointwise_conv2d(x_417);  x_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_420 = self.L__mod___cell_3_comb_iter_3_left_bn_sep_1(x_419);  x_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_421 = self.L__mod___cell_3_comb_iter_3_left_act_2(x_420);  x_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_422 = self.L__mod___cell_3_comb_iter_3_left_separable_2_depthwise_conv2d(x_421);  x_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_424 = self.L__mod___cell_3_comb_iter_3_left_separable_2_pointwise_conv2d(x_422);  x_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_5 = self.L__mod___cell_3_comb_iter_3_left_bn_sep_2(x_424);  x_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_5 = self.L__mod___cell_3_comb_iter_3_right(x_comb_iter_4_right_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_28 = x_comb_iter_3_left_5 + x_comb_iter_3_right_5;  x_comb_iter_3_left_5 = x_comb_iter_3_right_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_426 = self.L__mod___cell_3_comb_iter_4_left_act_1(x_left_4);  x_left_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_427 = self.L__mod___cell_3_comb_iter_4_left_separable_1_depthwise_conv2d(x_426);  x_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_429 = self.L__mod___cell_3_comb_iter_4_left_separable_1_pointwise_conv2d(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_430 = self.L__mod___cell_3_comb_iter_4_left_bn_sep_1(x_429);  x_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_431 = self.L__mod___cell_3_comb_iter_4_left_act_2(x_430);  x_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_432 = self.L__mod___cell_3_comb_iter_4_left_separable_2_depthwise_conv2d(x_431);  x_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_434 = self.L__mod___cell_3_comb_iter_4_left_separable_2_pointwise_conv2d(x_432);  x_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_5 = self.L__mod___cell_3_comb_iter_4_left_bn_sep_2(x_434);  x_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_29 = x_comb_iter_4_left_5 + x_comb_iter_4_right_5;  x_comb_iter_4_left_5 = x_comb_iter_4_right_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_3 = torch.cat([x_comb_iter_25, x_comb_iter_26, x_comb_iter_27, x_comb_iter_28, x_comb_iter_29], 1);  x_comb_iter_25 = x_comb_iter_26 = x_comb_iter_27 = x_comb_iter_28 = x_comb_iter_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_436 = self.L__mod___cell_4_conv_prev_1x1_act(x_cell_2);  x_cell_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_437 = self.L__mod___cell_4_conv_prev_1x1_conv(x_436);  x_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_5 = self.L__mod___cell_4_conv_prev_1x1_bn(x_437);  x_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_439 = self.L__mod___cell_4_conv_1x1_act(x_cell_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_440 = self.L__mod___cell_4_conv_1x1_conv(x_439);  x_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_right_6 = self.L__mod___cell_4_conv_1x1_bn(x_440);  x_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_442 = self.L__mod___cell_4_comb_iter_0_left_act_1(x_left_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_4_comb_iter_0_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_4_comb_iter_0_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_444 = torch.nn.functional.pad(x_442, (1, 2, 1, 2), value = 0);  x_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_445 = torch.conv2d(x_444, l__mod___cell_4_comb_iter_0_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 432);  x_444 = l__mod___cell_4_comb_iter_0_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_447 = self.L__mod___cell_4_comb_iter_0_left_separable_1_pointwise_conv2d(x_445);  x_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_448 = self.L__mod___cell_4_comb_iter_0_left_bn_sep_1(x_447);  x_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_449 = self.L__mod___cell_4_comb_iter_0_left_act_2(x_448);  x_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_450 = self.L__mod___cell_4_comb_iter_0_left_separable_2_depthwise_conv2d(x_449);  x_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_452 = self.L__mod___cell_4_comb_iter_0_left_separable_2_pointwise_conv2d(x_450);  x_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_6 = self.L__mod___cell_4_comb_iter_0_left_bn_sep_2(x_452);  x_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_455 = torch.nn.functional.pad(x_left_5, (0, 1, 0, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_0_right_6 = torch.nn.functional.max_pool2d(x_455, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_30 = x_comb_iter_0_left_6 + x_comb_iter_0_right_6;  x_comb_iter_0_left_6 = x_comb_iter_0_right_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_456 = self.L__mod___cell_4_comb_iter_1_left_act_1(x_right_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_4_comb_iter_1_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_4_comb_iter_1_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_458 = torch.nn.functional.pad(x_456, (2, 3, 2, 3), value = 0);  x_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_459 = torch.conv2d(x_458, l__mod___cell_4_comb_iter_1_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 432);  x_458 = l__mod___cell_4_comb_iter_1_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_461 = self.L__mod___cell_4_comb_iter_1_left_separable_1_pointwise_conv2d(x_459);  x_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_462 = self.L__mod___cell_4_comb_iter_1_left_bn_sep_1(x_461);  x_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_463 = self.L__mod___cell_4_comb_iter_1_left_act_2(x_462);  x_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_464 = self.L__mod___cell_4_comb_iter_1_left_separable_2_depthwise_conv2d(x_463);  x_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_466 = self.L__mod___cell_4_comb_iter_1_left_separable_2_pointwise_conv2d(x_464);  x_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_6 = self.L__mod___cell_4_comb_iter_1_left_bn_sep_2(x_466);  x_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_469 = torch.nn.functional.pad(x_right_6, (0, 1, 0, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_1_right_6 = torch.nn.functional.max_pool2d(x_469, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_31 = x_comb_iter_1_left_6 + x_comb_iter_1_right_6;  x_comb_iter_1_left_6 = x_comb_iter_1_right_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_470 = self.L__mod___cell_4_comb_iter_2_left_act_1(x_right_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_4_comb_iter_2_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_4_comb_iter_2_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_472 = torch.nn.functional.pad(x_470, (1, 2, 1, 2), value = 0);  x_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_473 = torch.conv2d(x_472, l__mod___cell_4_comb_iter_2_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 432);  x_472 = l__mod___cell_4_comb_iter_2_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_475 = self.L__mod___cell_4_comb_iter_2_left_separable_1_pointwise_conv2d(x_473);  x_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_476 = self.L__mod___cell_4_comb_iter_2_left_bn_sep_1(x_475);  x_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_477 = self.L__mod___cell_4_comb_iter_2_left_act_2(x_476);  x_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_478 = self.L__mod___cell_4_comb_iter_2_left_separable_2_depthwise_conv2d(x_477);  x_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_480 = self.L__mod___cell_4_comb_iter_2_left_separable_2_pointwise_conv2d(x_478);  x_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_6 = self.L__mod___cell_4_comb_iter_2_left_bn_sep_2(x_480);  x_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_482 = self.L__mod___cell_4_comb_iter_2_right_act_1(x_right_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_4_comb_iter_2_right_separable_1_depthwise_conv2d_weight = self.L__mod___cell_4_comb_iter_2_right_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_484 = torch.nn.functional.pad(x_482, (0, 1, 0, 1), value = 0);  x_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_485 = torch.conv2d(x_484, l__mod___cell_4_comb_iter_2_right_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 432);  x_484 = l__mod___cell_4_comb_iter_2_right_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_487 = self.L__mod___cell_4_comb_iter_2_right_separable_1_pointwise_conv2d(x_485);  x_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_488 = self.L__mod___cell_4_comb_iter_2_right_bn_sep_1(x_487);  x_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_489 = self.L__mod___cell_4_comb_iter_2_right_act_2(x_488);  x_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_490 = self.L__mod___cell_4_comb_iter_2_right_separable_2_depthwise_conv2d(x_489);  x_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_492 = self.L__mod___cell_4_comb_iter_2_right_separable_2_pointwise_conv2d(x_490);  x_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_6 = self.L__mod___cell_4_comb_iter_2_right_bn_sep_2(x_492);  x_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_32 = x_comb_iter_2_left_6 + x_comb_iter_2_right_6;  x_comb_iter_2_left_6 = x_comb_iter_2_right_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_494 = self.L__mod___cell_4_comb_iter_3_left_act_1(x_comb_iter_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_495 = self.L__mod___cell_4_comb_iter_3_left_separable_1_depthwise_conv2d(x_494);  x_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_497 = self.L__mod___cell_4_comb_iter_3_left_separable_1_pointwise_conv2d(x_495);  x_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_498 = self.L__mod___cell_4_comb_iter_3_left_bn_sep_1(x_497);  x_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_499 = self.L__mod___cell_4_comb_iter_3_left_act_2(x_498);  x_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_500 = self.L__mod___cell_4_comb_iter_3_left_separable_2_depthwise_conv2d(x_499);  x_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_502 = self.L__mod___cell_4_comb_iter_3_left_separable_2_pointwise_conv2d(x_500);  x_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_6 = self.L__mod___cell_4_comb_iter_3_left_bn_sep_2(x_502);  x_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_505 = torch.nn.functional.pad(x_right_6, (0, 1, 0, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_3_right_6 = torch.nn.functional.max_pool2d(x_505, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_33 = x_comb_iter_3_left_6 + x_comb_iter_3_right_6;  x_comb_iter_3_left_6 = x_comb_iter_3_right_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_506 = self.L__mod___cell_4_comb_iter_4_left_act_1(x_left_5);  x_left_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_4_comb_iter_4_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_4_comb_iter_4_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_508 = torch.nn.functional.pad(x_506, (0, 1, 0, 1), value = 0);  x_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_509 = torch.conv2d(x_508, l__mod___cell_4_comb_iter_4_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 432);  x_508 = l__mod___cell_4_comb_iter_4_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_511 = self.L__mod___cell_4_comb_iter_4_left_separable_1_pointwise_conv2d(x_509);  x_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_512 = self.L__mod___cell_4_comb_iter_4_left_bn_sep_1(x_511);  x_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_513 = self.L__mod___cell_4_comb_iter_4_left_act_2(x_512);  x_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_514 = self.L__mod___cell_4_comb_iter_4_left_separable_2_depthwise_conv2d(x_513);  x_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_516 = self.L__mod___cell_4_comb_iter_4_left_separable_2_pointwise_conv2d(x_514);  x_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_6 = self.L__mod___cell_4_comb_iter_4_left_bn_sep_2(x_516);  x_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_518 = self.L__mod___cell_4_comb_iter_4_right_act(x_right_6);  x_right_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_4_comb_iter_4_right_conv_weight = self.L__mod___cell_4_comb_iter_4_right_conv_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_520 = torch.nn.functional.pad(x_518, (0, 0, 0, 0), value = 0);  x_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_521 = torch.conv2d(x_520, l__mod___cell_4_comb_iter_4_right_conv_weight, None, (2, 2), (0, 0), (1, 1), 1);  x_520 = l__mod___cell_4_comb_iter_4_right_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_6 = self.L__mod___cell_4_comb_iter_4_right_bn(x_521);  x_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_34 = x_comb_iter_4_left_6 + x_comb_iter_4_right_6;  x_comb_iter_4_left_6 = x_comb_iter_4_right_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_4 = torch.cat([x_comb_iter_30, x_comb_iter_31, x_comb_iter_32, x_comb_iter_33, x_comb_iter_34], 1);  x_comb_iter_30 = x_comb_iter_31 = x_comb_iter_32 = x_comb_iter_33 = x_comb_iter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:95, code: x = self.act(x)
    x_523 = self.L__mod___cell_5_conv_prev_1x1_act(x_cell_3);  x_cell_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    l__mod___cell_5_conv_prev_1x1_path_1_avgpool = self.L__mod___cell_5_conv_prev_1x1_path_1_avgpool(x_523)
    x_path1_2 = self.L__mod___cell_5_conv_prev_1x1_path_1_conv(l__mod___cell_5_conv_prev_1x1_path_1_avgpool);  l__mod___cell_5_conv_prev_1x1_path_1_avgpool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    l__mod___cell_5_conv_prev_1x1_path_2_pad = self.L__mod___cell_5_conv_prev_1x1_path_2_pad(x_523);  x_523 = None
    l__mod___cell_5_conv_prev_1x1_path_2_avgpool = self.L__mod___cell_5_conv_prev_1x1_path_2_avgpool(l__mod___cell_5_conv_prev_1x1_path_2_pad);  l__mod___cell_5_conv_prev_1x1_path_2_pad = None
    x_path2_2 = self.L__mod___cell_5_conv_prev_1x1_path_2_conv(l__mod___cell_5_conv_prev_1x1_path_2_avgpool);  l__mod___cell_5_conv_prev_1x1_path_2_avgpool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    cat_9 = torch.cat([x_path1_2, x_path2_2], 1);  x_path1_2 = x_path2_2 = None
    x_left_6 = self.L__mod___cell_5_conv_prev_1x1_final_path_bn(cat_9);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_524 = self.L__mod___cell_5_conv_1x1_act(x_cell_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_525 = self.L__mod___cell_5_conv_1x1_conv(x_524);  x_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_7 = self.L__mod___cell_5_conv_1x1_bn(x_525);  x_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_527 = self.L__mod___cell_5_comb_iter_0_left_act_1(x_left_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_528 = self.L__mod___cell_5_comb_iter_0_left_separable_1_depthwise_conv2d(x_527);  x_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_530 = self.L__mod___cell_5_comb_iter_0_left_separable_1_pointwise_conv2d(x_528);  x_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_531 = self.L__mod___cell_5_comb_iter_0_left_bn_sep_1(x_530);  x_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_532 = self.L__mod___cell_5_comb_iter_0_left_act_2(x_531);  x_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_533 = self.L__mod___cell_5_comb_iter_0_left_separable_2_depthwise_conv2d(x_532);  x_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_535 = self.L__mod___cell_5_comb_iter_0_left_separable_2_pointwise_conv2d(x_533);  x_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_7 = self.L__mod___cell_5_comb_iter_0_left_bn_sep_2(x_535);  x_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_7 = self.L__mod___cell_5_comb_iter_0_right(x_left_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_35 = x_comb_iter_0_left_7 + x_comb_iter_0_right_7;  x_comb_iter_0_left_7 = x_comb_iter_0_right_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_537 = self.L__mod___cell_5_comb_iter_1_left_act_1(x_comb_iter_4_right_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_538 = self.L__mod___cell_5_comb_iter_1_left_separable_1_depthwise_conv2d(x_537);  x_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_540 = self.L__mod___cell_5_comb_iter_1_left_separable_1_pointwise_conv2d(x_538);  x_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_541 = self.L__mod___cell_5_comb_iter_1_left_bn_sep_1(x_540);  x_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_542 = self.L__mod___cell_5_comb_iter_1_left_act_2(x_541);  x_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_543 = self.L__mod___cell_5_comb_iter_1_left_separable_2_depthwise_conv2d(x_542);  x_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_545 = self.L__mod___cell_5_comb_iter_1_left_separable_2_pointwise_conv2d(x_543);  x_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_7 = self.L__mod___cell_5_comb_iter_1_left_bn_sep_2(x_545);  x_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_7 = self.L__mod___cell_5_comb_iter_1_right(x_comb_iter_4_right_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_36 = x_comb_iter_1_left_7 + x_comb_iter_1_right_7;  x_comb_iter_1_left_7 = x_comb_iter_1_right_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_547 = self.L__mod___cell_5_comb_iter_2_left_act_1(x_comb_iter_4_right_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_548 = self.L__mod___cell_5_comb_iter_2_left_separable_1_depthwise_conv2d(x_547);  x_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_550 = self.L__mod___cell_5_comb_iter_2_left_separable_1_pointwise_conv2d(x_548);  x_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_551 = self.L__mod___cell_5_comb_iter_2_left_bn_sep_1(x_550);  x_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_552 = self.L__mod___cell_5_comb_iter_2_left_act_2(x_551);  x_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_553 = self.L__mod___cell_5_comb_iter_2_left_separable_2_depthwise_conv2d(x_552);  x_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_555 = self.L__mod___cell_5_comb_iter_2_left_separable_2_pointwise_conv2d(x_553);  x_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_7 = self.L__mod___cell_5_comb_iter_2_left_bn_sep_2(x_555);  x_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_557 = self.L__mod___cell_5_comb_iter_2_right_act_1(x_comb_iter_4_right_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_558 = self.L__mod___cell_5_comb_iter_2_right_separable_1_depthwise_conv2d(x_557);  x_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_560 = self.L__mod___cell_5_comb_iter_2_right_separable_1_pointwise_conv2d(x_558);  x_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_561 = self.L__mod___cell_5_comb_iter_2_right_bn_sep_1(x_560);  x_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_562 = self.L__mod___cell_5_comb_iter_2_right_act_2(x_561);  x_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_563 = self.L__mod___cell_5_comb_iter_2_right_separable_2_depthwise_conv2d(x_562);  x_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_565 = self.L__mod___cell_5_comb_iter_2_right_separable_2_pointwise_conv2d(x_563);  x_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_7 = self.L__mod___cell_5_comb_iter_2_right_bn_sep_2(x_565);  x_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_37 = x_comb_iter_2_left_7 + x_comb_iter_2_right_7;  x_comb_iter_2_left_7 = x_comb_iter_2_right_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_567 = self.L__mod___cell_5_comb_iter_3_left_act_1(x_comb_iter_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_568 = self.L__mod___cell_5_comb_iter_3_left_separable_1_depthwise_conv2d(x_567);  x_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_570 = self.L__mod___cell_5_comb_iter_3_left_separable_1_pointwise_conv2d(x_568);  x_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_571 = self.L__mod___cell_5_comb_iter_3_left_bn_sep_1(x_570);  x_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_572 = self.L__mod___cell_5_comb_iter_3_left_act_2(x_571);  x_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_573 = self.L__mod___cell_5_comb_iter_3_left_separable_2_depthwise_conv2d(x_572);  x_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_575 = self.L__mod___cell_5_comb_iter_3_left_separable_2_pointwise_conv2d(x_573);  x_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_7 = self.L__mod___cell_5_comb_iter_3_left_bn_sep_2(x_575);  x_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_7 = self.L__mod___cell_5_comb_iter_3_right(x_comb_iter_4_right_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_38 = x_comb_iter_3_left_7 + x_comb_iter_3_right_7;  x_comb_iter_3_left_7 = x_comb_iter_3_right_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_577 = self.L__mod___cell_5_comb_iter_4_left_act_1(x_left_6);  x_left_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_578 = self.L__mod___cell_5_comb_iter_4_left_separable_1_depthwise_conv2d(x_577);  x_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_580 = self.L__mod___cell_5_comb_iter_4_left_separable_1_pointwise_conv2d(x_578);  x_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_581 = self.L__mod___cell_5_comb_iter_4_left_bn_sep_1(x_580);  x_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_582 = self.L__mod___cell_5_comb_iter_4_left_act_2(x_581);  x_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_583 = self.L__mod___cell_5_comb_iter_4_left_separable_2_depthwise_conv2d(x_582);  x_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_585 = self.L__mod___cell_5_comb_iter_4_left_separable_2_pointwise_conv2d(x_583);  x_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_7 = self.L__mod___cell_5_comb_iter_4_left_bn_sep_2(x_585);  x_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_39 = x_comb_iter_4_left_7 + x_comb_iter_4_right_7;  x_comb_iter_4_left_7 = x_comb_iter_4_right_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_5 = torch.cat([x_comb_iter_35, x_comb_iter_36, x_comb_iter_37, x_comb_iter_38, x_comb_iter_39], 1);  x_comb_iter_35 = x_comb_iter_36 = x_comb_iter_37 = x_comb_iter_38 = x_comb_iter_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_587 = self.L__mod___cell_6_conv_prev_1x1_act(x_cell_4);  x_cell_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_588 = self.L__mod___cell_6_conv_prev_1x1_conv(x_587);  x_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_7 = self.L__mod___cell_6_conv_prev_1x1_bn(x_588);  x_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_590 = self.L__mod___cell_6_conv_1x1_act(x_cell_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_591 = self.L__mod___cell_6_conv_1x1_conv(x_590);  x_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_8 = self.L__mod___cell_6_conv_1x1_bn(x_591);  x_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_593 = self.L__mod___cell_6_comb_iter_0_left_act_1(x_left_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_594 = self.L__mod___cell_6_comb_iter_0_left_separable_1_depthwise_conv2d(x_593);  x_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_596 = self.L__mod___cell_6_comb_iter_0_left_separable_1_pointwise_conv2d(x_594);  x_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_597 = self.L__mod___cell_6_comb_iter_0_left_bn_sep_1(x_596);  x_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_598 = self.L__mod___cell_6_comb_iter_0_left_act_2(x_597);  x_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_599 = self.L__mod___cell_6_comb_iter_0_left_separable_2_depthwise_conv2d(x_598);  x_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_601 = self.L__mod___cell_6_comb_iter_0_left_separable_2_pointwise_conv2d(x_599);  x_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_8 = self.L__mod___cell_6_comb_iter_0_left_bn_sep_2(x_601);  x_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_8 = self.L__mod___cell_6_comb_iter_0_right(x_left_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_40 = x_comb_iter_0_left_8 + x_comb_iter_0_right_8;  x_comb_iter_0_left_8 = x_comb_iter_0_right_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_603 = self.L__mod___cell_6_comb_iter_1_left_act_1(x_comb_iter_4_right_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_604 = self.L__mod___cell_6_comb_iter_1_left_separable_1_depthwise_conv2d(x_603);  x_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_606 = self.L__mod___cell_6_comb_iter_1_left_separable_1_pointwise_conv2d(x_604);  x_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_607 = self.L__mod___cell_6_comb_iter_1_left_bn_sep_1(x_606);  x_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_608 = self.L__mod___cell_6_comb_iter_1_left_act_2(x_607);  x_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_609 = self.L__mod___cell_6_comb_iter_1_left_separable_2_depthwise_conv2d(x_608);  x_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_611 = self.L__mod___cell_6_comb_iter_1_left_separable_2_pointwise_conv2d(x_609);  x_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_8 = self.L__mod___cell_6_comb_iter_1_left_bn_sep_2(x_611);  x_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_8 = self.L__mod___cell_6_comb_iter_1_right(x_comb_iter_4_right_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_41 = x_comb_iter_1_left_8 + x_comb_iter_1_right_8;  x_comb_iter_1_left_8 = x_comb_iter_1_right_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_613 = self.L__mod___cell_6_comb_iter_2_left_act_1(x_comb_iter_4_right_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_614 = self.L__mod___cell_6_comb_iter_2_left_separable_1_depthwise_conv2d(x_613);  x_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_616 = self.L__mod___cell_6_comb_iter_2_left_separable_1_pointwise_conv2d(x_614);  x_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_617 = self.L__mod___cell_6_comb_iter_2_left_bn_sep_1(x_616);  x_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_618 = self.L__mod___cell_6_comb_iter_2_left_act_2(x_617);  x_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_619 = self.L__mod___cell_6_comb_iter_2_left_separable_2_depthwise_conv2d(x_618);  x_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_621 = self.L__mod___cell_6_comb_iter_2_left_separable_2_pointwise_conv2d(x_619);  x_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_8 = self.L__mod___cell_6_comb_iter_2_left_bn_sep_2(x_621);  x_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_623 = self.L__mod___cell_6_comb_iter_2_right_act_1(x_comb_iter_4_right_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_624 = self.L__mod___cell_6_comb_iter_2_right_separable_1_depthwise_conv2d(x_623);  x_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_626 = self.L__mod___cell_6_comb_iter_2_right_separable_1_pointwise_conv2d(x_624);  x_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_627 = self.L__mod___cell_6_comb_iter_2_right_bn_sep_1(x_626);  x_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_628 = self.L__mod___cell_6_comb_iter_2_right_act_2(x_627);  x_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_629 = self.L__mod___cell_6_comb_iter_2_right_separable_2_depthwise_conv2d(x_628);  x_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_631 = self.L__mod___cell_6_comb_iter_2_right_separable_2_pointwise_conv2d(x_629);  x_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_8 = self.L__mod___cell_6_comb_iter_2_right_bn_sep_2(x_631);  x_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_42 = x_comb_iter_2_left_8 + x_comb_iter_2_right_8;  x_comb_iter_2_left_8 = x_comb_iter_2_right_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_633 = self.L__mod___cell_6_comb_iter_3_left_act_1(x_comb_iter_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_634 = self.L__mod___cell_6_comb_iter_3_left_separable_1_depthwise_conv2d(x_633);  x_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_636 = self.L__mod___cell_6_comb_iter_3_left_separable_1_pointwise_conv2d(x_634);  x_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_637 = self.L__mod___cell_6_comb_iter_3_left_bn_sep_1(x_636);  x_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_638 = self.L__mod___cell_6_comb_iter_3_left_act_2(x_637);  x_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_639 = self.L__mod___cell_6_comb_iter_3_left_separable_2_depthwise_conv2d(x_638);  x_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_641 = self.L__mod___cell_6_comb_iter_3_left_separable_2_pointwise_conv2d(x_639);  x_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_8 = self.L__mod___cell_6_comb_iter_3_left_bn_sep_2(x_641);  x_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_8 = self.L__mod___cell_6_comb_iter_3_right(x_comb_iter_4_right_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_43 = x_comb_iter_3_left_8 + x_comb_iter_3_right_8;  x_comb_iter_3_left_8 = x_comb_iter_3_right_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_643 = self.L__mod___cell_6_comb_iter_4_left_act_1(x_left_7);  x_left_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_644 = self.L__mod___cell_6_comb_iter_4_left_separable_1_depthwise_conv2d(x_643);  x_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_646 = self.L__mod___cell_6_comb_iter_4_left_separable_1_pointwise_conv2d(x_644);  x_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_647 = self.L__mod___cell_6_comb_iter_4_left_bn_sep_1(x_646);  x_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_648 = self.L__mod___cell_6_comb_iter_4_left_act_2(x_647);  x_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_649 = self.L__mod___cell_6_comb_iter_4_left_separable_2_depthwise_conv2d(x_648);  x_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_651 = self.L__mod___cell_6_comb_iter_4_left_separable_2_pointwise_conv2d(x_649);  x_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_8 = self.L__mod___cell_6_comb_iter_4_left_bn_sep_2(x_651);  x_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_44 = x_comb_iter_4_left_8 + x_comb_iter_4_right_8;  x_comb_iter_4_left_8 = x_comb_iter_4_right_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_6 = torch.cat([x_comb_iter_40, x_comb_iter_41, x_comb_iter_42, x_comb_iter_43, x_comb_iter_44], 1);  x_comb_iter_40 = x_comb_iter_41 = x_comb_iter_42 = x_comb_iter_43 = x_comb_iter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_653 = self.L__mod___cell_7_conv_prev_1x1_act(x_cell_5);  x_cell_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_654 = self.L__mod___cell_7_conv_prev_1x1_conv(x_653);  x_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_8 = self.L__mod___cell_7_conv_prev_1x1_bn(x_654);  x_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_656 = self.L__mod___cell_7_conv_1x1_act(x_cell_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_657 = self.L__mod___cell_7_conv_1x1_conv(x_656);  x_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_9 = self.L__mod___cell_7_conv_1x1_bn(x_657);  x_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_659 = self.L__mod___cell_7_comb_iter_0_left_act_1(x_left_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_660 = self.L__mod___cell_7_comb_iter_0_left_separable_1_depthwise_conv2d(x_659);  x_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_662 = self.L__mod___cell_7_comb_iter_0_left_separable_1_pointwise_conv2d(x_660);  x_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_663 = self.L__mod___cell_7_comb_iter_0_left_bn_sep_1(x_662);  x_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_664 = self.L__mod___cell_7_comb_iter_0_left_act_2(x_663);  x_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_665 = self.L__mod___cell_7_comb_iter_0_left_separable_2_depthwise_conv2d(x_664);  x_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_667 = self.L__mod___cell_7_comb_iter_0_left_separable_2_pointwise_conv2d(x_665);  x_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_9 = self.L__mod___cell_7_comb_iter_0_left_bn_sep_2(x_667);  x_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_9 = self.L__mod___cell_7_comb_iter_0_right(x_left_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_45 = x_comb_iter_0_left_9 + x_comb_iter_0_right_9;  x_comb_iter_0_left_9 = x_comb_iter_0_right_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_669 = self.L__mod___cell_7_comb_iter_1_left_act_1(x_comb_iter_4_right_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_670 = self.L__mod___cell_7_comb_iter_1_left_separable_1_depthwise_conv2d(x_669);  x_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_672 = self.L__mod___cell_7_comb_iter_1_left_separable_1_pointwise_conv2d(x_670);  x_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_673 = self.L__mod___cell_7_comb_iter_1_left_bn_sep_1(x_672);  x_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_674 = self.L__mod___cell_7_comb_iter_1_left_act_2(x_673);  x_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_675 = self.L__mod___cell_7_comb_iter_1_left_separable_2_depthwise_conv2d(x_674);  x_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_677 = self.L__mod___cell_7_comb_iter_1_left_separable_2_pointwise_conv2d(x_675);  x_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_9 = self.L__mod___cell_7_comb_iter_1_left_bn_sep_2(x_677);  x_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_9 = self.L__mod___cell_7_comb_iter_1_right(x_comb_iter_4_right_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_46 = x_comb_iter_1_left_9 + x_comb_iter_1_right_9;  x_comb_iter_1_left_9 = x_comb_iter_1_right_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_679 = self.L__mod___cell_7_comb_iter_2_left_act_1(x_comb_iter_4_right_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_680 = self.L__mod___cell_7_comb_iter_2_left_separable_1_depthwise_conv2d(x_679);  x_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_682 = self.L__mod___cell_7_comb_iter_2_left_separable_1_pointwise_conv2d(x_680);  x_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_683 = self.L__mod___cell_7_comb_iter_2_left_bn_sep_1(x_682);  x_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_684 = self.L__mod___cell_7_comb_iter_2_left_act_2(x_683);  x_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_685 = self.L__mod___cell_7_comb_iter_2_left_separable_2_depthwise_conv2d(x_684);  x_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_687 = self.L__mod___cell_7_comb_iter_2_left_separable_2_pointwise_conv2d(x_685);  x_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_9 = self.L__mod___cell_7_comb_iter_2_left_bn_sep_2(x_687);  x_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_689 = self.L__mod___cell_7_comb_iter_2_right_act_1(x_comb_iter_4_right_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_690 = self.L__mod___cell_7_comb_iter_2_right_separable_1_depthwise_conv2d(x_689);  x_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_692 = self.L__mod___cell_7_comb_iter_2_right_separable_1_pointwise_conv2d(x_690);  x_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_693 = self.L__mod___cell_7_comb_iter_2_right_bn_sep_1(x_692);  x_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_694 = self.L__mod___cell_7_comb_iter_2_right_act_2(x_693);  x_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_695 = self.L__mod___cell_7_comb_iter_2_right_separable_2_depthwise_conv2d(x_694);  x_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_697 = self.L__mod___cell_7_comb_iter_2_right_separable_2_pointwise_conv2d(x_695);  x_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_9 = self.L__mod___cell_7_comb_iter_2_right_bn_sep_2(x_697);  x_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_47 = x_comb_iter_2_left_9 + x_comb_iter_2_right_9;  x_comb_iter_2_left_9 = x_comb_iter_2_right_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_699 = self.L__mod___cell_7_comb_iter_3_left_act_1(x_comb_iter_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_700 = self.L__mod___cell_7_comb_iter_3_left_separable_1_depthwise_conv2d(x_699);  x_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_702 = self.L__mod___cell_7_comb_iter_3_left_separable_1_pointwise_conv2d(x_700);  x_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_703 = self.L__mod___cell_7_comb_iter_3_left_bn_sep_1(x_702);  x_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_704 = self.L__mod___cell_7_comb_iter_3_left_act_2(x_703);  x_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_705 = self.L__mod___cell_7_comb_iter_3_left_separable_2_depthwise_conv2d(x_704);  x_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_707 = self.L__mod___cell_7_comb_iter_3_left_separable_2_pointwise_conv2d(x_705);  x_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_9 = self.L__mod___cell_7_comb_iter_3_left_bn_sep_2(x_707);  x_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_9 = self.L__mod___cell_7_comb_iter_3_right(x_comb_iter_4_right_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_48 = x_comb_iter_3_left_9 + x_comb_iter_3_right_9;  x_comb_iter_3_left_9 = x_comb_iter_3_right_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_709 = self.L__mod___cell_7_comb_iter_4_left_act_1(x_left_8);  x_left_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_710 = self.L__mod___cell_7_comb_iter_4_left_separable_1_depthwise_conv2d(x_709);  x_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_712 = self.L__mod___cell_7_comb_iter_4_left_separable_1_pointwise_conv2d(x_710);  x_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_713 = self.L__mod___cell_7_comb_iter_4_left_bn_sep_1(x_712);  x_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_714 = self.L__mod___cell_7_comb_iter_4_left_act_2(x_713);  x_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_715 = self.L__mod___cell_7_comb_iter_4_left_separable_2_depthwise_conv2d(x_714);  x_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_717 = self.L__mod___cell_7_comb_iter_4_left_separable_2_pointwise_conv2d(x_715);  x_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_9 = self.L__mod___cell_7_comb_iter_4_left_bn_sep_2(x_717);  x_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_49 = x_comb_iter_4_left_9 + x_comb_iter_4_right_9;  x_comb_iter_4_left_9 = x_comb_iter_4_right_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_7 = torch.cat([x_comb_iter_45, x_comb_iter_46, x_comb_iter_47, x_comb_iter_48, x_comb_iter_49], 1);  x_comb_iter_45 = x_comb_iter_46 = x_comb_iter_47 = x_comb_iter_48 = x_comb_iter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_719 = self.L__mod___cell_8_conv_prev_1x1_act(x_cell_6);  x_cell_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_720 = self.L__mod___cell_8_conv_prev_1x1_conv(x_719);  x_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_9 = self.L__mod___cell_8_conv_prev_1x1_bn(x_720);  x_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_722 = self.L__mod___cell_8_conv_1x1_act(x_cell_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_723 = self.L__mod___cell_8_conv_1x1_conv(x_722);  x_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_right_10 = self.L__mod___cell_8_conv_1x1_bn(x_723);  x_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_725 = self.L__mod___cell_8_comb_iter_0_left_act_1(x_left_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_8_comb_iter_0_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_8_comb_iter_0_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_727 = torch.nn.functional.pad(x_725, (2, 2, 2, 2), value = 0);  x_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_728 = torch.conv2d(x_727, l__mod___cell_8_comb_iter_0_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 864);  x_727 = l__mod___cell_8_comb_iter_0_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_730 = self.L__mod___cell_8_comb_iter_0_left_separable_1_pointwise_conv2d(x_728);  x_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_731 = self.L__mod___cell_8_comb_iter_0_left_bn_sep_1(x_730);  x_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_732 = self.L__mod___cell_8_comb_iter_0_left_act_2(x_731);  x_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_733 = self.L__mod___cell_8_comb_iter_0_left_separable_2_depthwise_conv2d(x_732);  x_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_735 = self.L__mod___cell_8_comb_iter_0_left_separable_2_pointwise_conv2d(x_733);  x_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_10 = self.L__mod___cell_8_comb_iter_0_left_bn_sep_2(x_735);  x_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_738 = torch.nn.functional.pad(x_left_9, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_0_right_10 = torch.nn.functional.max_pool2d(x_738, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_50 = x_comb_iter_0_left_10 + x_comb_iter_0_right_10;  x_comb_iter_0_left_10 = x_comb_iter_0_right_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_739 = self.L__mod___cell_8_comb_iter_1_left_act_1(x_right_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_8_comb_iter_1_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_8_comb_iter_1_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_741 = torch.nn.functional.pad(x_739, (3, 3, 3, 3), value = 0);  x_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_742 = torch.conv2d(x_741, l__mod___cell_8_comb_iter_1_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 864);  x_741 = l__mod___cell_8_comb_iter_1_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_744 = self.L__mod___cell_8_comb_iter_1_left_separable_1_pointwise_conv2d(x_742);  x_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_745 = self.L__mod___cell_8_comb_iter_1_left_bn_sep_1(x_744);  x_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_746 = self.L__mod___cell_8_comb_iter_1_left_act_2(x_745);  x_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_747 = self.L__mod___cell_8_comb_iter_1_left_separable_2_depthwise_conv2d(x_746);  x_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_749 = self.L__mod___cell_8_comb_iter_1_left_separable_2_pointwise_conv2d(x_747);  x_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_10 = self.L__mod___cell_8_comb_iter_1_left_bn_sep_2(x_749);  x_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_752 = torch.nn.functional.pad(x_right_10, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_1_right_10 = torch.nn.functional.max_pool2d(x_752, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_51 = x_comb_iter_1_left_10 + x_comb_iter_1_right_10;  x_comb_iter_1_left_10 = x_comb_iter_1_right_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_753 = self.L__mod___cell_8_comb_iter_2_left_act_1(x_right_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_8_comb_iter_2_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_8_comb_iter_2_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_755 = torch.nn.functional.pad(x_753, (2, 2, 2, 2), value = 0);  x_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_756 = torch.conv2d(x_755, l__mod___cell_8_comb_iter_2_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 864);  x_755 = l__mod___cell_8_comb_iter_2_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_758 = self.L__mod___cell_8_comb_iter_2_left_separable_1_pointwise_conv2d(x_756);  x_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_759 = self.L__mod___cell_8_comb_iter_2_left_bn_sep_1(x_758);  x_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_760 = self.L__mod___cell_8_comb_iter_2_left_act_2(x_759);  x_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_761 = self.L__mod___cell_8_comb_iter_2_left_separable_2_depthwise_conv2d(x_760);  x_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_763 = self.L__mod___cell_8_comb_iter_2_left_separable_2_pointwise_conv2d(x_761);  x_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_10 = self.L__mod___cell_8_comb_iter_2_left_bn_sep_2(x_763);  x_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_765 = self.L__mod___cell_8_comb_iter_2_right_act_1(x_right_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_8_comb_iter_2_right_separable_1_depthwise_conv2d_weight = self.L__mod___cell_8_comb_iter_2_right_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_767 = torch.nn.functional.pad(x_765, (1, 1, 1, 1), value = 0);  x_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_768 = torch.conv2d(x_767, l__mod___cell_8_comb_iter_2_right_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 864);  x_767 = l__mod___cell_8_comb_iter_2_right_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_770 = self.L__mod___cell_8_comb_iter_2_right_separable_1_pointwise_conv2d(x_768);  x_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_771 = self.L__mod___cell_8_comb_iter_2_right_bn_sep_1(x_770);  x_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_772 = self.L__mod___cell_8_comb_iter_2_right_act_2(x_771);  x_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_773 = self.L__mod___cell_8_comb_iter_2_right_separable_2_depthwise_conv2d(x_772);  x_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_775 = self.L__mod___cell_8_comb_iter_2_right_separable_2_pointwise_conv2d(x_773);  x_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_10 = self.L__mod___cell_8_comb_iter_2_right_bn_sep_2(x_775);  x_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_52 = x_comb_iter_2_left_10 + x_comb_iter_2_right_10;  x_comb_iter_2_left_10 = x_comb_iter_2_right_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_777 = self.L__mod___cell_8_comb_iter_3_left_act_1(x_comb_iter_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_778 = self.L__mod___cell_8_comb_iter_3_left_separable_1_depthwise_conv2d(x_777);  x_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_780 = self.L__mod___cell_8_comb_iter_3_left_separable_1_pointwise_conv2d(x_778);  x_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_781 = self.L__mod___cell_8_comb_iter_3_left_bn_sep_1(x_780);  x_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_782 = self.L__mod___cell_8_comb_iter_3_left_act_2(x_781);  x_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_783 = self.L__mod___cell_8_comb_iter_3_left_separable_2_depthwise_conv2d(x_782);  x_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_785 = self.L__mod___cell_8_comb_iter_3_left_separable_2_pointwise_conv2d(x_783);  x_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_10 = self.L__mod___cell_8_comb_iter_3_left_bn_sep_2(x_785);  x_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_788 = torch.nn.functional.pad(x_right_10, (1, 1, 1, 1), value = -inf)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    x_comb_iter_3_right_10 = torch.nn.functional.max_pool2d(x_788, (3, 3), (2, 2), (0, 0), (1, 1), False);  x_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_53 = x_comb_iter_3_left_10 + x_comb_iter_3_right_10;  x_comb_iter_3_left_10 = x_comb_iter_3_right_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_789 = self.L__mod___cell_8_comb_iter_4_left_act_1(x_left_9);  x_left_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_8_comb_iter_4_left_separable_1_depthwise_conv2d_weight = self.L__mod___cell_8_comb_iter_4_left_separable_1_depthwise_conv2d_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_791 = torch.nn.functional.pad(x_789, (1, 1, 1, 1), value = 0);  x_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_792 = torch.conv2d(x_791, l__mod___cell_8_comb_iter_4_left_separable_1_depthwise_conv2d_weight, None, (2, 2), (0, 0), (1, 1), 864);  x_791 = l__mod___cell_8_comb_iter_4_left_separable_1_depthwise_conv2d_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_794 = self.L__mod___cell_8_comb_iter_4_left_separable_1_pointwise_conv2d(x_792);  x_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_795 = self.L__mod___cell_8_comb_iter_4_left_bn_sep_1(x_794);  x_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_796 = self.L__mod___cell_8_comb_iter_4_left_act_2(x_795);  x_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_797 = self.L__mod___cell_8_comb_iter_4_left_separable_2_depthwise_conv2d(x_796);  x_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_799 = self.L__mod___cell_8_comb_iter_4_left_separable_2_pointwise_conv2d(x_797);  x_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_10 = self.L__mod___cell_8_comb_iter_4_left_bn_sep_2(x_799);  x_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_801 = self.L__mod___cell_8_comb_iter_4_right_act(x_right_10);  x_right_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:52, code: x, self.weight, self.bias,
    l__mod___cell_8_comb_iter_4_right_conv_weight = self.L__mod___cell_8_comb_iter_4_right_conv_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    x_803 = torch.nn.functional.pad(x_801, (0, 0, 0, 0), value = 0);  x_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    x_804 = torch.conv2d(x_803, l__mod___cell_8_comb_iter_4_right_conv_weight, None, (2, 2), (0, 0), (1, 1), 1);  x_803 = l__mod___cell_8_comb_iter_4_right_conv_weight = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_10 = self.L__mod___cell_8_comb_iter_4_right_bn(x_804);  x_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_54 = x_comb_iter_4_left_10 + x_comb_iter_4_right_10;  x_comb_iter_4_left_10 = x_comb_iter_4_right_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_8 = torch.cat([x_comb_iter_50, x_comb_iter_51, x_comb_iter_52, x_comb_iter_53, x_comb_iter_54], 1);  x_comb_iter_50 = x_comb_iter_51 = x_comb_iter_52 = x_comb_iter_53 = x_comb_iter_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:95, code: x = self.act(x)
    x_806 = self.L__mod___cell_9_conv_prev_1x1_act(x_cell_7);  x_cell_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    l__mod___cell_9_conv_prev_1x1_path_1_avgpool = self.L__mod___cell_9_conv_prev_1x1_path_1_avgpool(x_806)
    x_path1_3 = self.L__mod___cell_9_conv_prev_1x1_path_1_conv(l__mod___cell_9_conv_prev_1x1_path_1_avgpool);  l__mod___cell_9_conv_prev_1x1_path_1_avgpool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    l__mod___cell_9_conv_prev_1x1_path_2_pad = self.L__mod___cell_9_conv_prev_1x1_path_2_pad(x_806);  x_806 = None
    l__mod___cell_9_conv_prev_1x1_path_2_avgpool = self.L__mod___cell_9_conv_prev_1x1_path_2_avgpool(l__mod___cell_9_conv_prev_1x1_path_2_pad);  l__mod___cell_9_conv_prev_1x1_path_2_pad = None
    x_path2_3 = self.L__mod___cell_9_conv_prev_1x1_path_2_conv(l__mod___cell_9_conv_prev_1x1_path_2_avgpool);  l__mod___cell_9_conv_prev_1x1_path_2_avgpool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    cat_14 = torch.cat([x_path1_3, x_path2_3], 1);  x_path1_3 = x_path2_3 = None
    x_left_10 = self.L__mod___cell_9_conv_prev_1x1_final_path_bn(cat_14);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_807 = self.L__mod___cell_9_conv_1x1_act(x_cell_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_808 = self.L__mod___cell_9_conv_1x1_conv(x_807);  x_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_11 = self.L__mod___cell_9_conv_1x1_bn(x_808);  x_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_810 = self.L__mod___cell_9_comb_iter_0_left_act_1(x_left_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_811 = self.L__mod___cell_9_comb_iter_0_left_separable_1_depthwise_conv2d(x_810);  x_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_813 = self.L__mod___cell_9_comb_iter_0_left_separable_1_pointwise_conv2d(x_811);  x_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_814 = self.L__mod___cell_9_comb_iter_0_left_bn_sep_1(x_813);  x_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_815 = self.L__mod___cell_9_comb_iter_0_left_act_2(x_814);  x_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_816 = self.L__mod___cell_9_comb_iter_0_left_separable_2_depthwise_conv2d(x_815);  x_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_818 = self.L__mod___cell_9_comb_iter_0_left_separable_2_pointwise_conv2d(x_816);  x_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_11 = self.L__mod___cell_9_comb_iter_0_left_bn_sep_2(x_818);  x_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_11 = self.L__mod___cell_9_comb_iter_0_right(x_left_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_55 = x_comb_iter_0_left_11 + x_comb_iter_0_right_11;  x_comb_iter_0_left_11 = x_comb_iter_0_right_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_820 = self.L__mod___cell_9_comb_iter_1_left_act_1(x_comb_iter_4_right_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_821 = self.L__mod___cell_9_comb_iter_1_left_separable_1_depthwise_conv2d(x_820);  x_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_823 = self.L__mod___cell_9_comb_iter_1_left_separable_1_pointwise_conv2d(x_821);  x_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_824 = self.L__mod___cell_9_comb_iter_1_left_bn_sep_1(x_823);  x_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_825 = self.L__mod___cell_9_comb_iter_1_left_act_2(x_824);  x_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_826 = self.L__mod___cell_9_comb_iter_1_left_separable_2_depthwise_conv2d(x_825);  x_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_828 = self.L__mod___cell_9_comb_iter_1_left_separable_2_pointwise_conv2d(x_826);  x_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_11 = self.L__mod___cell_9_comb_iter_1_left_bn_sep_2(x_828);  x_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_11 = self.L__mod___cell_9_comb_iter_1_right(x_comb_iter_4_right_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_56 = x_comb_iter_1_left_11 + x_comb_iter_1_right_11;  x_comb_iter_1_left_11 = x_comb_iter_1_right_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_830 = self.L__mod___cell_9_comb_iter_2_left_act_1(x_comb_iter_4_right_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_831 = self.L__mod___cell_9_comb_iter_2_left_separable_1_depthwise_conv2d(x_830);  x_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_833 = self.L__mod___cell_9_comb_iter_2_left_separable_1_pointwise_conv2d(x_831);  x_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_834 = self.L__mod___cell_9_comb_iter_2_left_bn_sep_1(x_833);  x_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_835 = self.L__mod___cell_9_comb_iter_2_left_act_2(x_834);  x_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_836 = self.L__mod___cell_9_comb_iter_2_left_separable_2_depthwise_conv2d(x_835);  x_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_838 = self.L__mod___cell_9_comb_iter_2_left_separable_2_pointwise_conv2d(x_836);  x_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_11 = self.L__mod___cell_9_comb_iter_2_left_bn_sep_2(x_838);  x_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_840 = self.L__mod___cell_9_comb_iter_2_right_act_1(x_comb_iter_4_right_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_841 = self.L__mod___cell_9_comb_iter_2_right_separable_1_depthwise_conv2d(x_840);  x_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_843 = self.L__mod___cell_9_comb_iter_2_right_separable_1_pointwise_conv2d(x_841);  x_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_844 = self.L__mod___cell_9_comb_iter_2_right_bn_sep_1(x_843);  x_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_845 = self.L__mod___cell_9_comb_iter_2_right_act_2(x_844);  x_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_846 = self.L__mod___cell_9_comb_iter_2_right_separable_2_depthwise_conv2d(x_845);  x_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_848 = self.L__mod___cell_9_comb_iter_2_right_separable_2_pointwise_conv2d(x_846);  x_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_11 = self.L__mod___cell_9_comb_iter_2_right_bn_sep_2(x_848);  x_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_57 = x_comb_iter_2_left_11 + x_comb_iter_2_right_11;  x_comb_iter_2_left_11 = x_comb_iter_2_right_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_850 = self.L__mod___cell_9_comb_iter_3_left_act_1(x_comb_iter_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_851 = self.L__mod___cell_9_comb_iter_3_left_separable_1_depthwise_conv2d(x_850);  x_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_853 = self.L__mod___cell_9_comb_iter_3_left_separable_1_pointwise_conv2d(x_851);  x_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_854 = self.L__mod___cell_9_comb_iter_3_left_bn_sep_1(x_853);  x_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_855 = self.L__mod___cell_9_comb_iter_3_left_act_2(x_854);  x_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_856 = self.L__mod___cell_9_comb_iter_3_left_separable_2_depthwise_conv2d(x_855);  x_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_858 = self.L__mod___cell_9_comb_iter_3_left_separable_2_pointwise_conv2d(x_856);  x_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_11 = self.L__mod___cell_9_comb_iter_3_left_bn_sep_2(x_858);  x_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_11 = self.L__mod___cell_9_comb_iter_3_right(x_comb_iter_4_right_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_58 = x_comb_iter_3_left_11 + x_comb_iter_3_right_11;  x_comb_iter_3_left_11 = x_comb_iter_3_right_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_860 = self.L__mod___cell_9_comb_iter_4_left_act_1(x_left_10);  x_left_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_861 = self.L__mod___cell_9_comb_iter_4_left_separable_1_depthwise_conv2d(x_860);  x_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_863 = self.L__mod___cell_9_comb_iter_4_left_separable_1_pointwise_conv2d(x_861);  x_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_864 = self.L__mod___cell_9_comb_iter_4_left_bn_sep_1(x_863);  x_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_865 = self.L__mod___cell_9_comb_iter_4_left_act_2(x_864);  x_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_866 = self.L__mod___cell_9_comb_iter_4_left_separable_2_depthwise_conv2d(x_865);  x_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_868 = self.L__mod___cell_9_comb_iter_4_left_separable_2_pointwise_conv2d(x_866);  x_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_11 = self.L__mod___cell_9_comb_iter_4_left_bn_sep_2(x_868);  x_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_59 = x_comb_iter_4_left_11 + x_comb_iter_4_right_11;  x_comb_iter_4_left_11 = x_comb_iter_4_right_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_9 = torch.cat([x_comb_iter_55, x_comb_iter_56, x_comb_iter_57, x_comb_iter_58, x_comb_iter_59], 1);  x_comb_iter_55 = x_comb_iter_56 = x_comb_iter_57 = x_comb_iter_58 = x_comb_iter_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_870 = self.L__mod___cell_10_conv_prev_1x1_act(x_cell_8);  x_cell_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_871 = self.L__mod___cell_10_conv_prev_1x1_conv(x_870);  x_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_11 = self.L__mod___cell_10_conv_prev_1x1_bn(x_871);  x_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_873 = self.L__mod___cell_10_conv_1x1_act(x_cell_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_874 = self.L__mod___cell_10_conv_1x1_conv(x_873);  x_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_12 = self.L__mod___cell_10_conv_1x1_bn(x_874);  x_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_876 = self.L__mod___cell_10_comb_iter_0_left_act_1(x_left_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_877 = self.L__mod___cell_10_comb_iter_0_left_separable_1_depthwise_conv2d(x_876);  x_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_879 = self.L__mod___cell_10_comb_iter_0_left_separable_1_pointwise_conv2d(x_877);  x_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_880 = self.L__mod___cell_10_comb_iter_0_left_bn_sep_1(x_879);  x_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_881 = self.L__mod___cell_10_comb_iter_0_left_act_2(x_880);  x_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_882 = self.L__mod___cell_10_comb_iter_0_left_separable_2_depthwise_conv2d(x_881);  x_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_884 = self.L__mod___cell_10_comb_iter_0_left_separable_2_pointwise_conv2d(x_882);  x_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_12 = self.L__mod___cell_10_comb_iter_0_left_bn_sep_2(x_884);  x_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_12 = self.L__mod___cell_10_comb_iter_0_right(x_left_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_60 = x_comb_iter_0_left_12 + x_comb_iter_0_right_12;  x_comb_iter_0_left_12 = x_comb_iter_0_right_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_886 = self.L__mod___cell_10_comb_iter_1_left_act_1(x_comb_iter_4_right_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_887 = self.L__mod___cell_10_comb_iter_1_left_separable_1_depthwise_conv2d(x_886);  x_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_889 = self.L__mod___cell_10_comb_iter_1_left_separable_1_pointwise_conv2d(x_887);  x_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_890 = self.L__mod___cell_10_comb_iter_1_left_bn_sep_1(x_889);  x_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_891 = self.L__mod___cell_10_comb_iter_1_left_act_2(x_890);  x_890 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_892 = self.L__mod___cell_10_comb_iter_1_left_separable_2_depthwise_conv2d(x_891);  x_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_894 = self.L__mod___cell_10_comb_iter_1_left_separable_2_pointwise_conv2d(x_892);  x_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_12 = self.L__mod___cell_10_comb_iter_1_left_bn_sep_2(x_894);  x_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_12 = self.L__mod___cell_10_comb_iter_1_right(x_comb_iter_4_right_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_61 = x_comb_iter_1_left_12 + x_comb_iter_1_right_12;  x_comb_iter_1_left_12 = x_comb_iter_1_right_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_896 = self.L__mod___cell_10_comb_iter_2_left_act_1(x_comb_iter_4_right_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_897 = self.L__mod___cell_10_comb_iter_2_left_separable_1_depthwise_conv2d(x_896);  x_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_899 = self.L__mod___cell_10_comb_iter_2_left_separable_1_pointwise_conv2d(x_897);  x_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_900 = self.L__mod___cell_10_comb_iter_2_left_bn_sep_1(x_899);  x_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_901 = self.L__mod___cell_10_comb_iter_2_left_act_2(x_900);  x_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_902 = self.L__mod___cell_10_comb_iter_2_left_separable_2_depthwise_conv2d(x_901);  x_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_904 = self.L__mod___cell_10_comb_iter_2_left_separable_2_pointwise_conv2d(x_902);  x_902 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_12 = self.L__mod___cell_10_comb_iter_2_left_bn_sep_2(x_904);  x_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_906 = self.L__mod___cell_10_comb_iter_2_right_act_1(x_comb_iter_4_right_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_907 = self.L__mod___cell_10_comb_iter_2_right_separable_1_depthwise_conv2d(x_906);  x_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_909 = self.L__mod___cell_10_comb_iter_2_right_separable_1_pointwise_conv2d(x_907);  x_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_910 = self.L__mod___cell_10_comb_iter_2_right_bn_sep_1(x_909);  x_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_911 = self.L__mod___cell_10_comb_iter_2_right_act_2(x_910);  x_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_912 = self.L__mod___cell_10_comb_iter_2_right_separable_2_depthwise_conv2d(x_911);  x_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_914 = self.L__mod___cell_10_comb_iter_2_right_separable_2_pointwise_conv2d(x_912);  x_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_12 = self.L__mod___cell_10_comb_iter_2_right_bn_sep_2(x_914);  x_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_62 = x_comb_iter_2_left_12 + x_comb_iter_2_right_12;  x_comb_iter_2_left_12 = x_comb_iter_2_right_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_916 = self.L__mod___cell_10_comb_iter_3_left_act_1(x_comb_iter_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_917 = self.L__mod___cell_10_comb_iter_3_left_separable_1_depthwise_conv2d(x_916);  x_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_919 = self.L__mod___cell_10_comb_iter_3_left_separable_1_pointwise_conv2d(x_917);  x_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_920 = self.L__mod___cell_10_comb_iter_3_left_bn_sep_1(x_919);  x_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_921 = self.L__mod___cell_10_comb_iter_3_left_act_2(x_920);  x_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_922 = self.L__mod___cell_10_comb_iter_3_left_separable_2_depthwise_conv2d(x_921);  x_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_924 = self.L__mod___cell_10_comb_iter_3_left_separable_2_pointwise_conv2d(x_922);  x_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_12 = self.L__mod___cell_10_comb_iter_3_left_bn_sep_2(x_924);  x_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_12 = self.L__mod___cell_10_comb_iter_3_right(x_comb_iter_4_right_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_63 = x_comb_iter_3_left_12 + x_comb_iter_3_right_12;  x_comb_iter_3_left_12 = x_comb_iter_3_right_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_926 = self.L__mod___cell_10_comb_iter_4_left_act_1(x_left_11);  x_left_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_927 = self.L__mod___cell_10_comb_iter_4_left_separable_1_depthwise_conv2d(x_926);  x_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_929 = self.L__mod___cell_10_comb_iter_4_left_separable_1_pointwise_conv2d(x_927);  x_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_930 = self.L__mod___cell_10_comb_iter_4_left_bn_sep_1(x_929);  x_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_931 = self.L__mod___cell_10_comb_iter_4_left_act_2(x_930);  x_930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_932 = self.L__mod___cell_10_comb_iter_4_left_separable_2_depthwise_conv2d(x_931);  x_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_934 = self.L__mod___cell_10_comb_iter_4_left_separable_2_pointwise_conv2d(x_932);  x_932 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_12 = self.L__mod___cell_10_comb_iter_4_left_bn_sep_2(x_934);  x_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_64 = x_comb_iter_4_left_12 + x_comb_iter_4_right_12;  x_comb_iter_4_left_12 = x_comb_iter_4_right_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_10 = torch.cat([x_comb_iter_60, x_comb_iter_61, x_comb_iter_62, x_comb_iter_63, x_comb_iter_64], 1);  x_comb_iter_60 = x_comb_iter_61 = x_comb_iter_62 = x_comb_iter_63 = x_comb_iter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_936 = self.L__mod___cell_11_conv_prev_1x1_act(x_cell_9);  x_cell_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_937 = self.L__mod___cell_11_conv_prev_1x1_conv(x_936);  x_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_left_12 = self.L__mod___cell_11_conv_prev_1x1_bn(x_937);  x_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    x_939 = self.L__mod___cell_11_conv_1x1_act(x_cell_10);  x_cell_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    x_940 = self.L__mod___cell_11_conv_1x1_conv(x_939);  x_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    x_comb_iter_4_right_13 = self.L__mod___cell_11_conv_1x1_bn(x_940);  x_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_942 = self.L__mod___cell_11_comb_iter_0_left_act_1(x_left_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_943 = self.L__mod___cell_11_comb_iter_0_left_separable_1_depthwise_conv2d(x_942);  x_942 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_945 = self.L__mod___cell_11_comb_iter_0_left_separable_1_pointwise_conv2d(x_943);  x_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_946 = self.L__mod___cell_11_comb_iter_0_left_bn_sep_1(x_945);  x_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_947 = self.L__mod___cell_11_comb_iter_0_left_act_2(x_946);  x_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_948 = self.L__mod___cell_11_comb_iter_0_left_separable_2_depthwise_conv2d(x_947);  x_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_950 = self.L__mod___cell_11_comb_iter_0_left_separable_2_pointwise_conv2d(x_948);  x_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_0_left_13 = self.L__mod___cell_11_comb_iter_0_left_bn_sep_2(x_950);  x_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0_right_13 = self.L__mod___cell_11_comb_iter_0_right(x_left_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:107, code: x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
    x_comb_iter_65 = x_comb_iter_0_left_13 + x_comb_iter_0_right_13;  x_comb_iter_0_left_13 = x_comb_iter_0_right_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_952 = self.L__mod___cell_11_comb_iter_1_left_act_1(x_comb_iter_4_right_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_953 = self.L__mod___cell_11_comb_iter_1_left_separable_1_depthwise_conv2d(x_952);  x_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_955 = self.L__mod___cell_11_comb_iter_1_left_separable_1_pointwise_conv2d(x_953);  x_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_956 = self.L__mod___cell_11_comb_iter_1_left_bn_sep_1(x_955);  x_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_957 = self.L__mod___cell_11_comb_iter_1_left_act_2(x_956);  x_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_958 = self.L__mod___cell_11_comb_iter_1_left_separable_2_depthwise_conv2d(x_957);  x_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_960 = self.L__mod___cell_11_comb_iter_1_left_separable_2_pointwise_conv2d(x_958);  x_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_1_left_13 = self.L__mod___cell_11_comb_iter_1_left_bn_sep_2(x_960);  x_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1_right_13 = self.L__mod___cell_11_comb_iter_1_right(x_comb_iter_4_right_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:111, code: x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
    x_comb_iter_66 = x_comb_iter_1_left_13 + x_comb_iter_1_right_13;  x_comb_iter_1_left_13 = x_comb_iter_1_right_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_962 = self.L__mod___cell_11_comb_iter_2_left_act_1(x_comb_iter_4_right_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_963 = self.L__mod___cell_11_comb_iter_2_left_separable_1_depthwise_conv2d(x_962);  x_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_965 = self.L__mod___cell_11_comb_iter_2_left_separable_1_pointwise_conv2d(x_963);  x_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_966 = self.L__mod___cell_11_comb_iter_2_left_bn_sep_1(x_965);  x_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_967 = self.L__mod___cell_11_comb_iter_2_left_act_2(x_966);  x_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_968 = self.L__mod___cell_11_comb_iter_2_left_separable_2_depthwise_conv2d(x_967);  x_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_970 = self.L__mod___cell_11_comb_iter_2_left_separable_2_pointwise_conv2d(x_968);  x_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_left_13 = self.L__mod___cell_11_comb_iter_2_left_bn_sep_2(x_970);  x_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_972 = self.L__mod___cell_11_comb_iter_2_right_act_1(x_comb_iter_4_right_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_973 = self.L__mod___cell_11_comb_iter_2_right_separable_1_depthwise_conv2d(x_972);  x_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_975 = self.L__mod___cell_11_comb_iter_2_right_separable_1_pointwise_conv2d(x_973);  x_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_976 = self.L__mod___cell_11_comb_iter_2_right_bn_sep_1(x_975);  x_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_977 = self.L__mod___cell_11_comb_iter_2_right_act_2(x_976);  x_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_978 = self.L__mod___cell_11_comb_iter_2_right_separable_2_depthwise_conv2d(x_977);  x_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_980 = self.L__mod___cell_11_comb_iter_2_right_separable_2_pointwise_conv2d(x_978);  x_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_2_right_13 = self.L__mod___cell_11_comb_iter_2_right_bn_sep_2(x_980);  x_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:115, code: x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
    x_comb_iter_67 = x_comb_iter_2_left_13 + x_comb_iter_2_right_13;  x_comb_iter_2_left_13 = x_comb_iter_2_right_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_982 = self.L__mod___cell_11_comb_iter_3_left_act_1(x_comb_iter_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_983 = self.L__mod___cell_11_comb_iter_3_left_separable_1_depthwise_conv2d(x_982);  x_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_985 = self.L__mod___cell_11_comb_iter_3_left_separable_1_pointwise_conv2d(x_983);  x_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_986 = self.L__mod___cell_11_comb_iter_3_left_bn_sep_1(x_985);  x_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_987 = self.L__mod___cell_11_comb_iter_3_left_act_2(x_986);  x_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_988 = self.L__mod___cell_11_comb_iter_3_left_separable_2_depthwise_conv2d(x_987);  x_987 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_990 = self.L__mod___cell_11_comb_iter_3_left_separable_2_pointwise_conv2d(x_988);  x_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_3_left_13 = self.L__mod___cell_11_comb_iter_3_left_bn_sep_2(x_990);  x_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3_right_13 = self.L__mod___cell_11_comb_iter_3_right(x_comb_iter_4_right_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:119, code: x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
    x_comb_iter_68 = x_comb_iter_3_left_13 + x_comb_iter_3_right_13;  x_comb_iter_3_left_13 = x_comb_iter_3_right_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    x_992 = self.L__mod___cell_11_comb_iter_4_left_act_1(x_left_12);  x_left_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_993 = self.L__mod___cell_11_comb_iter_4_left_separable_1_depthwise_conv2d(x_992);  x_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_995 = self.L__mod___cell_11_comb_iter_4_left_separable_1_pointwise_conv2d(x_993);  x_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    x_996 = self.L__mod___cell_11_comb_iter_4_left_bn_sep_1(x_995);  x_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    x_997 = self.L__mod___cell_11_comb_iter_4_left_act_2(x_996);  x_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    x_998 = self.L__mod___cell_11_comb_iter_4_left_separable_2_depthwise_conv2d(x_997);  x_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    x_1000 = self.L__mod___cell_11_comb_iter_4_left_separable_2_pointwise_conv2d(x_998);  x_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    x_comb_iter_4_left_13 = self.L__mod___cell_11_comb_iter_4_left_bn_sep_2(x_1000);  x_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:126, code: x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
    x_comb_iter_69 = x_comb_iter_4_left_13 + x_comb_iter_4_right_13;  x_comb_iter_4_left_13 = x_comb_iter_4_right_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    x_cell_11 = torch.cat([x_comb_iter_65, x_comb_iter_66, x_comb_iter_67, x_comb_iter_68, x_comb_iter_69], 1);  x_comb_iter_65 = x_comb_iter_66 = x_comb_iter_67 = x_comb_iter_68 = x_comb_iter_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:331, code: x = self.act(x_cell_11)
    x_1003 = self.L__mod___act(x_cell_11);  x_cell_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_1004 = self.L__mod___global_pool_pool(x_1003);  x_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_1006 = self.L__mod___global_pool_flatten(x_1004);  x_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:336, code: x = self.head_drop(x)
    x_1007 = self.L__mod___head_drop(x_1006);  x_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:337, code: return x if pre_logits else self.last_linear(x)
    pred = self.L__mod___last_linear(x_1007);  x_1007 = None
    return (pred,)
    