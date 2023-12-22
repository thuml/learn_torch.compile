from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:363, code: x = self.base_layer(x)
    l__mod___base_layer_0 = self.L__mod___base_layer_0(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    l__mod___base_layer_1 = self.L__mod___base_layer_1(l__mod___base_layer_0);  l__mod___base_layer_0 = None
    x = self.L__mod___base_layer_2(l__mod___base_layer_1);  l__mod___base_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:364, code: x = self.level0(x)
    l__mod___level0_0 = self.L__mod___level0_0(x);  x = None
    l__mod___level0_1 = self.L__mod___level0_1(l__mod___level0_0);  l__mod___level0_0 = None
    x_1 = self.L__mod___level0_2(l__mod___level0_1);  l__mod___level0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:365, code: x = self.level1(x)
    l__mod___level1_0 = self.L__mod___level1_0(x_1);  x_1 = None
    l__mod___level1_1 = self.L__mod___level1_1(l__mod___level1_0);  l__mod___level1_0 = None
    x_2 = self.L__mod___level1_2(l__mod___level1_1);  l__mod___level1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom = self.L__mod___level2_downsample(x_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    l__mod___level2_project_0 = self.L__mod___level2_project_0(bottom);  bottom = None
    shortcut = self.L__mod___level2_project_1(l__mod___level2_project_0);  l__mod___level2_project_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out = self.L__mod___level2_tree1_conv1(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_1 = self.L__mod___level2_tree1_bn1(out);  out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_2 = self.L__mod___level2_tree1_relu(out_1);  out_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_3 = self.L__mod___level2_tree1_conv2(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_4 = self.L__mod___level2_tree1_bn2(out_3);  out_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_5 = self.L__mod___level2_tree1_relu(out_4);  out_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_6 = self.L__mod___level2_tree1_conv3(out_5);  out_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_7 = self.L__mod___level2_tree1_bn3(out_6);  out_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_7 += shortcut;  out_8 = out_7;  out_7 = shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_1 = self.L__mod___level2_tree1_relu(out_8);  out_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_10 = self.L__mod___level2_tree2_conv1(shortcut_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_11 = self.L__mod___level2_tree2_bn1(out_10);  out_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_12 = self.L__mod___level2_tree2_relu(out_11);  out_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_13 = self.L__mod___level2_tree2_conv2(out_12);  out_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_14 = self.L__mod___level2_tree2_bn2(out_13);  out_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_15 = self.L__mod___level2_tree2_relu(out_14);  out_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_16 = self.L__mod___level2_tree2_conv3(out_15);  out_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_17 = self.L__mod___level2_tree2_bn3(out_16);  out_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_17 += shortcut_1;  out_18 = out_17;  out_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2 = self.L__mod___level2_tree2_relu(out_18);  out_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat = torch.cat([x2, shortcut_1], 1);  shortcut_1 = None
    x_3 = self.L__mod___level2_root_conv(cat);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_4 = self.L__mod___level2_root_bn(x_3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_4 += x2;  x_5 = x_4;  x_4 = x2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x_8 = self.L__mod___level2_root_relu(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_1 = self.L__mod___level3_downsample(x_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_2 = self.L__mod___level3_project(bottom_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_2 = self.L__mod___level3_tree1_downsample(x_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_3 = self.L__mod___level3_tree1_project(bottom_2);  bottom_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_3 = self.L__mod___level3_tree1_tree1_downsample(x_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    l__mod___level3_tree1_tree1_project_0 = self.L__mod___level3_tree1_tree1_project_0(bottom_3);  bottom_3 = None
    shortcut_4 = self.L__mod___level3_tree1_tree1_project_1(l__mod___level3_tree1_tree1_project_0);  l__mod___level3_tree1_tree1_project_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_20 = self.L__mod___level3_tree1_tree1_tree1_conv1(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_21 = self.L__mod___level3_tree1_tree1_tree1_bn1(out_20);  out_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_22 = self.L__mod___level3_tree1_tree1_tree1_relu(out_21);  out_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_23 = self.L__mod___level3_tree1_tree1_tree1_conv2(out_22);  out_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_24 = self.L__mod___level3_tree1_tree1_tree1_bn2(out_23);  out_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_25 = self.L__mod___level3_tree1_tree1_tree1_relu(out_24);  out_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_26 = self.L__mod___level3_tree1_tree1_tree1_conv3(out_25);  out_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_27 = self.L__mod___level3_tree1_tree1_tree1_bn3(out_26);  out_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_27 += shortcut_4;  out_28 = out_27;  out_27 = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_5 = self.L__mod___level3_tree1_tree1_tree1_relu(out_28);  out_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_30 = self.L__mod___level3_tree1_tree1_tree2_conv1(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_31 = self.L__mod___level3_tree1_tree1_tree2_bn1(out_30);  out_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_32 = self.L__mod___level3_tree1_tree1_tree2_relu(out_31);  out_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_33 = self.L__mod___level3_tree1_tree1_tree2_conv2(out_32);  out_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_34 = self.L__mod___level3_tree1_tree1_tree2_bn2(out_33);  out_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_35 = self.L__mod___level3_tree1_tree1_tree2_relu(out_34);  out_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_36 = self.L__mod___level3_tree1_tree1_tree2_conv3(out_35);  out_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_37 = self.L__mod___level3_tree1_tree1_tree2_bn3(out_36);  out_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_37 += shortcut_5;  out_38 = out_37;  out_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_1 = self.L__mod___level3_tree1_tree1_tree2_relu(out_38);  out_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_1 = torch.cat([x2_1, shortcut_5], 1);  shortcut_5 = None
    x_9 = self.L__mod___level3_tree1_tree1_root_conv(cat_1);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_10 = self.L__mod___level3_tree1_tree1_root_bn(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_10 += x2_1;  x_11 = x_10;  x_10 = x2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_2 = self.L__mod___level3_tree1_tree1_root_relu(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_4 = self.L__mod___level3_tree1_tree2_downsample(x1_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_6 = self.L__mod___level3_tree1_tree2_project(bottom_4);  bottom_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_40 = self.L__mod___level3_tree1_tree2_tree1_conv1(x1_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_41 = self.L__mod___level3_tree1_tree2_tree1_bn1(out_40);  out_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_42 = self.L__mod___level3_tree1_tree2_tree1_relu(out_41);  out_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_43 = self.L__mod___level3_tree1_tree2_tree1_conv2(out_42);  out_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_44 = self.L__mod___level3_tree1_tree2_tree1_bn2(out_43);  out_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_45 = self.L__mod___level3_tree1_tree2_tree1_relu(out_44);  out_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_46 = self.L__mod___level3_tree1_tree2_tree1_conv3(out_45);  out_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_47 = self.L__mod___level3_tree1_tree2_tree1_bn3(out_46);  out_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_47 += shortcut_6;  out_48 = out_47;  out_47 = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_7 = self.L__mod___level3_tree1_tree2_tree1_relu(out_48);  out_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_50 = self.L__mod___level3_tree1_tree2_tree2_conv1(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_51 = self.L__mod___level3_tree1_tree2_tree2_bn1(out_50);  out_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_52 = self.L__mod___level3_tree1_tree2_tree2_relu(out_51);  out_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_53 = self.L__mod___level3_tree1_tree2_tree2_conv2(out_52);  out_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_54 = self.L__mod___level3_tree1_tree2_tree2_bn2(out_53);  out_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_55 = self.L__mod___level3_tree1_tree2_tree2_relu(out_54);  out_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_56 = self.L__mod___level3_tree1_tree2_tree2_conv3(out_55);  out_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_57 = self.L__mod___level3_tree1_tree2_tree2_bn3(out_56);  out_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_57 += shortcut_7;  out_58 = out_57;  out_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_2 = self.L__mod___level3_tree1_tree2_tree2_relu(out_58);  out_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_2 = torch.cat([x2_2, shortcut_7, x1_2], 1);  shortcut_7 = x1_2 = None
    x_14 = self.L__mod___level3_tree1_tree2_root_conv(cat_2);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_15 = self.L__mod___level3_tree1_tree2_root_bn(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_15 += x2_2;  x_16 = x_15;  x_15 = x2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_4 = self.L__mod___level3_tree1_tree2_root_relu(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_5 = self.L__mod___level3_tree2_downsample(x1_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_8 = self.L__mod___level3_tree2_project(bottom_5);  bottom_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_6 = self.L__mod___level3_tree2_tree1_downsample(x1_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_9 = self.L__mod___level3_tree2_tree1_project(bottom_6);  bottom_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_60 = self.L__mod___level3_tree2_tree1_tree1_conv1(x1_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_61 = self.L__mod___level3_tree2_tree1_tree1_bn1(out_60);  out_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_62 = self.L__mod___level3_tree2_tree1_tree1_relu(out_61);  out_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_63 = self.L__mod___level3_tree2_tree1_tree1_conv2(out_62);  out_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_64 = self.L__mod___level3_tree2_tree1_tree1_bn2(out_63);  out_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_65 = self.L__mod___level3_tree2_tree1_tree1_relu(out_64);  out_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_66 = self.L__mod___level3_tree2_tree1_tree1_conv3(out_65);  out_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_67 = self.L__mod___level3_tree2_tree1_tree1_bn3(out_66);  out_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_67 += shortcut_9;  out_68 = out_67;  out_67 = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_10 = self.L__mod___level3_tree2_tree1_tree1_relu(out_68);  out_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_70 = self.L__mod___level3_tree2_tree1_tree2_conv1(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_71 = self.L__mod___level3_tree2_tree1_tree2_bn1(out_70);  out_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_72 = self.L__mod___level3_tree2_tree1_tree2_relu(out_71);  out_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_73 = self.L__mod___level3_tree2_tree1_tree2_conv2(out_72);  out_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_74 = self.L__mod___level3_tree2_tree1_tree2_bn2(out_73);  out_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_75 = self.L__mod___level3_tree2_tree1_tree2_relu(out_74);  out_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_76 = self.L__mod___level3_tree2_tree1_tree2_conv3(out_75);  out_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_77 = self.L__mod___level3_tree2_tree1_tree2_bn3(out_76);  out_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_77 += shortcut_10;  out_78 = out_77;  out_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_3 = self.L__mod___level3_tree2_tree1_tree2_relu(out_78);  out_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_3 = torch.cat([x2_3, shortcut_10], 1);  shortcut_10 = None
    x_20 = self.L__mod___level3_tree2_tree1_root_conv(cat_3);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_21 = self.L__mod___level3_tree2_tree1_root_bn(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_21 += x2_3;  x_22 = x_21;  x_21 = x2_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_6 = self.L__mod___level3_tree2_tree1_root_relu(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_7 = self.L__mod___level3_tree2_tree2_downsample(x1_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_11 = self.L__mod___level3_tree2_tree2_project(bottom_7);  bottom_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_80 = self.L__mod___level3_tree2_tree2_tree1_conv1(x1_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_81 = self.L__mod___level3_tree2_tree2_tree1_bn1(out_80);  out_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_82 = self.L__mod___level3_tree2_tree2_tree1_relu(out_81);  out_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_83 = self.L__mod___level3_tree2_tree2_tree1_conv2(out_82);  out_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_84 = self.L__mod___level3_tree2_tree2_tree1_bn2(out_83);  out_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_85 = self.L__mod___level3_tree2_tree2_tree1_relu(out_84);  out_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_86 = self.L__mod___level3_tree2_tree2_tree1_conv3(out_85);  out_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_87 = self.L__mod___level3_tree2_tree2_tree1_bn3(out_86);  out_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_87 += shortcut_11;  out_88 = out_87;  out_87 = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_12 = self.L__mod___level3_tree2_tree2_tree1_relu(out_88);  out_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_90 = self.L__mod___level3_tree2_tree2_tree2_conv1(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_91 = self.L__mod___level3_tree2_tree2_tree2_bn1(out_90);  out_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_92 = self.L__mod___level3_tree2_tree2_tree2_relu(out_91);  out_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_93 = self.L__mod___level3_tree2_tree2_tree2_conv2(out_92);  out_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_94 = self.L__mod___level3_tree2_tree2_tree2_bn2(out_93);  out_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_95 = self.L__mod___level3_tree2_tree2_tree2_relu(out_94);  out_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_96 = self.L__mod___level3_tree2_tree2_tree2_conv3(out_95);  out_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_97 = self.L__mod___level3_tree2_tree2_tree2_bn3(out_96);  out_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_97 += shortcut_12;  out_98 = out_97;  out_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_4 = self.L__mod___level3_tree2_tree2_tree2_relu(out_98);  out_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_4 = torch.cat([x2_4, shortcut_12, bottom_1, x1_4, x1_6], 1);  shortcut_12 = bottom_1 = x1_4 = x1_6 = None
    x_25 = self.L__mod___level3_tree2_tree2_root_conv(cat_4);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_26 = self.L__mod___level3_tree2_tree2_root_bn(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_26 += x2_4;  x_27 = x_26;  x_26 = x2_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x_32 = self.L__mod___level3_tree2_tree2_root_relu(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_8 = self.L__mod___level4_downsample(x_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_13 = self.L__mod___level4_project(bottom_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_9 = self.L__mod___level4_tree1_downsample(x_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_14 = self.L__mod___level4_tree1_project(bottom_9);  bottom_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_10 = self.L__mod___level4_tree1_tree1_downsample(x_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_15 = self.L__mod___level4_tree1_tree1_project(bottom_10);  bottom_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_11 = self.L__mod___level4_tree1_tree1_tree1_downsample(x_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    l__mod___level4_tree1_tree1_tree1_project_0 = self.L__mod___level4_tree1_tree1_tree1_project_0(bottom_11);  bottom_11 = None
    shortcut_16 = self.L__mod___level4_tree1_tree1_tree1_project_1(l__mod___level4_tree1_tree1_tree1_project_0);  l__mod___level4_tree1_tree1_tree1_project_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_100 = self.L__mod___level4_tree1_tree1_tree1_tree1_conv1(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_101 = self.L__mod___level4_tree1_tree1_tree1_tree1_bn1(out_100);  out_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_102 = self.L__mod___level4_tree1_tree1_tree1_tree1_relu(out_101);  out_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_103 = self.L__mod___level4_tree1_tree1_tree1_tree1_conv2(out_102);  out_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_104 = self.L__mod___level4_tree1_tree1_tree1_tree1_bn2(out_103);  out_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_105 = self.L__mod___level4_tree1_tree1_tree1_tree1_relu(out_104);  out_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_106 = self.L__mod___level4_tree1_tree1_tree1_tree1_conv3(out_105);  out_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_107 = self.L__mod___level4_tree1_tree1_tree1_tree1_bn3(out_106);  out_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_107 += shortcut_16;  out_108 = out_107;  out_107 = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_17 = self.L__mod___level4_tree1_tree1_tree1_tree1_relu(out_108);  out_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_110 = self.L__mod___level4_tree1_tree1_tree1_tree2_conv1(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_111 = self.L__mod___level4_tree1_tree1_tree1_tree2_bn1(out_110);  out_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_112 = self.L__mod___level4_tree1_tree1_tree1_tree2_relu(out_111);  out_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_113 = self.L__mod___level4_tree1_tree1_tree1_tree2_conv2(out_112);  out_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_114 = self.L__mod___level4_tree1_tree1_tree1_tree2_bn2(out_113);  out_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_115 = self.L__mod___level4_tree1_tree1_tree1_tree2_relu(out_114);  out_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_116 = self.L__mod___level4_tree1_tree1_tree1_tree2_conv3(out_115);  out_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_117 = self.L__mod___level4_tree1_tree1_tree1_tree2_bn3(out_116);  out_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_117 += shortcut_17;  out_118 = out_117;  out_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_5 = self.L__mod___level4_tree1_tree1_tree1_tree2_relu(out_118);  out_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_5 = torch.cat([x2_5, shortcut_17], 1);  shortcut_17 = None
    x_33 = self.L__mod___level4_tree1_tree1_tree1_root_conv(cat_5);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_34 = self.L__mod___level4_tree1_tree1_tree1_root_bn(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_34 += x2_5;  x_35 = x_34;  x_34 = x2_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_9 = self.L__mod___level4_tree1_tree1_tree1_root_relu(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_12 = self.L__mod___level4_tree1_tree1_tree2_downsample(x1_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_18 = self.L__mod___level4_tree1_tree1_tree2_project(bottom_12);  bottom_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_120 = self.L__mod___level4_tree1_tree1_tree2_tree1_conv1(x1_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_121 = self.L__mod___level4_tree1_tree1_tree2_tree1_bn1(out_120);  out_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_122 = self.L__mod___level4_tree1_tree1_tree2_tree1_relu(out_121);  out_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_123 = self.L__mod___level4_tree1_tree1_tree2_tree1_conv2(out_122);  out_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_124 = self.L__mod___level4_tree1_tree1_tree2_tree1_bn2(out_123);  out_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_125 = self.L__mod___level4_tree1_tree1_tree2_tree1_relu(out_124);  out_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_126 = self.L__mod___level4_tree1_tree1_tree2_tree1_conv3(out_125);  out_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_127 = self.L__mod___level4_tree1_tree1_tree2_tree1_bn3(out_126);  out_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_127 += shortcut_18;  out_128 = out_127;  out_127 = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_19 = self.L__mod___level4_tree1_tree1_tree2_tree1_relu(out_128);  out_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_130 = self.L__mod___level4_tree1_tree1_tree2_tree2_conv1(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_131 = self.L__mod___level4_tree1_tree1_tree2_tree2_bn1(out_130);  out_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_132 = self.L__mod___level4_tree1_tree1_tree2_tree2_relu(out_131);  out_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_133 = self.L__mod___level4_tree1_tree1_tree2_tree2_conv2(out_132);  out_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_134 = self.L__mod___level4_tree1_tree1_tree2_tree2_bn2(out_133);  out_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_135 = self.L__mod___level4_tree1_tree1_tree2_tree2_relu(out_134);  out_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_136 = self.L__mod___level4_tree1_tree1_tree2_tree2_conv3(out_135);  out_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_137 = self.L__mod___level4_tree1_tree1_tree2_tree2_bn3(out_136);  out_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_137 += shortcut_19;  out_138 = out_137;  out_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_6 = self.L__mod___level4_tree1_tree1_tree2_tree2_relu(out_138);  out_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_6 = torch.cat([x2_6, shortcut_19, x1_9], 1);  shortcut_19 = x1_9 = None
    x_38 = self.L__mod___level4_tree1_tree1_tree2_root_conv(cat_6);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_39 = self.L__mod___level4_tree1_tree1_tree2_root_bn(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_39 += x2_6;  x_40 = x_39;  x_39 = x2_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_11 = self.L__mod___level4_tree1_tree1_tree2_root_relu(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_13 = self.L__mod___level4_tree1_tree2_downsample(x1_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_20 = self.L__mod___level4_tree1_tree2_project(bottom_13);  bottom_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_14 = self.L__mod___level4_tree1_tree2_tree1_downsample(x1_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_21 = self.L__mod___level4_tree1_tree2_tree1_project(bottom_14);  bottom_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_140 = self.L__mod___level4_tree1_tree2_tree1_tree1_conv1(x1_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_141 = self.L__mod___level4_tree1_tree2_tree1_tree1_bn1(out_140);  out_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_142 = self.L__mod___level4_tree1_tree2_tree1_tree1_relu(out_141);  out_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_143 = self.L__mod___level4_tree1_tree2_tree1_tree1_conv2(out_142);  out_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_144 = self.L__mod___level4_tree1_tree2_tree1_tree1_bn2(out_143);  out_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_145 = self.L__mod___level4_tree1_tree2_tree1_tree1_relu(out_144);  out_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_146 = self.L__mod___level4_tree1_tree2_tree1_tree1_conv3(out_145);  out_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_147 = self.L__mod___level4_tree1_tree2_tree1_tree1_bn3(out_146);  out_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_147 += shortcut_21;  out_148 = out_147;  out_147 = shortcut_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_22 = self.L__mod___level4_tree1_tree2_tree1_tree1_relu(out_148);  out_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_150 = self.L__mod___level4_tree1_tree2_tree1_tree2_conv1(shortcut_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_151 = self.L__mod___level4_tree1_tree2_tree1_tree2_bn1(out_150);  out_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_152 = self.L__mod___level4_tree1_tree2_tree1_tree2_relu(out_151);  out_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_153 = self.L__mod___level4_tree1_tree2_tree1_tree2_conv2(out_152);  out_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_154 = self.L__mod___level4_tree1_tree2_tree1_tree2_bn2(out_153);  out_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_155 = self.L__mod___level4_tree1_tree2_tree1_tree2_relu(out_154);  out_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_156 = self.L__mod___level4_tree1_tree2_tree1_tree2_conv3(out_155);  out_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_157 = self.L__mod___level4_tree1_tree2_tree1_tree2_bn3(out_156);  out_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_157 += shortcut_22;  out_158 = out_157;  out_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_7 = self.L__mod___level4_tree1_tree2_tree1_tree2_relu(out_158);  out_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_7 = torch.cat([x2_7, shortcut_22], 1);  shortcut_22 = None
    x_44 = self.L__mod___level4_tree1_tree2_tree1_root_conv(cat_7);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_45 = self.L__mod___level4_tree1_tree2_tree1_root_bn(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_45 += x2_7;  x_46 = x_45;  x_45 = x2_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_13 = self.L__mod___level4_tree1_tree2_tree1_root_relu(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_15 = self.L__mod___level4_tree1_tree2_tree2_downsample(x1_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_23 = self.L__mod___level4_tree1_tree2_tree2_project(bottom_15);  bottom_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_160 = self.L__mod___level4_tree1_tree2_tree2_tree1_conv1(x1_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_161 = self.L__mod___level4_tree1_tree2_tree2_tree1_bn1(out_160);  out_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_162 = self.L__mod___level4_tree1_tree2_tree2_tree1_relu(out_161);  out_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_163 = self.L__mod___level4_tree1_tree2_tree2_tree1_conv2(out_162);  out_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_164 = self.L__mod___level4_tree1_tree2_tree2_tree1_bn2(out_163);  out_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_165 = self.L__mod___level4_tree1_tree2_tree2_tree1_relu(out_164);  out_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_166 = self.L__mod___level4_tree1_tree2_tree2_tree1_conv3(out_165);  out_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_167 = self.L__mod___level4_tree1_tree2_tree2_tree1_bn3(out_166);  out_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_167 += shortcut_23;  out_168 = out_167;  out_167 = shortcut_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_24 = self.L__mod___level4_tree1_tree2_tree2_tree1_relu(out_168);  out_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_170 = self.L__mod___level4_tree1_tree2_tree2_tree2_conv1(shortcut_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_171 = self.L__mod___level4_tree1_tree2_tree2_tree2_bn1(out_170);  out_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_172 = self.L__mod___level4_tree1_tree2_tree2_tree2_relu(out_171);  out_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_173 = self.L__mod___level4_tree1_tree2_tree2_tree2_conv2(out_172);  out_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_174 = self.L__mod___level4_tree1_tree2_tree2_tree2_bn2(out_173);  out_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_175 = self.L__mod___level4_tree1_tree2_tree2_tree2_relu(out_174);  out_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_176 = self.L__mod___level4_tree1_tree2_tree2_tree2_conv3(out_175);  out_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_177 = self.L__mod___level4_tree1_tree2_tree2_tree2_bn3(out_176);  out_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_177 += shortcut_24;  out_178 = out_177;  out_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_8 = self.L__mod___level4_tree1_tree2_tree2_tree2_relu(out_178);  out_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_8 = torch.cat([x2_8, shortcut_24, x1_11, x1_13], 1);  shortcut_24 = x1_11 = x1_13 = None
    x_49 = self.L__mod___level4_tree1_tree2_tree2_root_conv(cat_8);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_50 = self.L__mod___level4_tree1_tree2_tree2_root_bn(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_50 += x2_8;  x_51 = x_50;  x_50 = x2_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_15 = self.L__mod___level4_tree1_tree2_tree2_root_relu(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_16 = self.L__mod___level4_tree2_downsample(x1_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_25 = self.L__mod___level4_tree2_project(bottom_16);  bottom_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_17 = self.L__mod___level4_tree2_tree1_downsample(x1_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_26 = self.L__mod___level4_tree2_tree1_project(bottom_17);  bottom_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_18 = self.L__mod___level4_tree2_tree1_tree1_downsample(x1_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_27 = self.L__mod___level4_tree2_tree1_tree1_project(bottom_18);  bottom_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_180 = self.L__mod___level4_tree2_tree1_tree1_tree1_conv1(x1_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_181 = self.L__mod___level4_tree2_tree1_tree1_tree1_bn1(out_180);  out_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_182 = self.L__mod___level4_tree2_tree1_tree1_tree1_relu(out_181);  out_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_183 = self.L__mod___level4_tree2_tree1_tree1_tree1_conv2(out_182);  out_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_184 = self.L__mod___level4_tree2_tree1_tree1_tree1_bn2(out_183);  out_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_185 = self.L__mod___level4_tree2_tree1_tree1_tree1_relu(out_184);  out_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_186 = self.L__mod___level4_tree2_tree1_tree1_tree1_conv3(out_185);  out_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_187 = self.L__mod___level4_tree2_tree1_tree1_tree1_bn3(out_186);  out_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_187 += shortcut_27;  out_188 = out_187;  out_187 = shortcut_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_28 = self.L__mod___level4_tree2_tree1_tree1_tree1_relu(out_188);  out_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_190 = self.L__mod___level4_tree2_tree1_tree1_tree2_conv1(shortcut_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_191 = self.L__mod___level4_tree2_tree1_tree1_tree2_bn1(out_190);  out_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_192 = self.L__mod___level4_tree2_tree1_tree1_tree2_relu(out_191);  out_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_193 = self.L__mod___level4_tree2_tree1_tree1_tree2_conv2(out_192);  out_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_194 = self.L__mod___level4_tree2_tree1_tree1_tree2_bn2(out_193);  out_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_195 = self.L__mod___level4_tree2_tree1_tree1_tree2_relu(out_194);  out_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_196 = self.L__mod___level4_tree2_tree1_tree1_tree2_conv3(out_195);  out_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_197 = self.L__mod___level4_tree2_tree1_tree1_tree2_bn3(out_196);  out_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_197 += shortcut_28;  out_198 = out_197;  out_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_9 = self.L__mod___level4_tree2_tree1_tree1_tree2_relu(out_198);  out_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_9 = torch.cat([x2_9, shortcut_28], 1);  shortcut_28 = None
    x_56 = self.L__mod___level4_tree2_tree1_tree1_root_conv(cat_9);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_57 = self.L__mod___level4_tree2_tree1_tree1_root_bn(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_57 += x2_9;  x_58 = x_57;  x_57 = x2_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_17 = self.L__mod___level4_tree2_tree1_tree1_root_relu(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_19 = self.L__mod___level4_tree2_tree1_tree2_downsample(x1_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_29 = self.L__mod___level4_tree2_tree1_tree2_project(bottom_19);  bottom_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_200 = self.L__mod___level4_tree2_tree1_tree2_tree1_conv1(x1_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_201 = self.L__mod___level4_tree2_tree1_tree2_tree1_bn1(out_200);  out_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_202 = self.L__mod___level4_tree2_tree1_tree2_tree1_relu(out_201);  out_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_203 = self.L__mod___level4_tree2_tree1_tree2_tree1_conv2(out_202);  out_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_204 = self.L__mod___level4_tree2_tree1_tree2_tree1_bn2(out_203);  out_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_205 = self.L__mod___level4_tree2_tree1_tree2_tree1_relu(out_204);  out_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_206 = self.L__mod___level4_tree2_tree1_tree2_tree1_conv3(out_205);  out_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_207 = self.L__mod___level4_tree2_tree1_tree2_tree1_bn3(out_206);  out_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_207 += shortcut_29;  out_208 = out_207;  out_207 = shortcut_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_30 = self.L__mod___level4_tree2_tree1_tree2_tree1_relu(out_208);  out_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_210 = self.L__mod___level4_tree2_tree1_tree2_tree2_conv1(shortcut_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_211 = self.L__mod___level4_tree2_tree1_tree2_tree2_bn1(out_210);  out_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_212 = self.L__mod___level4_tree2_tree1_tree2_tree2_relu(out_211);  out_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_213 = self.L__mod___level4_tree2_tree1_tree2_tree2_conv2(out_212);  out_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_214 = self.L__mod___level4_tree2_tree1_tree2_tree2_bn2(out_213);  out_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_215 = self.L__mod___level4_tree2_tree1_tree2_tree2_relu(out_214);  out_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_216 = self.L__mod___level4_tree2_tree1_tree2_tree2_conv3(out_215);  out_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_217 = self.L__mod___level4_tree2_tree1_tree2_tree2_bn3(out_216);  out_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_217 += shortcut_30;  out_218 = out_217;  out_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_10 = self.L__mod___level4_tree2_tree1_tree2_tree2_relu(out_218);  out_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_10 = torch.cat([x2_10, shortcut_30, x1_17], 1);  shortcut_30 = x1_17 = None
    x_61 = self.L__mod___level4_tree2_tree1_tree2_root_conv(cat_10);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_62 = self.L__mod___level4_tree2_tree1_tree2_root_bn(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_62 += x2_10;  x_63 = x_62;  x_62 = x2_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_19 = self.L__mod___level4_tree2_tree1_tree2_root_relu(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_20 = self.L__mod___level4_tree2_tree2_downsample(x1_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_31 = self.L__mod___level4_tree2_tree2_project(bottom_20);  bottom_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_21 = self.L__mod___level4_tree2_tree2_tree1_downsample(x1_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_32 = self.L__mod___level4_tree2_tree2_tree1_project(bottom_21);  bottom_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_220 = self.L__mod___level4_tree2_tree2_tree1_tree1_conv1(x1_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_221 = self.L__mod___level4_tree2_tree2_tree1_tree1_bn1(out_220);  out_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_222 = self.L__mod___level4_tree2_tree2_tree1_tree1_relu(out_221);  out_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_223 = self.L__mod___level4_tree2_tree2_tree1_tree1_conv2(out_222);  out_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_224 = self.L__mod___level4_tree2_tree2_tree1_tree1_bn2(out_223);  out_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_225 = self.L__mod___level4_tree2_tree2_tree1_tree1_relu(out_224);  out_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_226 = self.L__mod___level4_tree2_tree2_tree1_tree1_conv3(out_225);  out_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_227 = self.L__mod___level4_tree2_tree2_tree1_tree1_bn3(out_226);  out_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_227 += shortcut_32;  out_228 = out_227;  out_227 = shortcut_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_33 = self.L__mod___level4_tree2_tree2_tree1_tree1_relu(out_228);  out_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_230 = self.L__mod___level4_tree2_tree2_tree1_tree2_conv1(shortcut_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_231 = self.L__mod___level4_tree2_tree2_tree1_tree2_bn1(out_230);  out_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_232 = self.L__mod___level4_tree2_tree2_tree1_tree2_relu(out_231);  out_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_233 = self.L__mod___level4_tree2_tree2_tree1_tree2_conv2(out_232);  out_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_234 = self.L__mod___level4_tree2_tree2_tree1_tree2_bn2(out_233);  out_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_235 = self.L__mod___level4_tree2_tree2_tree1_tree2_relu(out_234);  out_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_236 = self.L__mod___level4_tree2_tree2_tree1_tree2_conv3(out_235);  out_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_237 = self.L__mod___level4_tree2_tree2_tree1_tree2_bn3(out_236);  out_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_237 += shortcut_33;  out_238 = out_237;  out_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_11 = self.L__mod___level4_tree2_tree2_tree1_tree2_relu(out_238);  out_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_11 = torch.cat([x2_11, shortcut_33], 1);  shortcut_33 = None
    x_67 = self.L__mod___level4_tree2_tree2_tree1_root_conv(cat_11);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_68 = self.L__mod___level4_tree2_tree2_tree1_root_bn(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_68 += x2_11;  x_69 = x_68;  x_68 = x2_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x1_21 = self.L__mod___level4_tree2_tree2_tree1_root_relu(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_22 = self.L__mod___level4_tree2_tree2_tree2_downsample(x1_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    shortcut_34 = self.L__mod___level4_tree2_tree2_tree2_project(bottom_22);  bottom_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_240 = self.L__mod___level4_tree2_tree2_tree2_tree1_conv1(x1_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_241 = self.L__mod___level4_tree2_tree2_tree2_tree1_bn1(out_240);  out_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_242 = self.L__mod___level4_tree2_tree2_tree2_tree1_relu(out_241);  out_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_243 = self.L__mod___level4_tree2_tree2_tree2_tree1_conv2(out_242);  out_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_244 = self.L__mod___level4_tree2_tree2_tree2_tree1_bn2(out_243);  out_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_245 = self.L__mod___level4_tree2_tree2_tree2_tree1_relu(out_244);  out_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_246 = self.L__mod___level4_tree2_tree2_tree2_tree1_conv3(out_245);  out_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_247 = self.L__mod___level4_tree2_tree2_tree2_tree1_bn3(out_246);  out_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_247 += shortcut_34;  out_248 = out_247;  out_247 = shortcut_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_35 = self.L__mod___level4_tree2_tree2_tree2_tree1_relu(out_248);  out_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_250 = self.L__mod___level4_tree2_tree2_tree2_tree2_conv1(shortcut_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_251 = self.L__mod___level4_tree2_tree2_tree2_tree2_bn1(out_250);  out_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_252 = self.L__mod___level4_tree2_tree2_tree2_tree2_relu(out_251);  out_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_253 = self.L__mod___level4_tree2_tree2_tree2_tree2_conv2(out_252);  out_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_254 = self.L__mod___level4_tree2_tree2_tree2_tree2_bn2(out_253);  out_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_255 = self.L__mod___level4_tree2_tree2_tree2_tree2_relu(out_254);  out_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_256 = self.L__mod___level4_tree2_tree2_tree2_tree2_conv3(out_255);  out_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_257 = self.L__mod___level4_tree2_tree2_tree2_tree2_bn3(out_256);  out_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_257 += shortcut_35;  out_258 = out_257;  out_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_12 = self.L__mod___level4_tree2_tree2_tree2_tree2_relu(out_258);  out_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_12 = torch.cat([x2_12, shortcut_35, bottom_8, x1_15, x1_19, x1_21], 1);  shortcut_35 = bottom_8 = x1_15 = x1_19 = x1_21 = None
    x_72 = self.L__mod___level4_tree2_tree2_tree2_root_conv(cat_12);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_73 = self.L__mod___level4_tree2_tree2_tree2_root_bn(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_73 += x2_12;  x_74 = x_73;  x_73 = x2_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x_80 = self.L__mod___level4_tree2_tree2_tree2_root_relu(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    bottom_23 = self.L__mod___level5_downsample(x_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    l__mod___level5_project_0 = self.L__mod___level5_project_0(bottom_23)
    shortcut_36 = self.L__mod___level5_project_1(l__mod___level5_project_0);  l__mod___level5_project_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_260 = self.L__mod___level5_tree1_conv1(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_261 = self.L__mod___level5_tree1_bn1(out_260);  out_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_262 = self.L__mod___level5_tree1_relu(out_261);  out_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_263 = self.L__mod___level5_tree1_conv2(out_262);  out_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_264 = self.L__mod___level5_tree1_bn2(out_263);  out_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_265 = self.L__mod___level5_tree1_relu(out_264);  out_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_266 = self.L__mod___level5_tree1_conv3(out_265);  out_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_267 = self.L__mod___level5_tree1_bn3(out_266);  out_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_267 += shortcut_36;  out_268 = out_267;  out_267 = shortcut_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    shortcut_37 = self.L__mod___level5_tree1_relu(out_268);  out_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    out_270 = self.L__mod___level5_tree2_conv1(shortcut_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    out_271 = self.L__mod___level5_tree2_bn1(out_270);  out_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    out_272 = self.L__mod___level5_tree2_relu(out_271);  out_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    out_273 = self.L__mod___level5_tree2_conv2(out_272);  out_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    out_274 = self.L__mod___level5_tree2_bn2(out_273);  out_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    out_275 = self.L__mod___level5_tree2_relu(out_274);  out_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    out_276 = self.L__mod___level5_tree2_conv3(out_275);  out_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    out_277 = self.L__mod___level5_tree2_bn3(out_276);  out_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    out_277 += shortcut_37;  out_278 = out_277;  out_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    x2_13 = self.L__mod___level5_tree2_relu(out_278);  out_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_13 = torch.cat([x2_13, shortcut_37, bottom_23], 1);  shortcut_37 = bottom_23 = None
    x_81 = self.L__mod___level5_root_conv(cat_13);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    x_82 = self.L__mod___level5_root_bn(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    x_82 += x2_13;  x_83 = x_82;  x_82 = x2_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    x_87 = self.L__mod___level5_root_relu(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_88 = self.L__mod___global_pool_pool(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_90 = self.L__mod___global_pool_flatten(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:374, code: x = self.head_drop(x)
    x_91 = self.L__mod___head_drop(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:377, code: x = self.fc(x)
    x_92 = self.L__mod___fc(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:378, code: return self.flatten(x)
    pred = self.L__mod___flatten(x_92);  x_92 = None
    return (pred,)
    