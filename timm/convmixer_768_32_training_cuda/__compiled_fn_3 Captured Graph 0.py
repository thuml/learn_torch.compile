from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:85, code: x = self.stem(x)
    l__mod___stem_0 = self.L__mod___stem_0(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    l__mod___stem_1 = self.L__mod___stem_1(l__mod___stem_0);  l__mod___stem_0 = None
    x = self.L__mod___stem_2(l__mod___stem_1);  l__mod___stem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___0_____0___fn_0 = self.getattr_getattr_L__mod___blocks___0_____0___fn_0(x)
    getattr_getattr_l__mod___blocks___0_____0___fn_1 = self.getattr_getattr_L__mod___blocks___0_____0___fn_1(getattr_getattr_l__mod___blocks___0_____0___fn_0);  getattr_getattr_l__mod___blocks___0_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___0_____0___fn_2 = self.getattr_getattr_L__mod___blocks___0_____0___fn_2(getattr_getattr_l__mod___blocks___0_____0___fn_1);  getattr_getattr_l__mod___blocks___0_____0___fn_1 = None
    add = getattr_getattr_l__mod___blocks___0_____0___fn_2 + x;  getattr_getattr_l__mod___blocks___0_____0___fn_2 = x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_0_1 = self.L__mod___blocks_0_1(add);  add = None
    l__mod___blocks_0_2 = self.L__mod___blocks_0_2(l__mod___blocks_0_1);  l__mod___blocks_0_1 = None
    l__mod___blocks_0_3 = self.L__mod___blocks_0_3(l__mod___blocks_0_2);  l__mod___blocks_0_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___1_____0___fn_0 = self.getattr_getattr_L__mod___blocks___1_____0___fn_0(l__mod___blocks_0_3)
    getattr_getattr_l__mod___blocks___1_____0___fn_1 = self.getattr_getattr_L__mod___blocks___1_____0___fn_1(getattr_getattr_l__mod___blocks___1_____0___fn_0);  getattr_getattr_l__mod___blocks___1_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___1_____0___fn_2 = self.getattr_getattr_L__mod___blocks___1_____0___fn_2(getattr_getattr_l__mod___blocks___1_____0___fn_1);  getattr_getattr_l__mod___blocks___1_____0___fn_1 = None
    add_1 = getattr_getattr_l__mod___blocks___1_____0___fn_2 + l__mod___blocks_0_3;  getattr_getattr_l__mod___blocks___1_____0___fn_2 = l__mod___blocks_0_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_1_1 = self.L__mod___blocks_1_1(add_1);  add_1 = None
    l__mod___blocks_1_2 = self.L__mod___blocks_1_2(l__mod___blocks_1_1);  l__mod___blocks_1_1 = None
    l__mod___blocks_1_3 = self.L__mod___blocks_1_3(l__mod___blocks_1_2);  l__mod___blocks_1_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___2_____0___fn_0 = self.getattr_getattr_L__mod___blocks___2_____0___fn_0(l__mod___blocks_1_3)
    getattr_getattr_l__mod___blocks___2_____0___fn_1 = self.getattr_getattr_L__mod___blocks___2_____0___fn_1(getattr_getattr_l__mod___blocks___2_____0___fn_0);  getattr_getattr_l__mod___blocks___2_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___2_____0___fn_2 = self.getattr_getattr_L__mod___blocks___2_____0___fn_2(getattr_getattr_l__mod___blocks___2_____0___fn_1);  getattr_getattr_l__mod___blocks___2_____0___fn_1 = None
    add_2 = getattr_getattr_l__mod___blocks___2_____0___fn_2 + l__mod___blocks_1_3;  getattr_getattr_l__mod___blocks___2_____0___fn_2 = l__mod___blocks_1_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_2_1 = self.L__mod___blocks_2_1(add_2);  add_2 = None
    l__mod___blocks_2_2 = self.L__mod___blocks_2_2(l__mod___blocks_2_1);  l__mod___blocks_2_1 = None
    l__mod___blocks_2_3 = self.L__mod___blocks_2_3(l__mod___blocks_2_2);  l__mod___blocks_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___3_____0___fn_0 = self.getattr_getattr_L__mod___blocks___3_____0___fn_0(l__mod___blocks_2_3)
    getattr_getattr_l__mod___blocks___3_____0___fn_1 = self.getattr_getattr_L__mod___blocks___3_____0___fn_1(getattr_getattr_l__mod___blocks___3_____0___fn_0);  getattr_getattr_l__mod___blocks___3_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___3_____0___fn_2 = self.getattr_getattr_L__mod___blocks___3_____0___fn_2(getattr_getattr_l__mod___blocks___3_____0___fn_1);  getattr_getattr_l__mod___blocks___3_____0___fn_1 = None
    add_3 = getattr_getattr_l__mod___blocks___3_____0___fn_2 + l__mod___blocks_2_3;  getattr_getattr_l__mod___blocks___3_____0___fn_2 = l__mod___blocks_2_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_3_1 = self.L__mod___blocks_3_1(add_3);  add_3 = None
    l__mod___blocks_3_2 = self.L__mod___blocks_3_2(l__mod___blocks_3_1);  l__mod___blocks_3_1 = None
    l__mod___blocks_3_3 = self.L__mod___blocks_3_3(l__mod___blocks_3_2);  l__mod___blocks_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___4_____0___fn_0 = self.getattr_getattr_L__mod___blocks___4_____0___fn_0(l__mod___blocks_3_3)
    getattr_getattr_l__mod___blocks___4_____0___fn_1 = self.getattr_getattr_L__mod___blocks___4_____0___fn_1(getattr_getattr_l__mod___blocks___4_____0___fn_0);  getattr_getattr_l__mod___blocks___4_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___4_____0___fn_2 = self.getattr_getattr_L__mod___blocks___4_____0___fn_2(getattr_getattr_l__mod___blocks___4_____0___fn_1);  getattr_getattr_l__mod___blocks___4_____0___fn_1 = None
    add_4 = getattr_getattr_l__mod___blocks___4_____0___fn_2 + l__mod___blocks_3_3;  getattr_getattr_l__mod___blocks___4_____0___fn_2 = l__mod___blocks_3_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_4_1 = self.L__mod___blocks_4_1(add_4);  add_4 = None
    l__mod___blocks_4_2 = self.L__mod___blocks_4_2(l__mod___blocks_4_1);  l__mod___blocks_4_1 = None
    l__mod___blocks_4_3 = self.L__mod___blocks_4_3(l__mod___blocks_4_2);  l__mod___blocks_4_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___5_____0___fn_0 = self.getattr_getattr_L__mod___blocks___5_____0___fn_0(l__mod___blocks_4_3)
    getattr_getattr_l__mod___blocks___5_____0___fn_1 = self.getattr_getattr_L__mod___blocks___5_____0___fn_1(getattr_getattr_l__mod___blocks___5_____0___fn_0);  getattr_getattr_l__mod___blocks___5_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___5_____0___fn_2 = self.getattr_getattr_L__mod___blocks___5_____0___fn_2(getattr_getattr_l__mod___blocks___5_____0___fn_1);  getattr_getattr_l__mod___blocks___5_____0___fn_1 = None
    add_5 = getattr_getattr_l__mod___blocks___5_____0___fn_2 + l__mod___blocks_4_3;  getattr_getattr_l__mod___blocks___5_____0___fn_2 = l__mod___blocks_4_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_5_1 = self.L__mod___blocks_5_1(add_5);  add_5 = None
    l__mod___blocks_5_2 = self.L__mod___blocks_5_2(l__mod___blocks_5_1);  l__mod___blocks_5_1 = None
    l__mod___blocks_5_3 = self.L__mod___blocks_5_3(l__mod___blocks_5_2);  l__mod___blocks_5_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___6_____0___fn_0 = self.getattr_getattr_L__mod___blocks___6_____0___fn_0(l__mod___blocks_5_3)
    getattr_getattr_l__mod___blocks___6_____0___fn_1 = self.getattr_getattr_L__mod___blocks___6_____0___fn_1(getattr_getattr_l__mod___blocks___6_____0___fn_0);  getattr_getattr_l__mod___blocks___6_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___6_____0___fn_2 = self.getattr_getattr_L__mod___blocks___6_____0___fn_2(getattr_getattr_l__mod___blocks___6_____0___fn_1);  getattr_getattr_l__mod___blocks___6_____0___fn_1 = None
    add_6 = getattr_getattr_l__mod___blocks___6_____0___fn_2 + l__mod___blocks_5_3;  getattr_getattr_l__mod___blocks___6_____0___fn_2 = l__mod___blocks_5_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_6_1 = self.L__mod___blocks_6_1(add_6);  add_6 = None
    l__mod___blocks_6_2 = self.L__mod___blocks_6_2(l__mod___blocks_6_1);  l__mod___blocks_6_1 = None
    l__mod___blocks_6_3 = self.L__mod___blocks_6_3(l__mod___blocks_6_2);  l__mod___blocks_6_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___7_____0___fn_0 = self.getattr_getattr_L__mod___blocks___7_____0___fn_0(l__mod___blocks_6_3)
    getattr_getattr_l__mod___blocks___7_____0___fn_1 = self.getattr_getattr_L__mod___blocks___7_____0___fn_1(getattr_getattr_l__mod___blocks___7_____0___fn_0);  getattr_getattr_l__mod___blocks___7_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___7_____0___fn_2 = self.getattr_getattr_L__mod___blocks___7_____0___fn_2(getattr_getattr_l__mod___blocks___7_____0___fn_1);  getattr_getattr_l__mod___blocks___7_____0___fn_1 = None
    add_7 = getattr_getattr_l__mod___blocks___7_____0___fn_2 + l__mod___blocks_6_3;  getattr_getattr_l__mod___blocks___7_____0___fn_2 = l__mod___blocks_6_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_7_1 = self.L__mod___blocks_7_1(add_7);  add_7 = None
    l__mod___blocks_7_2 = self.L__mod___blocks_7_2(l__mod___blocks_7_1);  l__mod___blocks_7_1 = None
    l__mod___blocks_7_3 = self.L__mod___blocks_7_3(l__mod___blocks_7_2);  l__mod___blocks_7_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___8_____0___fn_0 = self.getattr_getattr_L__mod___blocks___8_____0___fn_0(l__mod___blocks_7_3)
    getattr_getattr_l__mod___blocks___8_____0___fn_1 = self.getattr_getattr_L__mod___blocks___8_____0___fn_1(getattr_getattr_l__mod___blocks___8_____0___fn_0);  getattr_getattr_l__mod___blocks___8_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___8_____0___fn_2 = self.getattr_getattr_L__mod___blocks___8_____0___fn_2(getattr_getattr_l__mod___blocks___8_____0___fn_1);  getattr_getattr_l__mod___blocks___8_____0___fn_1 = None
    add_8 = getattr_getattr_l__mod___blocks___8_____0___fn_2 + l__mod___blocks_7_3;  getattr_getattr_l__mod___blocks___8_____0___fn_2 = l__mod___blocks_7_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_8_1 = self.L__mod___blocks_8_1(add_8);  add_8 = None
    l__mod___blocks_8_2 = self.L__mod___blocks_8_2(l__mod___blocks_8_1);  l__mod___blocks_8_1 = None
    l__mod___blocks_8_3 = self.L__mod___blocks_8_3(l__mod___blocks_8_2);  l__mod___blocks_8_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___9_____0___fn_0 = self.getattr_getattr_L__mod___blocks___9_____0___fn_0(l__mod___blocks_8_3)
    getattr_getattr_l__mod___blocks___9_____0___fn_1 = self.getattr_getattr_L__mod___blocks___9_____0___fn_1(getattr_getattr_l__mod___blocks___9_____0___fn_0);  getattr_getattr_l__mod___blocks___9_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___9_____0___fn_2 = self.getattr_getattr_L__mod___blocks___9_____0___fn_2(getattr_getattr_l__mod___blocks___9_____0___fn_1);  getattr_getattr_l__mod___blocks___9_____0___fn_1 = None
    add_9 = getattr_getattr_l__mod___blocks___9_____0___fn_2 + l__mod___blocks_8_3;  getattr_getattr_l__mod___blocks___9_____0___fn_2 = l__mod___blocks_8_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_9_1 = self.L__mod___blocks_9_1(add_9);  add_9 = None
    l__mod___blocks_9_2 = self.L__mod___blocks_9_2(l__mod___blocks_9_1);  l__mod___blocks_9_1 = None
    l__mod___blocks_9_3 = self.L__mod___blocks_9_3(l__mod___blocks_9_2);  l__mod___blocks_9_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___10_____0___fn_0 = self.getattr_getattr_L__mod___blocks___10_____0___fn_0(l__mod___blocks_9_3)
    getattr_getattr_l__mod___blocks___10_____0___fn_1 = self.getattr_getattr_L__mod___blocks___10_____0___fn_1(getattr_getattr_l__mod___blocks___10_____0___fn_0);  getattr_getattr_l__mod___blocks___10_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___10_____0___fn_2 = self.getattr_getattr_L__mod___blocks___10_____0___fn_2(getattr_getattr_l__mod___blocks___10_____0___fn_1);  getattr_getattr_l__mod___blocks___10_____0___fn_1 = None
    add_10 = getattr_getattr_l__mod___blocks___10_____0___fn_2 + l__mod___blocks_9_3;  getattr_getattr_l__mod___blocks___10_____0___fn_2 = l__mod___blocks_9_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_10_1 = self.L__mod___blocks_10_1(add_10);  add_10 = None
    l__mod___blocks_10_2 = self.L__mod___blocks_10_2(l__mod___blocks_10_1);  l__mod___blocks_10_1 = None
    l__mod___blocks_10_3 = self.L__mod___blocks_10_3(l__mod___blocks_10_2);  l__mod___blocks_10_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___11_____0___fn_0 = self.getattr_getattr_L__mod___blocks___11_____0___fn_0(l__mod___blocks_10_3)
    getattr_getattr_l__mod___blocks___11_____0___fn_1 = self.getattr_getattr_L__mod___blocks___11_____0___fn_1(getattr_getattr_l__mod___blocks___11_____0___fn_0);  getattr_getattr_l__mod___blocks___11_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___11_____0___fn_2 = self.getattr_getattr_L__mod___blocks___11_____0___fn_2(getattr_getattr_l__mod___blocks___11_____0___fn_1);  getattr_getattr_l__mod___blocks___11_____0___fn_1 = None
    add_11 = getattr_getattr_l__mod___blocks___11_____0___fn_2 + l__mod___blocks_10_3;  getattr_getattr_l__mod___blocks___11_____0___fn_2 = l__mod___blocks_10_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_11_1 = self.L__mod___blocks_11_1(add_11);  add_11 = None
    l__mod___blocks_11_2 = self.L__mod___blocks_11_2(l__mod___blocks_11_1);  l__mod___blocks_11_1 = None
    l__mod___blocks_11_3 = self.L__mod___blocks_11_3(l__mod___blocks_11_2);  l__mod___blocks_11_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___12_____0___fn_0 = self.getattr_getattr_L__mod___blocks___12_____0___fn_0(l__mod___blocks_11_3)
    getattr_getattr_l__mod___blocks___12_____0___fn_1 = self.getattr_getattr_L__mod___blocks___12_____0___fn_1(getattr_getattr_l__mod___blocks___12_____0___fn_0);  getattr_getattr_l__mod___blocks___12_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___12_____0___fn_2 = self.getattr_getattr_L__mod___blocks___12_____0___fn_2(getattr_getattr_l__mod___blocks___12_____0___fn_1);  getattr_getattr_l__mod___blocks___12_____0___fn_1 = None
    add_12 = getattr_getattr_l__mod___blocks___12_____0___fn_2 + l__mod___blocks_11_3;  getattr_getattr_l__mod___blocks___12_____0___fn_2 = l__mod___blocks_11_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_12_1 = self.L__mod___blocks_12_1(add_12);  add_12 = None
    l__mod___blocks_12_2 = self.L__mod___blocks_12_2(l__mod___blocks_12_1);  l__mod___blocks_12_1 = None
    l__mod___blocks_12_3 = self.L__mod___blocks_12_3(l__mod___blocks_12_2);  l__mod___blocks_12_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___13_____0___fn_0 = self.getattr_getattr_L__mod___blocks___13_____0___fn_0(l__mod___blocks_12_3)
    getattr_getattr_l__mod___blocks___13_____0___fn_1 = self.getattr_getattr_L__mod___blocks___13_____0___fn_1(getattr_getattr_l__mod___blocks___13_____0___fn_0);  getattr_getattr_l__mod___blocks___13_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___13_____0___fn_2 = self.getattr_getattr_L__mod___blocks___13_____0___fn_2(getattr_getattr_l__mod___blocks___13_____0___fn_1);  getattr_getattr_l__mod___blocks___13_____0___fn_1 = None
    add_13 = getattr_getattr_l__mod___blocks___13_____0___fn_2 + l__mod___blocks_12_3;  getattr_getattr_l__mod___blocks___13_____0___fn_2 = l__mod___blocks_12_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_13_1 = self.L__mod___blocks_13_1(add_13);  add_13 = None
    l__mod___blocks_13_2 = self.L__mod___blocks_13_2(l__mod___blocks_13_1);  l__mod___blocks_13_1 = None
    l__mod___blocks_13_3 = self.L__mod___blocks_13_3(l__mod___blocks_13_2);  l__mod___blocks_13_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___14_____0___fn_0 = self.getattr_getattr_L__mod___blocks___14_____0___fn_0(l__mod___blocks_13_3)
    getattr_getattr_l__mod___blocks___14_____0___fn_1 = self.getattr_getattr_L__mod___blocks___14_____0___fn_1(getattr_getattr_l__mod___blocks___14_____0___fn_0);  getattr_getattr_l__mod___blocks___14_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___14_____0___fn_2 = self.getattr_getattr_L__mod___blocks___14_____0___fn_2(getattr_getattr_l__mod___blocks___14_____0___fn_1);  getattr_getattr_l__mod___blocks___14_____0___fn_1 = None
    add_14 = getattr_getattr_l__mod___blocks___14_____0___fn_2 + l__mod___blocks_13_3;  getattr_getattr_l__mod___blocks___14_____0___fn_2 = l__mod___blocks_13_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_14_1 = self.L__mod___blocks_14_1(add_14);  add_14 = None
    l__mod___blocks_14_2 = self.L__mod___blocks_14_2(l__mod___blocks_14_1);  l__mod___blocks_14_1 = None
    l__mod___blocks_14_3 = self.L__mod___blocks_14_3(l__mod___blocks_14_2);  l__mod___blocks_14_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___15_____0___fn_0 = self.getattr_getattr_L__mod___blocks___15_____0___fn_0(l__mod___blocks_14_3)
    getattr_getattr_l__mod___blocks___15_____0___fn_1 = self.getattr_getattr_L__mod___blocks___15_____0___fn_1(getattr_getattr_l__mod___blocks___15_____0___fn_0);  getattr_getattr_l__mod___blocks___15_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___15_____0___fn_2 = self.getattr_getattr_L__mod___blocks___15_____0___fn_2(getattr_getattr_l__mod___blocks___15_____0___fn_1);  getattr_getattr_l__mod___blocks___15_____0___fn_1 = None
    add_15 = getattr_getattr_l__mod___blocks___15_____0___fn_2 + l__mod___blocks_14_3;  getattr_getattr_l__mod___blocks___15_____0___fn_2 = l__mod___blocks_14_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_15_1 = self.L__mod___blocks_15_1(add_15);  add_15 = None
    l__mod___blocks_15_2 = self.L__mod___blocks_15_2(l__mod___blocks_15_1);  l__mod___blocks_15_1 = None
    l__mod___blocks_15_3 = self.L__mod___blocks_15_3(l__mod___blocks_15_2);  l__mod___blocks_15_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___16_____0___fn_0 = self.getattr_getattr_L__mod___blocks___16_____0___fn_0(l__mod___blocks_15_3)
    getattr_getattr_l__mod___blocks___16_____0___fn_1 = self.getattr_getattr_L__mod___blocks___16_____0___fn_1(getattr_getattr_l__mod___blocks___16_____0___fn_0);  getattr_getattr_l__mod___blocks___16_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___16_____0___fn_2 = self.getattr_getattr_L__mod___blocks___16_____0___fn_2(getattr_getattr_l__mod___blocks___16_____0___fn_1);  getattr_getattr_l__mod___blocks___16_____0___fn_1 = None
    add_16 = getattr_getattr_l__mod___blocks___16_____0___fn_2 + l__mod___blocks_15_3;  getattr_getattr_l__mod___blocks___16_____0___fn_2 = l__mod___blocks_15_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_16_1 = self.L__mod___blocks_16_1(add_16);  add_16 = None
    l__mod___blocks_16_2 = self.L__mod___blocks_16_2(l__mod___blocks_16_1);  l__mod___blocks_16_1 = None
    l__mod___blocks_16_3 = self.L__mod___blocks_16_3(l__mod___blocks_16_2);  l__mod___blocks_16_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___17_____0___fn_0 = self.getattr_getattr_L__mod___blocks___17_____0___fn_0(l__mod___blocks_16_3)
    getattr_getattr_l__mod___blocks___17_____0___fn_1 = self.getattr_getattr_L__mod___blocks___17_____0___fn_1(getattr_getattr_l__mod___blocks___17_____0___fn_0);  getattr_getattr_l__mod___blocks___17_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___17_____0___fn_2 = self.getattr_getattr_L__mod___blocks___17_____0___fn_2(getattr_getattr_l__mod___blocks___17_____0___fn_1);  getattr_getattr_l__mod___blocks___17_____0___fn_1 = None
    add_17 = getattr_getattr_l__mod___blocks___17_____0___fn_2 + l__mod___blocks_16_3;  getattr_getattr_l__mod___blocks___17_____0___fn_2 = l__mod___blocks_16_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_17_1 = self.L__mod___blocks_17_1(add_17);  add_17 = None
    l__mod___blocks_17_2 = self.L__mod___blocks_17_2(l__mod___blocks_17_1);  l__mod___blocks_17_1 = None
    l__mod___blocks_17_3 = self.L__mod___blocks_17_3(l__mod___blocks_17_2);  l__mod___blocks_17_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___18_____0___fn_0 = self.getattr_getattr_L__mod___blocks___18_____0___fn_0(l__mod___blocks_17_3)
    getattr_getattr_l__mod___blocks___18_____0___fn_1 = self.getattr_getattr_L__mod___blocks___18_____0___fn_1(getattr_getattr_l__mod___blocks___18_____0___fn_0);  getattr_getattr_l__mod___blocks___18_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___18_____0___fn_2 = self.getattr_getattr_L__mod___blocks___18_____0___fn_2(getattr_getattr_l__mod___blocks___18_____0___fn_1);  getattr_getattr_l__mod___blocks___18_____0___fn_1 = None
    add_18 = getattr_getattr_l__mod___blocks___18_____0___fn_2 + l__mod___blocks_17_3;  getattr_getattr_l__mod___blocks___18_____0___fn_2 = l__mod___blocks_17_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_18_1 = self.L__mod___blocks_18_1(add_18);  add_18 = None
    l__mod___blocks_18_2 = self.L__mod___blocks_18_2(l__mod___blocks_18_1);  l__mod___blocks_18_1 = None
    l__mod___blocks_18_3 = self.L__mod___blocks_18_3(l__mod___blocks_18_2);  l__mod___blocks_18_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___19_____0___fn_0 = self.getattr_getattr_L__mod___blocks___19_____0___fn_0(l__mod___blocks_18_3)
    getattr_getattr_l__mod___blocks___19_____0___fn_1 = self.getattr_getattr_L__mod___blocks___19_____0___fn_1(getattr_getattr_l__mod___blocks___19_____0___fn_0);  getattr_getattr_l__mod___blocks___19_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___19_____0___fn_2 = self.getattr_getattr_L__mod___blocks___19_____0___fn_2(getattr_getattr_l__mod___blocks___19_____0___fn_1);  getattr_getattr_l__mod___blocks___19_____0___fn_1 = None
    add_19 = getattr_getattr_l__mod___blocks___19_____0___fn_2 + l__mod___blocks_18_3;  getattr_getattr_l__mod___blocks___19_____0___fn_2 = l__mod___blocks_18_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_19_1 = self.L__mod___blocks_19_1(add_19);  add_19 = None
    l__mod___blocks_19_2 = self.L__mod___blocks_19_2(l__mod___blocks_19_1);  l__mod___blocks_19_1 = None
    l__mod___blocks_19_3 = self.L__mod___blocks_19_3(l__mod___blocks_19_2);  l__mod___blocks_19_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___20_____0___fn_0 = self.getattr_getattr_L__mod___blocks___20_____0___fn_0(l__mod___blocks_19_3)
    getattr_getattr_l__mod___blocks___20_____0___fn_1 = self.getattr_getattr_L__mod___blocks___20_____0___fn_1(getattr_getattr_l__mod___blocks___20_____0___fn_0);  getattr_getattr_l__mod___blocks___20_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___20_____0___fn_2 = self.getattr_getattr_L__mod___blocks___20_____0___fn_2(getattr_getattr_l__mod___blocks___20_____0___fn_1);  getattr_getattr_l__mod___blocks___20_____0___fn_1 = None
    add_20 = getattr_getattr_l__mod___blocks___20_____0___fn_2 + l__mod___blocks_19_3;  getattr_getattr_l__mod___blocks___20_____0___fn_2 = l__mod___blocks_19_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_20_1 = self.L__mod___blocks_20_1(add_20);  add_20 = None
    l__mod___blocks_20_2 = self.L__mod___blocks_20_2(l__mod___blocks_20_1);  l__mod___blocks_20_1 = None
    l__mod___blocks_20_3 = self.L__mod___blocks_20_3(l__mod___blocks_20_2);  l__mod___blocks_20_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___21_____0___fn_0 = self.getattr_getattr_L__mod___blocks___21_____0___fn_0(l__mod___blocks_20_3)
    getattr_getattr_l__mod___blocks___21_____0___fn_1 = self.getattr_getattr_L__mod___blocks___21_____0___fn_1(getattr_getattr_l__mod___blocks___21_____0___fn_0);  getattr_getattr_l__mod___blocks___21_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___21_____0___fn_2 = self.getattr_getattr_L__mod___blocks___21_____0___fn_2(getattr_getattr_l__mod___blocks___21_____0___fn_1);  getattr_getattr_l__mod___blocks___21_____0___fn_1 = None
    add_21 = getattr_getattr_l__mod___blocks___21_____0___fn_2 + l__mod___blocks_20_3;  getattr_getattr_l__mod___blocks___21_____0___fn_2 = l__mod___blocks_20_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_21_1 = self.L__mod___blocks_21_1(add_21);  add_21 = None
    l__mod___blocks_21_2 = self.L__mod___blocks_21_2(l__mod___blocks_21_1);  l__mod___blocks_21_1 = None
    l__mod___blocks_21_3 = self.L__mod___blocks_21_3(l__mod___blocks_21_2);  l__mod___blocks_21_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___22_____0___fn_0 = self.getattr_getattr_L__mod___blocks___22_____0___fn_0(l__mod___blocks_21_3)
    getattr_getattr_l__mod___blocks___22_____0___fn_1 = self.getattr_getattr_L__mod___blocks___22_____0___fn_1(getattr_getattr_l__mod___blocks___22_____0___fn_0);  getattr_getattr_l__mod___blocks___22_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___22_____0___fn_2 = self.getattr_getattr_L__mod___blocks___22_____0___fn_2(getattr_getattr_l__mod___blocks___22_____0___fn_1);  getattr_getattr_l__mod___blocks___22_____0___fn_1 = None
    add_22 = getattr_getattr_l__mod___blocks___22_____0___fn_2 + l__mod___blocks_21_3;  getattr_getattr_l__mod___blocks___22_____0___fn_2 = l__mod___blocks_21_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_22_1 = self.L__mod___blocks_22_1(add_22);  add_22 = None
    l__mod___blocks_22_2 = self.L__mod___blocks_22_2(l__mod___blocks_22_1);  l__mod___blocks_22_1 = None
    l__mod___blocks_22_3 = self.L__mod___blocks_22_3(l__mod___blocks_22_2);  l__mod___blocks_22_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___23_____0___fn_0 = self.getattr_getattr_L__mod___blocks___23_____0___fn_0(l__mod___blocks_22_3)
    getattr_getattr_l__mod___blocks___23_____0___fn_1 = self.getattr_getattr_L__mod___blocks___23_____0___fn_1(getattr_getattr_l__mod___blocks___23_____0___fn_0);  getattr_getattr_l__mod___blocks___23_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___23_____0___fn_2 = self.getattr_getattr_L__mod___blocks___23_____0___fn_2(getattr_getattr_l__mod___blocks___23_____0___fn_1);  getattr_getattr_l__mod___blocks___23_____0___fn_1 = None
    add_23 = getattr_getattr_l__mod___blocks___23_____0___fn_2 + l__mod___blocks_22_3;  getattr_getattr_l__mod___blocks___23_____0___fn_2 = l__mod___blocks_22_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_23_1 = self.L__mod___blocks_23_1(add_23);  add_23 = None
    l__mod___blocks_23_2 = self.L__mod___blocks_23_2(l__mod___blocks_23_1);  l__mod___blocks_23_1 = None
    l__mod___blocks_23_3 = self.L__mod___blocks_23_3(l__mod___blocks_23_2);  l__mod___blocks_23_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___24_____0___fn_0 = self.getattr_getattr_L__mod___blocks___24_____0___fn_0(l__mod___blocks_23_3)
    getattr_getattr_l__mod___blocks___24_____0___fn_1 = self.getattr_getattr_L__mod___blocks___24_____0___fn_1(getattr_getattr_l__mod___blocks___24_____0___fn_0);  getattr_getattr_l__mod___blocks___24_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___24_____0___fn_2 = self.getattr_getattr_L__mod___blocks___24_____0___fn_2(getattr_getattr_l__mod___blocks___24_____0___fn_1);  getattr_getattr_l__mod___blocks___24_____0___fn_1 = None
    add_24 = getattr_getattr_l__mod___blocks___24_____0___fn_2 + l__mod___blocks_23_3;  getattr_getattr_l__mod___blocks___24_____0___fn_2 = l__mod___blocks_23_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_24_1 = self.L__mod___blocks_24_1(add_24);  add_24 = None
    l__mod___blocks_24_2 = self.L__mod___blocks_24_2(l__mod___blocks_24_1);  l__mod___blocks_24_1 = None
    l__mod___blocks_24_3 = self.L__mod___blocks_24_3(l__mod___blocks_24_2);  l__mod___blocks_24_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___25_____0___fn_0 = self.getattr_getattr_L__mod___blocks___25_____0___fn_0(l__mod___blocks_24_3)
    getattr_getattr_l__mod___blocks___25_____0___fn_1 = self.getattr_getattr_L__mod___blocks___25_____0___fn_1(getattr_getattr_l__mod___blocks___25_____0___fn_0);  getattr_getattr_l__mod___blocks___25_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___25_____0___fn_2 = self.getattr_getattr_L__mod___blocks___25_____0___fn_2(getattr_getattr_l__mod___blocks___25_____0___fn_1);  getattr_getattr_l__mod___blocks___25_____0___fn_1 = None
    add_25 = getattr_getattr_l__mod___blocks___25_____0___fn_2 + l__mod___blocks_24_3;  getattr_getattr_l__mod___blocks___25_____0___fn_2 = l__mod___blocks_24_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_25_1 = self.L__mod___blocks_25_1(add_25);  add_25 = None
    l__mod___blocks_25_2 = self.L__mod___blocks_25_2(l__mod___blocks_25_1);  l__mod___blocks_25_1 = None
    l__mod___blocks_25_3 = self.L__mod___blocks_25_3(l__mod___blocks_25_2);  l__mod___blocks_25_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___26_____0___fn_0 = self.getattr_getattr_L__mod___blocks___26_____0___fn_0(l__mod___blocks_25_3)
    getattr_getattr_l__mod___blocks___26_____0___fn_1 = self.getattr_getattr_L__mod___blocks___26_____0___fn_1(getattr_getattr_l__mod___blocks___26_____0___fn_0);  getattr_getattr_l__mod___blocks___26_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___26_____0___fn_2 = self.getattr_getattr_L__mod___blocks___26_____0___fn_2(getattr_getattr_l__mod___blocks___26_____0___fn_1);  getattr_getattr_l__mod___blocks___26_____0___fn_1 = None
    add_26 = getattr_getattr_l__mod___blocks___26_____0___fn_2 + l__mod___blocks_25_3;  getattr_getattr_l__mod___blocks___26_____0___fn_2 = l__mod___blocks_25_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_26_1 = self.L__mod___blocks_26_1(add_26);  add_26 = None
    l__mod___blocks_26_2 = self.L__mod___blocks_26_2(l__mod___blocks_26_1);  l__mod___blocks_26_1 = None
    l__mod___blocks_26_3 = self.L__mod___blocks_26_3(l__mod___blocks_26_2);  l__mod___blocks_26_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___27_____0___fn_0 = self.getattr_getattr_L__mod___blocks___27_____0___fn_0(l__mod___blocks_26_3)
    getattr_getattr_l__mod___blocks___27_____0___fn_1 = self.getattr_getattr_L__mod___blocks___27_____0___fn_1(getattr_getattr_l__mod___blocks___27_____0___fn_0);  getattr_getattr_l__mod___blocks___27_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___27_____0___fn_2 = self.getattr_getattr_L__mod___blocks___27_____0___fn_2(getattr_getattr_l__mod___blocks___27_____0___fn_1);  getattr_getattr_l__mod___blocks___27_____0___fn_1 = None
    add_27 = getattr_getattr_l__mod___blocks___27_____0___fn_2 + l__mod___blocks_26_3;  getattr_getattr_l__mod___blocks___27_____0___fn_2 = l__mod___blocks_26_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_27_1 = self.L__mod___blocks_27_1(add_27);  add_27 = None
    l__mod___blocks_27_2 = self.L__mod___blocks_27_2(l__mod___blocks_27_1);  l__mod___blocks_27_1 = None
    l__mod___blocks_27_3 = self.L__mod___blocks_27_3(l__mod___blocks_27_2);  l__mod___blocks_27_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___28_____0___fn_0 = self.getattr_getattr_L__mod___blocks___28_____0___fn_0(l__mod___blocks_27_3)
    getattr_getattr_l__mod___blocks___28_____0___fn_1 = self.getattr_getattr_L__mod___blocks___28_____0___fn_1(getattr_getattr_l__mod___blocks___28_____0___fn_0);  getattr_getattr_l__mod___blocks___28_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___28_____0___fn_2 = self.getattr_getattr_L__mod___blocks___28_____0___fn_2(getattr_getattr_l__mod___blocks___28_____0___fn_1);  getattr_getattr_l__mod___blocks___28_____0___fn_1 = None
    add_28 = getattr_getattr_l__mod___blocks___28_____0___fn_2 + l__mod___blocks_27_3;  getattr_getattr_l__mod___blocks___28_____0___fn_2 = l__mod___blocks_27_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_28_1 = self.L__mod___blocks_28_1(add_28);  add_28 = None
    l__mod___blocks_28_2 = self.L__mod___blocks_28_2(l__mod___blocks_28_1);  l__mod___blocks_28_1 = None
    l__mod___blocks_28_3 = self.L__mod___blocks_28_3(l__mod___blocks_28_2);  l__mod___blocks_28_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___29_____0___fn_0 = self.getattr_getattr_L__mod___blocks___29_____0___fn_0(l__mod___blocks_28_3)
    getattr_getattr_l__mod___blocks___29_____0___fn_1 = self.getattr_getattr_L__mod___blocks___29_____0___fn_1(getattr_getattr_l__mod___blocks___29_____0___fn_0);  getattr_getattr_l__mod___blocks___29_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___29_____0___fn_2 = self.getattr_getattr_L__mod___blocks___29_____0___fn_2(getattr_getattr_l__mod___blocks___29_____0___fn_1);  getattr_getattr_l__mod___blocks___29_____0___fn_1 = None
    add_29 = getattr_getattr_l__mod___blocks___29_____0___fn_2 + l__mod___blocks_28_3;  getattr_getattr_l__mod___blocks___29_____0___fn_2 = l__mod___blocks_28_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_29_1 = self.L__mod___blocks_29_1(add_29);  add_29 = None
    l__mod___blocks_29_2 = self.L__mod___blocks_29_2(l__mod___blocks_29_1);  l__mod___blocks_29_1 = None
    l__mod___blocks_29_3 = self.L__mod___blocks_29_3(l__mod___blocks_29_2);  l__mod___blocks_29_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___30_____0___fn_0 = self.getattr_getattr_L__mod___blocks___30_____0___fn_0(l__mod___blocks_29_3)
    getattr_getattr_l__mod___blocks___30_____0___fn_1 = self.getattr_getattr_L__mod___blocks___30_____0___fn_1(getattr_getattr_l__mod___blocks___30_____0___fn_0);  getattr_getattr_l__mod___blocks___30_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___30_____0___fn_2 = self.getattr_getattr_L__mod___blocks___30_____0___fn_2(getattr_getattr_l__mod___blocks___30_____0___fn_1);  getattr_getattr_l__mod___blocks___30_____0___fn_1 = None
    add_30 = getattr_getattr_l__mod___blocks___30_____0___fn_2 + l__mod___blocks_29_3;  getattr_getattr_l__mod___blocks___30_____0___fn_2 = l__mod___blocks_29_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_30_1 = self.L__mod___blocks_30_1(add_30);  add_30 = None
    l__mod___blocks_30_2 = self.L__mod___blocks_30_2(l__mod___blocks_30_1);  l__mod___blocks_30_1 = None
    l__mod___blocks_30_3 = self.L__mod___blocks_30_3(l__mod___blocks_30_2);  l__mod___blocks_30_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    getattr_getattr_l__mod___blocks___31_____0___fn_0 = self.getattr_getattr_L__mod___blocks___31_____0___fn_0(l__mod___blocks_30_3)
    getattr_getattr_l__mod___blocks___31_____0___fn_1 = self.getattr_getattr_L__mod___blocks___31_____0___fn_1(getattr_getattr_l__mod___blocks___31_____0___fn_0);  getattr_getattr_l__mod___blocks___31_____0___fn_0 = None
    getattr_getattr_l__mod___blocks___31_____0___fn_2 = self.getattr_getattr_L__mod___blocks___31_____0___fn_2(getattr_getattr_l__mod___blocks___31_____0___fn_1);  getattr_getattr_l__mod___blocks___31_____0___fn_1 = None
    add_31 = getattr_getattr_l__mod___blocks___31_____0___fn_2 + l__mod___blocks_30_3;  getattr_getattr_l__mod___blocks___31_____0___fn_2 = l__mod___blocks_30_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    l__mod___blocks_31_1 = self.L__mod___blocks_31_1(add_31);  add_31 = None
    l__mod___blocks_31_2 = self.L__mod___blocks_31_2(l__mod___blocks_31_1);  l__mod___blocks_31_1 = None
    x_2 = self.L__mod___blocks_31_3(l__mod___blocks_31_2);  l__mod___blocks_31_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_3 = self.L__mod___pooling_pool(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_5 = self.L__mod___pooling_flatten(x_3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:94, code: x = self.head_drop(x)
    x_6 = self.L__mod___head_drop(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:95, code: return x if pre_logits else self.head(x)
    pred = self.L__mod___head(x_6);  x_6 = None
    return (pred,)
    