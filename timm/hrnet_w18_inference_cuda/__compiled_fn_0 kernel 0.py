
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2pireug6rdmcbcakb5biu7ggk5jqzu4blznrmd5fad7q2edxnk.py
# Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
# x_3 => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rw/crwxcbbcfeaez32rev7l3jqg7nvdiuwyzhdxnf7o5lfjds5sr3p7.py
# Source Nodes: [shortcut, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# shortcut => relu_1
# x_4 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxtomacuodz5d7up2g6kal5uy7zn5hr7jbazoahb2qqwheav72f.py
# Source Nodes: [shortcut_1, shortcut_2, x_15, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_1 => add_11, mul_16, mul_17, sub_5
# shortcut_2 => relu_4
# x_15 => add_9, mul_13, mul_14, sub_4
# x_16 => add_12
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6f/c6fuekgtr7lp5mkjcnzllglpkdhclne7rjr6wlk4inrjohge7gt4.py
# Source Nodes: [shortcut_3, x_27, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_3 => relu_7
# x_27 => add_18, mul_25, mul_26, sub_8
# x_28 => add_19
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/35/c35prg75qwwze6qm5myjppwfkdf2guk5kdwm53gzg5ri22i3h3j5.py
# Source Nodes: [l__mod___transition1_0_1, shortcut_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___transition1_0_1 => add_35, mul_46, mul_47, sub_15
# shortcut_5 => relu_14
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 18
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gn/cgncdll5734sg2byeyormuzjf4ejune4kipbaoqesqrrrcivamkc.py
# Source Nodes: [shortcut_6, x_60, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_6 => relu_17
# x_60 => add_41, mul_55, mul_56, sub_18
# x_61 => add_42
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 18
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmuqqjtyucqjsekv72f6nhlg6cfb6lepkhqwbbahchx4wavf3ho.py
# Source Nodes: [l__mod___transition1_1_0_1, shortcut_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___transition1_1_0_1 => add_37, mul_49, mul_50, sub_16
# shortcut_9 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 36
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckc7vclyklylvyx3qj74v2s55auyrq5u3v4pjquiu7ybzxhyr7fe.py
# Source Nodes: [shortcut_10, x_96, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_10 => relu_25
# x_96 => add_61, mul_79, mul_80, sub_26
# x_97 => add_62
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 36
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crlblpxdqnrwje2nwm3dj67jovhcou7evs5kx57mz33gqnaa55wz.py
# Source Nodes: [l__mod___stage2_0_fuse_layers_0_1_1, l__mod___stage2_0_fuse_layers_0_1_2, shortcut_13, x_87, x_88, x_89, y_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# l__mod___stage2_0_fuse_layers_0_1_1 => add_79, mul_100, mul_101, sub_33
# l__mod___stage2_0_fuse_layers_0_1_2 => _unsafe_index
# shortcut_13 => relu_32
# x_87 => add_56, mul_73, mul_74, sub_24
# x_88 => add_57
# x_89 => relu_23
# y_1 => add_84
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 18
    x5 = (xindex // 56) % 56
    x4 = xindex % 56
    x6 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = x5
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp8
    tmp21 = 0.0
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 + tmp21
    tmp24 = 0.5
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.int32)
    tmp27 = x4
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp8
    tmp30 = tmp29 + tmp21
    tmp31 = tmp30 + tmp21
    tmp32 = tmp31 * tmp24
    tmp33 = tmp32.to(tl.int32)
    tmp34 = tl.load(in_ptr5 + (tmp33 + (28*tmp26) + (784*x6)), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 + tmp4
    tmp39 = tl.sqrt(tmp38)
    tmp40 = 1 / tmp39
    tmp41 = tmp40 * tmp8
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp17 + tmp46
    tmp48 = triton_helpers.maximum(0, tmp47)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr0 + (x3), tmp48, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfch22i6qrduix2vzczj42gvyy46ys3rmclh26t2xbmgxgbvgsk.py
# Source Nodes: [shortcut_18, x_168, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_18 => relu_44
# x_168 => add_113, mul_140, mul_141, sub_45
# x_169 => add_114
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 36
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2y/c2ylpchqir6lxsc2wjjopyfpinbjoyj2p4p2usdq5ey4sgztc2bh.py
# Source Nodes: [l__mod___transition2_2_0_1, shortcut_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___transition2_2_0_1 => add_89, mul_110, mul_111, sub_35
# shortcut_21 => relu_34
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 72
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iz/cizpcqofijfqg5towfstkupza7jjslot64xiffubk34kwy3jc6hw.py
# Source Nodes: [shortcut_22, x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_22 => relu_52
# x_204 => add_133, mul_164, mul_165, sub_53
# x_205 => add_134
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 72
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sn/csnzck77k6smbcokwooy5o733nkc6ivr7nezyeqh2zsl6ill6dqp.py
# Source Nodes: [l__mod___stage3_0_fuse_layers_0_1_1, l__mod___stage3_0_fuse_layers_0_1_2, l__mod___stage3_0_fuse_layers_0_2_1, l__mod___stage3_0_fuse_layers_0_2_2, shortcut_25, x_159, x_160, x_161, y_5, y_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# l__mod___stage3_0_fuse_layers_0_1_1 => add_151, mul_185, mul_186, sub_60
# l__mod___stage3_0_fuse_layers_0_1_2 => _unsafe_index_1
# l__mod___stage3_0_fuse_layers_0_2_1 => add_158, mul_192, mul_193, sub_61
# l__mod___stage3_0_fuse_layers_0_2_2 => _unsafe_index_2
# shortcut_25 => relu_59
# x_159 => add_108, mul_134, mul_135, sub_43
# x_160 => add_109
# x_161 => relu_42
# y_5 => add_156
# y_6 => add_163
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 18
    x5 = (xindex // 56) % 56
    x4 = xindex % 56
    x6 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = x5
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp8
    tmp21 = 0.0
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 + tmp21
    tmp24 = 0.5
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.int32)
    tmp27 = x4
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp8
    tmp30 = tmp29 + tmp21
    tmp31 = tmp30 + tmp21
    tmp32 = tmp31 * tmp24
    tmp33 = tmp32.to(tl.int32)
    tmp34 = tl.load(in_ptr5 + (tmp33 + (28*tmp26) + (784*x6)), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 + tmp4
    tmp39 = tl.sqrt(tmp38)
    tmp40 = 1 / tmp39
    tmp41 = tmp40 * tmp8
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp17 + tmp46
    tmp48 = 0.25
    tmp49 = tmp23 * tmp48
    tmp50 = tmp49.to(tl.int32)
    tmp51 = tmp31 * tmp48
    tmp52 = tmp51.to(tl.int32)
    tmp53 = tl.load(in_ptr10 + (tmp52 + (14*tmp50) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp55 = tmp53 - tmp54
    tmp57 = tmp56 + tmp4
    tmp58 = tl.sqrt(tmp57)
    tmp59 = 1 / tmp58
    tmp60 = tmp59 * tmp8
    tmp61 = tmp55 * tmp60
    tmp63 = tmp61 * tmp62
    tmp65 = tmp63 + tmp64
    tmp66 = tmp47 + tmp65
    tmp67 = triton_helpers.maximum(0, tmp66)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
    tl.store(in_out_ptr1 + (x3), tmp67, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gz/cgzig4omfhyjqwnlnn4lmmkvch2b65ohvy75uynctk7a3bdjwjzf.py
# Source Nodes: [l__mod___stage3_0_fuse_layers_1_2_1, l__mod___stage3_0_fuse_layers_1_2_2, shortcut_29, y_7, y_8, y_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# l__mod___stage3_0_fuse_layers_1_2_1 => add_168, mul_202, mul_203, sub_63
# l__mod___stage3_0_fuse_layers_1_2_2 => _unsafe_index_3
# shortcut_29 => relu_60
# y_7 => add_165, mul_199, mul_200, sub_62
# y_8 => add_166
# y_9 => add_173
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 784) % 36
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x6 = (xindex // 784)
    tmp0 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x4), xmask)
    tmp34 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = x1
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp8
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21 + tmp20
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.int32)
    tmp26 = x0
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp8
    tmp29 = tmp28 + tmp20
    tmp30 = tmp29 + tmp20
    tmp31 = tmp30 * tmp23
    tmp32 = tmp31.to(tl.int32)
    tmp33 = tl.load(in_ptr5 + (tmp32 + (14*tmp25) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp35 = tmp33 - tmp34
    tmp37 = tmp36 + tmp4
    tmp38 = tl.sqrt(tmp37)
    tmp39 = 1 / tmp38
    tmp40 = tmp39 * tmp8
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp16 + tmp45
    tmp47 = triton_helpers.maximum(0, tmp46)
    tl.store(in_out_ptr0 + (x4), tmp47, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3pg4gb4ui7itjrs4cca55azwmortvyeshk3jbjnwzfv43pl7bn.py
# Source Nodes: [l__mod___stage3_0_fuse_layers_2_0_0_1, l__mod___stage3_0_fuse_layers_2_0_0_2, l__mod___stage3_0_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___stage3_0_fuse_layers_2_0_0_1 => add_175, mul_209, mul_210, sub_64
# l__mod___stage3_0_fuse_layers_2_0_0_2 => relu_61
# l__mod___stage3_0_fuse_layers_2_0_1_0 => convolution_65
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 18
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/na/cnauc2m3jndokdogzc2ohs7qkb2b3newf4o55c7rojaj3rqucgyl.py
# Source Nodes: [l__mod___stage3_0_fuse_layers_2_1_0_1, shortcut_33, y_10, y_11, y_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# l__mod___stage3_0_fuse_layers_2_1_0_1 => add_179, mul_215, mul_216, sub_66
# shortcut_33 => relu_62
# y_10 => add_177, mul_212, mul_213, sub_65
# y_11 => add_180
# y_12 => add_181
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 72
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = triton_helpers.maximum(0, tmp30)
    tl.store(in_out_ptr1 + (x3), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65oop7qnueiv3qggue4u7kikv76jt242owo6xref7jqisgadvfm.py
# Source Nodes: [shortcut_70, x_636, x_637], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_70 => relu_165
# x_636 => add_503, mul_587, mul_588, sub_178
# x_637 => add_504
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 72
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfkyiffgyms75yiko47ok7yta7ycyfo6zmufvy3lbhbwuj4ptwf.py
# Source Nodes: [l__mod___transition3_3_0_1, shortcut_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___transition3_3_0_1 => add_459, mul_533, mul_534, sub_160
# shortcut_73 => relu_147
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 144
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lu/cluapnij3ovmrenqjkxi4eyylj5ipqc3qo4k56wjgxzdoxlwbvkl.py
# Source Nodes: [shortcut_74, x_672, x_673], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_74 => relu_173
# x_672 => add_523, mul_611, mul_612, sub_186
# x_673 => add_524
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 144
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbv7x7v7fqc2grn34mbzopmibh4azjhyrxm3hmn57dejanncvojo.py
# Source Nodes: [l__mod___stage4_0_fuse_layers_0_1_1, l__mod___stage4_0_fuse_layers_0_1_2, l__mod___stage4_0_fuse_layers_0_2_1, l__mod___stage4_0_fuse_layers_0_2_2, l__mod___stage4_0_fuse_layers_0_3_1, l__mod___stage4_0_fuse_layers_0_3_2, shortcut_77, x_591, x_592, x_593, y_41, y_42, y_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# l__mod___stage4_0_fuse_layers_0_1_1 => add_541, mul_632, mul_633, sub_193
# l__mod___stage4_0_fuse_layers_0_1_2 => _unsafe_index_13
# l__mod___stage4_0_fuse_layers_0_2_1 => add_548, mul_639, mul_640, sub_194
# l__mod___stage4_0_fuse_layers_0_2_2 => _unsafe_index_14
# l__mod___stage4_0_fuse_layers_0_3_1 => add_555, mul_646, mul_647, sub_195
# l__mod___stage4_0_fuse_layers_0_3_2 => _unsafe_index_15
# shortcut_77 => relu_180
# x_591 => add_478, mul_557, mul_558, sub_168
# x_592 => add_479
# x_593 => relu_155
# y_41 => add_546
# y_42 => add_553
# y_43 => add_560
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(22,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 18
    x5 = (xindex // 56) % 56
    x4 = xindex % 56
    x6 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr16 + (x1), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr17 + (x1), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr19 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = x5
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp8
    tmp21 = 0.0
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 + tmp21
    tmp24 = 0.5
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.int32)
    tmp27 = x4
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp8
    tmp30 = tmp29 + tmp21
    tmp31 = tmp30 + tmp21
    tmp32 = tmp31 * tmp24
    tmp33 = tmp32.to(tl.int32)
    tmp34 = tl.load(in_ptr5 + (tmp33 + (28*tmp26) + (784*x6)), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 + tmp4
    tmp39 = tl.sqrt(tmp38)
    tmp40 = 1 / tmp39
    tmp41 = tmp40 * tmp8
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp17 + tmp46
    tmp48 = 0.25
    tmp49 = tmp23 * tmp48
    tmp50 = tmp49.to(tl.int32)
    tmp51 = tmp31 * tmp48
    tmp52 = tmp51.to(tl.int32)
    tmp53 = tl.load(in_ptr10 + (tmp52 + (14*tmp50) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp55 = tmp53 - tmp54
    tmp57 = tmp56 + tmp4
    tmp58 = tl.sqrt(tmp57)
    tmp59 = 1 / tmp58
    tmp60 = tmp59 * tmp8
    tmp61 = tmp55 * tmp60
    tmp63 = tmp61 * tmp62
    tmp65 = tmp63 + tmp64
    tmp66 = tmp47 + tmp65
    tmp67 = 0.125
    tmp68 = tmp23 * tmp67
    tmp69 = tmp68.to(tl.int32)
    tmp70 = tmp31 * tmp67
    tmp71 = tmp70.to(tl.int32)
    tmp72 = tl.load(in_ptr15 + (tmp71 + (7*tmp69) + (49*x6)), xmask, eviction_policy='evict_last')
    tmp74 = tmp72 - tmp73
    tmp76 = tmp75 + tmp4
    tmp77 = tl.sqrt(tmp76)
    tmp78 = 1 / tmp77
    tmp79 = tmp78 * tmp8
    tmp80 = tmp74 * tmp79
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp85 = tmp66 + tmp84
    tmp86 = triton_helpers.maximum(0, tmp85)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
    tl.store(in_out_ptr1 + (x3), tmp86, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjflqb6xgkcwkqpk3peyfou2tyzlxv3kpqprnmzlnwsufdrm5ddk.py
# Source Nodes: [l__mod___stage4_0_fuse_layers_1_2_1, l__mod___stage4_0_fuse_layers_1_2_2, l__mod___stage4_0_fuse_layers_1_3_1, l__mod___stage4_0_fuse_layers_1_3_2, shortcut_81, y_44, y_45, y_46, y_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# l__mod___stage4_0_fuse_layers_1_2_1 => add_565, mul_656, mul_657, sub_197
# l__mod___stage4_0_fuse_layers_1_2_2 => _unsafe_index_16
# l__mod___stage4_0_fuse_layers_1_3_1 => add_572, mul_663, mul_664, sub_198
# l__mod___stage4_0_fuse_layers_1_3_2 => _unsafe_index_17
# shortcut_81 => relu_181
# y_44 => add_562, mul_653, mul_654, sub_196
# y_45 => add_563
# y_46 => add_570
# y_47 => add_577
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 784) % 36
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x6 = (xindex // 784)
    tmp0 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x4), xmask)
    tmp34 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr14 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = x1
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp8
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21 + tmp20
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.int32)
    tmp26 = x0
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp8
    tmp29 = tmp28 + tmp20
    tmp30 = tmp29 + tmp20
    tmp31 = tmp30 * tmp23
    tmp32 = tmp31.to(tl.int32)
    tmp33 = tl.load(in_ptr5 + (tmp32 + (14*tmp25) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp35 = tmp33 - tmp34
    tmp37 = tmp36 + tmp4
    tmp38 = tl.sqrt(tmp37)
    tmp39 = 1 / tmp38
    tmp40 = tmp39 * tmp8
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp16 + tmp45
    tmp47 = 0.25
    tmp48 = tmp22 * tmp47
    tmp49 = tmp48.to(tl.int32)
    tmp50 = tmp30 * tmp47
    tmp51 = tmp50.to(tl.int32)
    tmp52 = tl.load(in_ptr10 + (tmp51 + (7*tmp49) + (49*x6)), xmask, eviction_policy='evict_last')
    tmp54 = tmp52 - tmp53
    tmp56 = tmp55 + tmp4
    tmp57 = tl.sqrt(tmp56)
    tmp58 = 1 / tmp57
    tmp59 = tmp58 * tmp8
    tmp60 = tmp54 * tmp59
    tmp62 = tmp60 * tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp46 + tmp64
    tmp66 = triton_helpers.maximum(0, tmp65)
    tl.store(in_out_ptr0 + (x4), tmp66, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tyg2es22zlx276kkwxtxatvdjfsqbghilmjx4a3ea5vkux7jd7.py
# Source Nodes: [l__mod___stage4_0_fuse_layers_2_1_0_1, l__mod___stage4_0_fuse_layers_2_3_1, l__mod___stage4_0_fuse_layers_2_3_2, shortcut_85, y_48, y_49, y_50, y_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# l__mod___stage4_0_fuse_layers_2_1_0_1 => add_583, mul_676, mul_677, sub_201
# l__mod___stage4_0_fuse_layers_2_3_1 => add_587, mul_679, mul_680, sub_202
# l__mod___stage4_0_fuse_layers_2_3_2 => _unsafe_index_18
# shortcut_85 => relu_183
# y_48 => add_581, mul_673, mul_674, sub_200
# y_49 => add_584
# y_50 => add_585
# y_51 => add_592
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 72
    x5 = (xindex // 14) % 14
    x4 = xindex % 14
    x6 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x3), xmask)
    tmp48 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = x5
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp8
    tmp34 = 0.0
    tmp35 = tmp33 + tmp34
    tmp36 = tmp35 + tmp34
    tmp37 = 0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tmp38.to(tl.int32)
    tmp40 = x4
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 * tmp8
    tmp43 = tmp42 + tmp34
    tmp44 = tmp43 + tmp34
    tmp45 = tmp44 * tmp37
    tmp46 = tmp45.to(tl.int32)
    tmp47 = tl.load(in_ptr10 + (tmp46 + (7*tmp39) + (49*x6)), xmask, eviction_policy='evict_last')
    tmp49 = tmp47 - tmp48
    tmp51 = tmp50 + tmp4
    tmp52 = tl.sqrt(tmp51)
    tmp53 = 1 / tmp52
    tmp54 = tmp53 * tmp8
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tmp30 + tmp59
    tmp61 = triton_helpers.maximum(0, tmp60)
    tl.store(in_out_ptr0 + (x3), tmp61, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceaotwni6uubwr7pl4tqftoks7zbvuktl2mhol5rzglxgwuyqvwx.py
# Source Nodes: [l__mod___stage4_0_fuse_layers_3_0_1_1, l__mod___stage4_0_fuse_layers_3_0_1_2, l__mod___stage4_0_fuse_layers_3_0_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___stage4_0_fuse_layers_3_0_1_1 => add_596, mul_689, mul_690, sub_204
# l__mod___stage4_0_fuse_layers_3_0_1_2 => relu_185
# l__mod___stage4_0_fuse_layers_3_0_2_0 => convolution_205
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 18
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p3/cp34fbx3x76jqllcuifdlyfgf7rpf2wdbsnwwkosfps5f7w6fb4d.py
# Source Nodes: [l__mod___stage4_0_fuse_layers_3_1_0_1, l__mod___stage4_0_fuse_layers_3_1_0_2, l__mod___stage4_0_fuse_layers_3_1_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___stage4_0_fuse_layers_3_1_0_1 => add_600, mul_695, mul_696, sub_206
# l__mod___stage4_0_fuse_layers_3_1_0_2 => relu_186
# l__mod___stage4_0_fuse_layers_3_1_1_0 => convolution_207
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 36
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnzgw5v76tfz5d6boz7e3hyy5dqxduplvnj3yfvmatcuspdmmhq.py
# Source Nodes: [l__mod___stage4_0_fuse_layers_3_1_1_1, l__mod___stage4_0_fuse_layers_3_2_0_1, shortcut_89, y_52, y_53, y_54, y_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# l__mod___stage4_0_fuse_layers_3_1_1_1 => add_602, mul_698, mul_699, sub_207
# l__mod___stage4_0_fuse_layers_3_2_0_1 => add_605, mul_701, mul_702, sub_208
# shortcut_89 => relu_187
# y_52 => add_598, mul_692, mul_693, sub_205
# y_53 => add_603
# y_54 => add_606
# y_55 => add_607
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 144
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x3), xmask)
    tmp30 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp4
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1 / tmp34
    tmp36 = tmp35 * tmp8
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp28 + tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tl.store(in_out_ptr1 + (x3), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/im/cimnmxknl3gyy4ll4aj2rmoaxfwqf2377dghhs6ewmznujc2qb5y.py
# Source Nodes: [l__mod___stage4_2_fuse_layers_3_1_1_1, l__mod___stage4_2_fuse_layers_3_2_0_1, shortcut_115, y_84, y_85, y_86, y_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# l__mod___stage4_2_fuse_layers_3_1_1_1 => add_898, mul_1034, mul_1035, sub_303
# l__mod___stage4_2_fuse_layers_3_2_0_1 => add_901, mul_1037, mul_1038, sub_304
# shortcut_115 => relu_267
# y_84 => add_894, mul_1028, mul_1029, sub_301
# y_85 => add_899
# y_86 => add_902
# y_87 => add_903
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 144
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), xmask)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x3), xmask)
    tmp30 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr14 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp31 = tmp29 - tmp30
    tmp33 = tmp32 + tmp4
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1 / tmp34
    tmp36 = tmp35 * tmp8
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp28 + tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tl.store(in_out_ptr0 + (x3), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpumq4gkwdszbu5rnhsojr6wp2bcdfhkgambajemtsjsnuppnmhj.py
# Source Nodes: [x_1027, x_1028, x_1029], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_1027 => add_938, mul_1082, mul_1083, sub_319
# x_1028 => relu_279
# x_1029 => convolution_320
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tcipfrgfjsdrqiv74ohkvlxdtnxskxpxxsxznpbc5xdk6udfaj.py
# Source Nodes: [x_1015, x_1016, x_1017], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_1015 => add_926, mul_1067, mul_1068, sub_314
# x_1016 => relu_275
# x_1017 => convolution_315
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zjmoado7p3y2vzmv46yqci24kj52bnkgatbyc7k7ttvb4lql2p.py
# Source Nodes: [l__mod___stage4_2_fuse_layers_1_2_1, l__mod___stage4_2_fuse_layers_1_2_2, l__mod___stage4_2_fuse_layers_1_3_1, l__mod___stage4_2_fuse_layers_1_3_2, shortcut_111, y_76, y_77, y_78, y_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
# l__mod___stage4_2_fuse_layers_1_2_1 => add_861, mul_992, mul_993, sub_293
# l__mod___stage4_2_fuse_layers_1_2_2 => _unsafe_index_28
# l__mod___stage4_2_fuse_layers_1_3_1 => add_868, mul_1000, mul_999, sub_294
# l__mod___stage4_2_fuse_layers_1_3_2 => _unsafe_index_29
# shortcut_111 => relu_261
# y_76 => add_858, mul_989, mul_990, sub_292
# y_77 => add_859
# y_78 => add_866
# y_79 => add_873
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 784) % 36
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x6 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp34 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr14 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = x1
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp8
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21 + tmp20
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.int32)
    tmp26 = x0
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp8
    tmp29 = tmp28 + tmp20
    tmp30 = tmp29 + tmp20
    tmp31 = tmp30 * tmp23
    tmp32 = tmp31.to(tl.int32)
    tmp33 = tl.load(in_ptr5 + (tmp32 + (14*tmp25) + (196*x6)), xmask, eviction_policy='evict_last')
    tmp35 = tmp33 - tmp34
    tmp37 = tmp36 + tmp4
    tmp38 = tl.sqrt(tmp37)
    tmp39 = 1 / tmp38
    tmp40 = tmp39 * tmp8
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp16 + tmp45
    tmp47 = 0.25
    tmp48 = tmp22 * tmp47
    tmp49 = tmp48.to(tl.int32)
    tmp50 = tmp30 * tmp47
    tmp51 = tmp50.to(tl.int32)
    tmp52 = tl.load(in_ptr10 + (tmp51 + (7*tmp49) + (49*x6)), xmask, eviction_policy='evict_last')
    tmp54 = tmp52 - tmp53
    tmp56 = tmp55 + tmp4
    tmp57 = tl.sqrt(tmp56)
    tmp58 = 1 / tmp57
    tmp59 = tmp58 * tmp8
    tmp60 = tmp54 * tmp59
    tmp62 = tmp60 * tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp46 + tmp64
    tmp66 = triton_helpers.maximum(0, tmp65)
    tl.store(in_out_ptr0 + (x4), tmp66, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2q/c2q666q6pdq4hai55kto7rzqbwlhhivqfzpi3eyrsbyzeaof36dv.py
# Source Nodes: [x_1003, x_1004, x_1005], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_1003 => add_914, mul_1052, mul_1053, sub_309
# x_1004 => relu_271
# x_1005 => convolution_310
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/72/c72kmxv5apwtljtan54rcsvloci5miszdu3jch2gzd7tqkpvsxx2.py
# Source Nodes: [x_991, x_992, x_993], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_991 => add_905, mul_1040, mul_1041, sub_305
# x_992 => relu_268
# x_993 => convolution_306
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6u7vi7czqjawodxjngqds6mxykkyzff5fhsalybsfdlhypkxnp.py
# Source Nodes: [forward, shortcut_110, x_1000, x_999, y_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# forward => convolution_313
# shortcut_110 => add_911, mul_1049, mul_1050, sub_308
# x_1000 => add_912
# x_999 => add_909, mul_1046, mul_1047, sub_307
# y_88 => relu_270
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vtzsgnch4hgc5wi5wh3g5ssu3u7qjbzvfebckwbic5sm6in6gm.py
# Source Nodes: [forward, forward_1, shortcut_112, x_1011, x_1012, x_1013, y_88, y_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# forward => add_923, convolution_313, mul_1064, mul_1065, relu_274, sub_313
# forward_1 => convolution_318
# shortcut_112 => add_920, mul_1061, mul_1062, sub_312
# x_1011 => add_918, mul_1058, mul_1059, sub_311
# x_1012 => add_921
# x_1013 => relu_273
# y_88 => relu_270
# y_89 => add_924
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), None)
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 - tmp33
    tmp36 = tmp35 + tmp4
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = tmp38 * tmp8
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tmp46 = tmp29 + tmp45
    tl.store(in_out_ptr0 + (x3), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yd/cydgyyuz5axxt55gma7ebwct6uoc7hwxnwyeqg4ev3boxphsrqn3.py
# Source Nodes: [forward, forward_1, forward_2, shortcut_114, x_1013, x_1023, x_1024, x_1025, y_88, y_89, y_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# forward => add_923, convolution_313, mul_1064, mul_1065, relu_274, sub_313
# forward_1 => add_935, convolution_318, mul_1079, mul_1080, relu_278, sub_318
# forward_2 => convolution_323
# shortcut_114 => add_932, mul_1076, mul_1077, sub_317
# x_1013 => relu_273
# x_1023 => add_930, mul_1073, mul_1074, sub_316
# x_1024 => add_933
# x_1025 => relu_277
# y_88 => relu_270
# y_89 => add_924
# y_90 => add_936
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), None)
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 - tmp33
    tmp36 = tmp35 + tmp4
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = tmp38 * tmp8
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tmp46 = tmp29 + tmp45
    tl.store(in_out_ptr0 + (x3), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bmatptojlwkjyvkczrqob3a2s5h2owg3p7bpfkbeav5dwml52t.py
# Source Nodes: [forward, forward_1, forward_2, l__mod___final_layer_0, shortcut_116, x_1013, x_1025, x_1035, x_1036, x_1037, y_88, y_89, y_90, y_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# forward => add_923, convolution_313, mul_1064, mul_1065, relu_274, sub_313
# forward_1 => add_935, convolution_318, mul_1079, mul_1080, relu_278, sub_318
# forward_2 => add_947, convolution_323, mul_1094, mul_1095, relu_282, sub_323
# l__mod___final_layer_0 => convolution_324
# shortcut_116 => add_944, mul_1091, mul_1092, sub_322
# x_1013 => relu_273
# x_1025 => relu_277
# x_1035 => add_942, mul_1088, mul_1089, sub_321
# x_1036 => add_945
# x_1037 => relu_281
# y_88 => relu_270
# y_89 => add_924
# y_90 => add_936
# y_91 => add_948
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), None)
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 - tmp33
    tmp36 = tmp35 + tmp4
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = tmp38 * tmp8
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tmp46 = tmp29 + tmp45
    tl.store(in_out_ptr0 + (x3), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxe2bivmfzwc6yhswxtxquxc3xb76kwnnixuj7rkkpq7fbvvah7p.py
# Source Nodes: [forward, forward_1, forward_2, l__mod___final_layer_0, l__mod___final_layer_1, x_1013, x_1025, x_1037, x_1038, y_88, y_89, y_90, y_91, y_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.mean, aten.relu]
# forward => add_923, convolution_313, mul_1064, mul_1065, relu_274, sub_313
# forward_1 => add_935, convolution_318, mul_1079, mul_1080, relu_278, sub_318
# forward_2 => add_947, convolution_323, mul_1094, mul_1095, relu_282, sub_323
# l__mod___final_layer_0 => convolution_324
# l__mod___final_layer_1 => add_950, mul_1097, mul_1098, sub_324
# x_1013 => relu_273
# x_1025 => relu_277
# x_1037 => relu_281
# x_1038 => mean
# y_88 => relu_270
# y_89 => add_924
# y_90 => add_936
# y_91 => add_948
# y_93 => relu_283
triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_35', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1488_1, arg1489_1, arg1490_1, arg1491_1, arg1492_1, arg1493_1, arg1494_1, arg1495_1, arg1496_1, arg1497_1, arg1498_1, arg1499_1, arg1500_1, arg1501_1, arg1502_1, arg1503_1, arg1504_1, arg1505_1, arg1506_1, arg1507_1, arg1508_1, arg1509_1, arg1510_1, arg1511_1, arg1512_1, arg1513_1, arg1514_1, arg1515_1, arg1516_1, arg1517_1, arg1518_1, arg1519_1, arg1520_1, arg1521_1, arg1522_1, arg1523_1, arg1524_1, arg1525_1, arg1526_1, arg1527_1, arg1528_1, arg1529_1, arg1530_1, arg1531_1, arg1532_1, arg1533_1, arg1534_1, arg1535_1, arg1536_1, arg1537_1, arg1538_1, arg1539_1, arg1540_1, arg1541_1, arg1542_1, arg1543_1, arg1544_1, arg1545_1, arg1546_1, arg1547_1, arg1548_1, arg1549_1, arg1550_1, arg1551_1, arg1552_1, arg1553_1, arg1554_1, arg1555_1, arg1556_1, arg1557_1, arg1558_1, arg1559_1, arg1560_1, arg1561_1, arg1562_1, arg1563_1, arg1564_1, arg1565_1, arg1566_1, arg1567_1, arg1568_1, arg1569_1, arg1570_1, arg1571_1, arg1572_1, arg1573_1, arg1574_1, arg1575_1, arg1576_1, arg1577_1, arg1578_1, arg1579_1, arg1580_1, arg1581_1, arg1582_1, arg1583_1, arg1584_1, arg1585_1, arg1586_1, arg1587_1, arg1588_1, arg1589_1, arg1590_1, arg1591_1, arg1592_1, arg1593_1, arg1594_1, arg1595_1, arg1596_1, arg1597_1, arg1598_1, arg1599_1, arg1600_1, arg1601_1, arg1602_1, arg1603_1, arg1604_1, arg1605_1, arg1606_1, arg1607_1, arg1608_1, arg1609_1, arg1610_1, arg1611_1, arg1612_1, arg1613_1, arg1614_1, arg1615_1, arg1616_1, arg1617_1, arg1618_1, arg1619_1, arg1620_1, arg1621_1, arg1622_1, arg1623_1, arg1624_1, arg1625_1, arg1626_1, arg1627_1, arg1628_1, arg1629_1, arg1630_1, arg1631_1, arg1632_1, arg1633_1, arg1634_1, arg1635_1, arg1636_1, arg1637_1, arg1638_1, arg1639_1, arg1640_1, arg1641_1, arg1642_1, arg1643_1, arg1644_1, arg1645_1, arg1646_1, arg1647_1, arg1648_1, arg1649_1, arg1650_1, arg1651_1, arg1652_1, arg1653_1, arg1654_1, arg1655_1, arg1656_1, arg1657_1, arg1658_1, arg1659_1, arg1660_1, arg1661_1, arg1662_1, arg1663_1, arg1664_1, arg1665_1, arg1666_1, arg1667_1, arg1668_1, arg1669_1, arg1670_1, arg1671_1, arg1672_1, arg1673_1, arg1674_1, arg1675_1, arg1676_1, arg1677_1, arg1678_1, arg1679_1, arg1680_1, arg1681_1, arg1682_1, arg1683_1, arg1684_1, arg1685_1, arg1686_1, arg1687_1, arg1688_1, arg1689_1, arg1690_1, arg1691_1, arg1692_1, arg1693_1, arg1694_1, arg1695_1, arg1696_1, arg1697_1, arg1698_1, arg1699_1, arg1700_1, arg1701_1, arg1702_1, arg1703_1, arg1704_1, arg1705_1, arg1706_1, arg1707_1, arg1708_1, arg1709_1, arg1710_1, arg1711_1, arg1712_1, arg1713_1, arg1714_1, arg1715_1, arg1716_1, arg1717_1, arg1718_1, arg1719_1, arg1720_1, arg1721_1, arg1722_1, arg1723_1, arg1724_1, arg1725_1, arg1726_1, arg1727_1, arg1728_1, arg1729_1, arg1730_1, arg1731_1, arg1732_1, arg1733_1, arg1734_1, arg1735_1, arg1736_1, arg1737_1, arg1738_1, arg1739_1, arg1740_1, arg1741_1, arg1742_1, arg1743_1, arg1744_1, arg1745_1, arg1746_1, arg1747_1, arg1748_1, arg1749_1, arg1750_1, arg1751_1, arg1752_1, arg1753_1, arg1754_1, arg1755_1, arg1756_1, arg1757_1, arg1758_1, arg1759_1, arg1760_1, arg1761_1, arg1762_1, arg1763_1, arg1764_1, arg1765_1, arg1766_1, arg1767_1, arg1768_1, arg1769_1, arg1770_1, arg1771_1, arg1772_1, arg1773_1, arg1774_1, arg1775_1, arg1776_1, arg1777_1, arg1778_1, arg1779_1, arg1780_1, arg1781_1, arg1782_1, arg1783_1, arg1784_1, arg1785_1, arg1786_1, arg1787_1, arg1788_1, arg1789_1, arg1790_1, arg1791_1, arg1792_1, arg1793_1, arg1794_1, arg1795_1, arg1796_1, arg1797_1, arg1798_1, arg1799_1, arg1800_1, arg1801_1, arg1802_1, arg1803_1, arg1804_1, arg1805_1, arg1806_1, arg1807_1, arg1808_1, arg1809_1, arg1810_1, arg1811_1, arg1812_1, arg1813_1, arg1814_1, arg1815_1, arg1816_1, arg1817_1, arg1818_1, arg1819_1, arg1820_1, arg1821_1, arg1822_1, arg1823_1, arg1824_1, arg1825_1, arg1826_1, arg1827_1, arg1828_1, arg1829_1, arg1830_1, arg1831_1, arg1832_1, arg1833_1, arg1834_1, arg1835_1, arg1836_1, arg1837_1, arg1838_1, arg1839_1, arg1840_1, arg1841_1, arg1842_1, arg1843_1, arg1844_1, arg1845_1, arg1846_1, arg1847_1, arg1848_1, arg1849_1, arg1850_1, arg1851_1, arg1852_1, arg1853_1, arg1854_1, arg1855_1, arg1856_1, arg1857_1, arg1858_1, arg1859_1, arg1860_1, arg1861_1, arg1862_1, arg1863_1, arg1864_1, arg1865_1, arg1866_1, arg1867_1, arg1868_1, arg1869_1, arg1870_1, arg1871_1, arg1872_1, arg1873_1, arg1874_1, arg1875_1, arg1876_1, arg1877_1, arg1878_1, arg1879_1, arg1880_1, arg1881_1, arg1882_1, arg1883_1, arg1884_1, arg1885_1, arg1886_1, arg1887_1, arg1888_1, arg1889_1, arg1890_1, arg1891_1, arg1892_1, arg1893_1, arg1894_1, arg1895_1, arg1896_1, arg1897_1, arg1898_1, arg1899_1, arg1900_1, arg1901_1, arg1902_1, arg1903_1, arg1904_1, arg1905_1, arg1906_1, arg1907_1, arg1908_1, arg1909_1, arg1910_1, arg1911_1, arg1912_1, arg1913_1, arg1914_1, arg1915_1, arg1916_1, arg1917_1, arg1918_1, arg1919_1, arg1920_1, arg1921_1, arg1922_1, arg1923_1, arg1924_1, arg1925_1, arg1926_1, arg1927_1, arg1928_1, arg1929_1, arg1930_1, arg1931_1, arg1932_1, arg1933_1, arg1934_1, arg1935_1, arg1936_1, arg1937_1, arg1938_1, arg1939_1, arg1940_1, arg1941_1, arg1942_1, arg1943_1, arg1944_1, arg1945_1, arg1946_1, arg1947_1, arg1948_1, arg1949_1, arg1950_1, arg1951_1, arg1952_1, arg1953_1, arg1954_1, arg1955_1, arg1956_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (18, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg46_1, (18, ), (1, ))
    assert_size_stride(arg47_1, (18, ), (1, ))
    assert_size_stride(arg48_1, (36, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg49_1, (36, ), (1, ))
    assert_size_stride(arg50_1, (36, ), (1, ))
    assert_size_stride(arg51_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg52_1, (18, ), (1, ))
    assert_size_stride(arg53_1, (18, ), (1, ))
    assert_size_stride(arg54_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg55_1, (18, ), (1, ))
    assert_size_stride(arg56_1, (18, ), (1, ))
    assert_size_stride(arg57_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg58_1, (18, ), (1, ))
    assert_size_stride(arg59_1, (18, ), (1, ))
    assert_size_stride(arg60_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg61_1, (18, ), (1, ))
    assert_size_stride(arg62_1, (18, ), (1, ))
    assert_size_stride(arg63_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg64_1, (18, ), (1, ))
    assert_size_stride(arg65_1, (18, ), (1, ))
    assert_size_stride(arg66_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg67_1, (18, ), (1, ))
    assert_size_stride(arg68_1, (18, ), (1, ))
    assert_size_stride(arg69_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg70_1, (18, ), (1, ))
    assert_size_stride(arg71_1, (18, ), (1, ))
    assert_size_stride(arg72_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg73_1, (18, ), (1, ))
    assert_size_stride(arg74_1, (18, ), (1, ))
    assert_size_stride(arg75_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg76_1, (36, ), (1, ))
    assert_size_stride(arg77_1, (36, ), (1, ))
    assert_size_stride(arg78_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg79_1, (36, ), (1, ))
    assert_size_stride(arg80_1, (36, ), (1, ))
    assert_size_stride(arg81_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg82_1, (36, ), (1, ))
    assert_size_stride(arg83_1, (36, ), (1, ))
    assert_size_stride(arg84_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg85_1, (36, ), (1, ))
    assert_size_stride(arg86_1, (36, ), (1, ))
    assert_size_stride(arg87_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg88_1, (36, ), (1, ))
    assert_size_stride(arg89_1, (36, ), (1, ))
    assert_size_stride(arg90_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg91_1, (36, ), (1, ))
    assert_size_stride(arg92_1, (36, ), (1, ))
    assert_size_stride(arg93_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg94_1, (36, ), (1, ))
    assert_size_stride(arg95_1, (36, ), (1, ))
    assert_size_stride(arg96_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg97_1, (36, ), (1, ))
    assert_size_stride(arg98_1, (36, ), (1, ))
    assert_size_stride(arg99_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg100_1, (18, ), (1, ))
    assert_size_stride(arg101_1, (18, ), (1, ))
    assert_size_stride(arg102_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg103_1, (36, ), (1, ))
    assert_size_stride(arg104_1, (36, ), (1, ))
    assert_size_stride(arg105_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg106_1, (72, ), (1, ))
    assert_size_stride(arg107_1, (72, ), (1, ))
    assert_size_stride(arg108_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg109_1, (18, ), (1, ))
    assert_size_stride(arg110_1, (18, ), (1, ))
    assert_size_stride(arg111_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg112_1, (18, ), (1, ))
    assert_size_stride(arg113_1, (18, ), (1, ))
    assert_size_stride(arg114_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg115_1, (18, ), (1, ))
    assert_size_stride(arg116_1, (18, ), (1, ))
    assert_size_stride(arg117_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg118_1, (18, ), (1, ))
    assert_size_stride(arg119_1, (18, ), (1, ))
    assert_size_stride(arg120_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg121_1, (18, ), (1, ))
    assert_size_stride(arg122_1, (18, ), (1, ))
    assert_size_stride(arg123_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg124_1, (18, ), (1, ))
    assert_size_stride(arg125_1, (18, ), (1, ))
    assert_size_stride(arg126_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg127_1, (18, ), (1, ))
    assert_size_stride(arg128_1, (18, ), (1, ))
    assert_size_stride(arg129_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg130_1, (18, ), (1, ))
    assert_size_stride(arg131_1, (18, ), (1, ))
    assert_size_stride(arg132_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg133_1, (36, ), (1, ))
    assert_size_stride(arg134_1, (36, ), (1, ))
    assert_size_stride(arg135_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg136_1, (36, ), (1, ))
    assert_size_stride(arg137_1, (36, ), (1, ))
    assert_size_stride(arg138_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg139_1, (36, ), (1, ))
    assert_size_stride(arg140_1, (36, ), (1, ))
    assert_size_stride(arg141_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg142_1, (36, ), (1, ))
    assert_size_stride(arg143_1, (36, ), (1, ))
    assert_size_stride(arg144_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg145_1, (36, ), (1, ))
    assert_size_stride(arg146_1, (36, ), (1, ))
    assert_size_stride(arg147_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg148_1, (36, ), (1, ))
    assert_size_stride(arg149_1, (36, ), (1, ))
    assert_size_stride(arg150_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg151_1, (36, ), (1, ))
    assert_size_stride(arg152_1, (36, ), (1, ))
    assert_size_stride(arg153_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg154_1, (36, ), (1, ))
    assert_size_stride(arg155_1, (36, ), (1, ))
    assert_size_stride(arg156_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg157_1, (72, ), (1, ))
    assert_size_stride(arg158_1, (72, ), (1, ))
    assert_size_stride(arg159_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg160_1, (72, ), (1, ))
    assert_size_stride(arg161_1, (72, ), (1, ))
    assert_size_stride(arg162_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg163_1, (72, ), (1, ))
    assert_size_stride(arg164_1, (72, ), (1, ))
    assert_size_stride(arg165_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg166_1, (72, ), (1, ))
    assert_size_stride(arg167_1, (72, ), (1, ))
    assert_size_stride(arg168_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg169_1, (72, ), (1, ))
    assert_size_stride(arg170_1, (72, ), (1, ))
    assert_size_stride(arg171_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg172_1, (72, ), (1, ))
    assert_size_stride(arg173_1, (72, ), (1, ))
    assert_size_stride(arg174_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg175_1, (72, ), (1, ))
    assert_size_stride(arg176_1, (72, ), (1, ))
    assert_size_stride(arg177_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg178_1, (72, ), (1, ))
    assert_size_stride(arg179_1, (72, ), (1, ))
    assert_size_stride(arg180_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg181_1, (18, ), (1, ))
    assert_size_stride(arg182_1, (18, ), (1, ))
    assert_size_stride(arg183_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg184_1, (18, ), (1, ))
    assert_size_stride(arg185_1, (18, ), (1, ))
    assert_size_stride(arg186_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg187_1, (36, ), (1, ))
    assert_size_stride(arg188_1, (36, ), (1, ))
    assert_size_stride(arg189_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg190_1, (36, ), (1, ))
    assert_size_stride(arg191_1, (36, ), (1, ))
    assert_size_stride(arg192_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg193_1, (18, ), (1, ))
    assert_size_stride(arg194_1, (18, ), (1, ))
    assert_size_stride(arg195_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg196_1, (72, ), (1, ))
    assert_size_stride(arg197_1, (72, ), (1, ))
    assert_size_stride(arg198_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg199_1, (72, ), (1, ))
    assert_size_stride(arg200_1, (72, ), (1, ))
    assert_size_stride(arg201_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg202_1, (18, ), (1, ))
    assert_size_stride(arg203_1, (18, ), (1, ))
    assert_size_stride(arg204_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg205_1, (18, ), (1, ))
    assert_size_stride(arg206_1, (18, ), (1, ))
    assert_size_stride(arg207_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg208_1, (18, ), (1, ))
    assert_size_stride(arg209_1, (18, ), (1, ))
    assert_size_stride(arg210_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg211_1, (18, ), (1, ))
    assert_size_stride(arg212_1, (18, ), (1, ))
    assert_size_stride(arg213_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg214_1, (18, ), (1, ))
    assert_size_stride(arg215_1, (18, ), (1, ))
    assert_size_stride(arg216_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg217_1, (18, ), (1, ))
    assert_size_stride(arg218_1, (18, ), (1, ))
    assert_size_stride(arg219_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg220_1, (18, ), (1, ))
    assert_size_stride(arg221_1, (18, ), (1, ))
    assert_size_stride(arg222_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg223_1, (18, ), (1, ))
    assert_size_stride(arg224_1, (18, ), (1, ))
    assert_size_stride(arg225_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg226_1, (36, ), (1, ))
    assert_size_stride(arg227_1, (36, ), (1, ))
    assert_size_stride(arg228_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg229_1, (36, ), (1, ))
    assert_size_stride(arg230_1, (36, ), (1, ))
    assert_size_stride(arg231_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg232_1, (36, ), (1, ))
    assert_size_stride(arg233_1, (36, ), (1, ))
    assert_size_stride(arg234_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg235_1, (36, ), (1, ))
    assert_size_stride(arg236_1, (36, ), (1, ))
    assert_size_stride(arg237_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg238_1, (36, ), (1, ))
    assert_size_stride(arg239_1, (36, ), (1, ))
    assert_size_stride(arg240_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg241_1, (36, ), (1, ))
    assert_size_stride(arg242_1, (36, ), (1, ))
    assert_size_stride(arg243_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg244_1, (36, ), (1, ))
    assert_size_stride(arg245_1, (36, ), (1, ))
    assert_size_stride(arg246_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg247_1, (36, ), (1, ))
    assert_size_stride(arg248_1, (36, ), (1, ))
    assert_size_stride(arg249_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg250_1, (72, ), (1, ))
    assert_size_stride(arg251_1, (72, ), (1, ))
    assert_size_stride(arg252_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg253_1, (72, ), (1, ))
    assert_size_stride(arg254_1, (72, ), (1, ))
    assert_size_stride(arg255_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg256_1, (72, ), (1, ))
    assert_size_stride(arg257_1, (72, ), (1, ))
    assert_size_stride(arg258_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg259_1, (72, ), (1, ))
    assert_size_stride(arg260_1, (72, ), (1, ))
    assert_size_stride(arg261_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg262_1, (72, ), (1, ))
    assert_size_stride(arg263_1, (72, ), (1, ))
    assert_size_stride(arg264_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg265_1, (72, ), (1, ))
    assert_size_stride(arg266_1, (72, ), (1, ))
    assert_size_stride(arg267_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg268_1, (72, ), (1, ))
    assert_size_stride(arg269_1, (72, ), (1, ))
    assert_size_stride(arg270_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg271_1, (72, ), (1, ))
    assert_size_stride(arg272_1, (72, ), (1, ))
    assert_size_stride(arg273_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg274_1, (18, ), (1, ))
    assert_size_stride(arg275_1, (18, ), (1, ))
    assert_size_stride(arg276_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg277_1, (18, ), (1, ))
    assert_size_stride(arg278_1, (18, ), (1, ))
    assert_size_stride(arg279_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg280_1, (36, ), (1, ))
    assert_size_stride(arg281_1, (36, ), (1, ))
    assert_size_stride(arg282_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg283_1, (36, ), (1, ))
    assert_size_stride(arg284_1, (36, ), (1, ))
    assert_size_stride(arg285_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg286_1, (18, ), (1, ))
    assert_size_stride(arg287_1, (18, ), (1, ))
    assert_size_stride(arg288_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg289_1, (72, ), (1, ))
    assert_size_stride(arg290_1, (72, ), (1, ))
    assert_size_stride(arg291_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg292_1, (72, ), (1, ))
    assert_size_stride(arg293_1, (72, ), (1, ))
    assert_size_stride(arg294_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg295_1, (18, ), (1, ))
    assert_size_stride(arg296_1, (18, ), (1, ))
    assert_size_stride(arg297_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg298_1, (18, ), (1, ))
    assert_size_stride(arg299_1, (18, ), (1, ))
    assert_size_stride(arg300_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg301_1, (18, ), (1, ))
    assert_size_stride(arg302_1, (18, ), (1, ))
    assert_size_stride(arg303_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg304_1, (18, ), (1, ))
    assert_size_stride(arg305_1, (18, ), (1, ))
    assert_size_stride(arg306_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg307_1, (18, ), (1, ))
    assert_size_stride(arg308_1, (18, ), (1, ))
    assert_size_stride(arg309_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg310_1, (18, ), (1, ))
    assert_size_stride(arg311_1, (18, ), (1, ))
    assert_size_stride(arg312_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg313_1, (18, ), (1, ))
    assert_size_stride(arg314_1, (18, ), (1, ))
    assert_size_stride(arg315_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg316_1, (18, ), (1, ))
    assert_size_stride(arg317_1, (18, ), (1, ))
    assert_size_stride(arg318_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg319_1, (36, ), (1, ))
    assert_size_stride(arg320_1, (36, ), (1, ))
    assert_size_stride(arg321_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg322_1, (36, ), (1, ))
    assert_size_stride(arg323_1, (36, ), (1, ))
    assert_size_stride(arg324_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg325_1, (36, ), (1, ))
    assert_size_stride(arg326_1, (36, ), (1, ))
    assert_size_stride(arg327_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg328_1, (36, ), (1, ))
    assert_size_stride(arg329_1, (36, ), (1, ))
    assert_size_stride(arg330_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg331_1, (36, ), (1, ))
    assert_size_stride(arg332_1, (36, ), (1, ))
    assert_size_stride(arg333_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg334_1, (36, ), (1, ))
    assert_size_stride(arg335_1, (36, ), (1, ))
    assert_size_stride(arg336_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg337_1, (36, ), (1, ))
    assert_size_stride(arg338_1, (36, ), (1, ))
    assert_size_stride(arg339_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg340_1, (36, ), (1, ))
    assert_size_stride(arg341_1, (36, ), (1, ))
    assert_size_stride(arg342_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg343_1, (72, ), (1, ))
    assert_size_stride(arg344_1, (72, ), (1, ))
    assert_size_stride(arg345_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg346_1, (72, ), (1, ))
    assert_size_stride(arg347_1, (72, ), (1, ))
    assert_size_stride(arg348_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg349_1, (72, ), (1, ))
    assert_size_stride(arg350_1, (72, ), (1, ))
    assert_size_stride(arg351_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg352_1, (72, ), (1, ))
    assert_size_stride(arg353_1, (72, ), (1, ))
    assert_size_stride(arg354_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg355_1, (72, ), (1, ))
    assert_size_stride(arg356_1, (72, ), (1, ))
    assert_size_stride(arg357_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg358_1, (72, ), (1, ))
    assert_size_stride(arg359_1, (72, ), (1, ))
    assert_size_stride(arg360_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg361_1, (72, ), (1, ))
    assert_size_stride(arg362_1, (72, ), (1, ))
    assert_size_stride(arg363_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg364_1, (72, ), (1, ))
    assert_size_stride(arg365_1, (72, ), (1, ))
    assert_size_stride(arg366_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg367_1, (18, ), (1, ))
    assert_size_stride(arg368_1, (18, ), (1, ))
    assert_size_stride(arg369_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg370_1, (18, ), (1, ))
    assert_size_stride(arg371_1, (18, ), (1, ))
    assert_size_stride(arg372_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg373_1, (36, ), (1, ))
    assert_size_stride(arg374_1, (36, ), (1, ))
    assert_size_stride(arg375_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg376_1, (36, ), (1, ))
    assert_size_stride(arg377_1, (36, ), (1, ))
    assert_size_stride(arg378_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg379_1, (18, ), (1, ))
    assert_size_stride(arg380_1, (18, ), (1, ))
    assert_size_stride(arg381_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg382_1, (72, ), (1, ))
    assert_size_stride(arg383_1, (72, ), (1, ))
    assert_size_stride(arg384_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg385_1, (72, ), (1, ))
    assert_size_stride(arg386_1, (72, ), (1, ))
    assert_size_stride(arg387_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg388_1, (18, ), (1, ))
    assert_size_stride(arg389_1, (18, ), (1, ))
    assert_size_stride(arg390_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg391_1, (18, ), (1, ))
    assert_size_stride(arg392_1, (18, ), (1, ))
    assert_size_stride(arg393_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg394_1, (18, ), (1, ))
    assert_size_stride(arg395_1, (18, ), (1, ))
    assert_size_stride(arg396_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg397_1, (18, ), (1, ))
    assert_size_stride(arg398_1, (18, ), (1, ))
    assert_size_stride(arg399_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg400_1, (18, ), (1, ))
    assert_size_stride(arg401_1, (18, ), (1, ))
    assert_size_stride(arg402_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg403_1, (18, ), (1, ))
    assert_size_stride(arg404_1, (18, ), (1, ))
    assert_size_stride(arg405_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg406_1, (18, ), (1, ))
    assert_size_stride(arg407_1, (18, ), (1, ))
    assert_size_stride(arg408_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg409_1, (18, ), (1, ))
    assert_size_stride(arg410_1, (18, ), (1, ))
    assert_size_stride(arg411_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg412_1, (36, ), (1, ))
    assert_size_stride(arg413_1, (36, ), (1, ))
    assert_size_stride(arg414_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg415_1, (36, ), (1, ))
    assert_size_stride(arg416_1, (36, ), (1, ))
    assert_size_stride(arg417_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg418_1, (36, ), (1, ))
    assert_size_stride(arg419_1, (36, ), (1, ))
    assert_size_stride(arg420_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg421_1, (36, ), (1, ))
    assert_size_stride(arg422_1, (36, ), (1, ))
    assert_size_stride(arg423_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg424_1, (36, ), (1, ))
    assert_size_stride(arg425_1, (36, ), (1, ))
    assert_size_stride(arg426_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg427_1, (36, ), (1, ))
    assert_size_stride(arg428_1, (36, ), (1, ))
    assert_size_stride(arg429_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg430_1, (36, ), (1, ))
    assert_size_stride(arg431_1, (36, ), (1, ))
    assert_size_stride(arg432_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg433_1, (36, ), (1, ))
    assert_size_stride(arg434_1, (36, ), (1, ))
    assert_size_stride(arg435_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg436_1, (72, ), (1, ))
    assert_size_stride(arg437_1, (72, ), (1, ))
    assert_size_stride(arg438_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg439_1, (72, ), (1, ))
    assert_size_stride(arg440_1, (72, ), (1, ))
    assert_size_stride(arg441_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg442_1, (72, ), (1, ))
    assert_size_stride(arg443_1, (72, ), (1, ))
    assert_size_stride(arg444_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg445_1, (72, ), (1, ))
    assert_size_stride(arg446_1, (72, ), (1, ))
    assert_size_stride(arg447_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg448_1, (72, ), (1, ))
    assert_size_stride(arg449_1, (72, ), (1, ))
    assert_size_stride(arg450_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg451_1, (72, ), (1, ))
    assert_size_stride(arg452_1, (72, ), (1, ))
    assert_size_stride(arg453_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg454_1, (72, ), (1, ))
    assert_size_stride(arg455_1, (72, ), (1, ))
    assert_size_stride(arg456_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg457_1, (72, ), (1, ))
    assert_size_stride(arg458_1, (72, ), (1, ))
    assert_size_stride(arg459_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg460_1, (18, ), (1, ))
    assert_size_stride(arg461_1, (18, ), (1, ))
    assert_size_stride(arg462_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg463_1, (18, ), (1, ))
    assert_size_stride(arg464_1, (18, ), (1, ))
    assert_size_stride(arg465_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg466_1, (36, ), (1, ))
    assert_size_stride(arg467_1, (36, ), (1, ))
    assert_size_stride(arg468_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg469_1, (36, ), (1, ))
    assert_size_stride(arg470_1, (36, ), (1, ))
    assert_size_stride(arg471_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg472_1, (18, ), (1, ))
    assert_size_stride(arg473_1, (18, ), (1, ))
    assert_size_stride(arg474_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg475_1, (72, ), (1, ))
    assert_size_stride(arg476_1, (72, ), (1, ))
    assert_size_stride(arg477_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg478_1, (72, ), (1, ))
    assert_size_stride(arg479_1, (72, ), (1, ))
    assert_size_stride(arg480_1, (144, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg481_1, (144, ), (1, ))
    assert_size_stride(arg482_1, (144, ), (1, ))
    assert_size_stride(arg483_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg484_1, (18, ), (1, ))
    assert_size_stride(arg485_1, (18, ), (1, ))
    assert_size_stride(arg486_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg487_1, (18, ), (1, ))
    assert_size_stride(arg488_1, (18, ), (1, ))
    assert_size_stride(arg489_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg490_1, (18, ), (1, ))
    assert_size_stride(arg491_1, (18, ), (1, ))
    assert_size_stride(arg492_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg493_1, (18, ), (1, ))
    assert_size_stride(arg494_1, (18, ), (1, ))
    assert_size_stride(arg495_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg496_1, (18, ), (1, ))
    assert_size_stride(arg497_1, (18, ), (1, ))
    assert_size_stride(arg498_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg499_1, (18, ), (1, ))
    assert_size_stride(arg500_1, (18, ), (1, ))
    assert_size_stride(arg501_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg502_1, (18, ), (1, ))
    assert_size_stride(arg503_1, (18, ), (1, ))
    assert_size_stride(arg504_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg505_1, (18, ), (1, ))
    assert_size_stride(arg506_1, (18, ), (1, ))
    assert_size_stride(arg507_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg508_1, (36, ), (1, ))
    assert_size_stride(arg509_1, (36, ), (1, ))
    assert_size_stride(arg510_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg511_1, (36, ), (1, ))
    assert_size_stride(arg512_1, (36, ), (1, ))
    assert_size_stride(arg513_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg514_1, (36, ), (1, ))
    assert_size_stride(arg515_1, (36, ), (1, ))
    assert_size_stride(arg516_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg517_1, (36, ), (1, ))
    assert_size_stride(arg518_1, (36, ), (1, ))
    assert_size_stride(arg519_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg520_1, (36, ), (1, ))
    assert_size_stride(arg521_1, (36, ), (1, ))
    assert_size_stride(arg522_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg523_1, (36, ), (1, ))
    assert_size_stride(arg524_1, (36, ), (1, ))
    assert_size_stride(arg525_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg526_1, (36, ), (1, ))
    assert_size_stride(arg527_1, (36, ), (1, ))
    assert_size_stride(arg528_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg529_1, (36, ), (1, ))
    assert_size_stride(arg530_1, (36, ), (1, ))
    assert_size_stride(arg531_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg532_1, (72, ), (1, ))
    assert_size_stride(arg533_1, (72, ), (1, ))
    assert_size_stride(arg534_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg535_1, (72, ), (1, ))
    assert_size_stride(arg536_1, (72, ), (1, ))
    assert_size_stride(arg537_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg538_1, (72, ), (1, ))
    assert_size_stride(arg539_1, (72, ), (1, ))
    assert_size_stride(arg540_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg541_1, (72, ), (1, ))
    assert_size_stride(arg542_1, (72, ), (1, ))
    assert_size_stride(arg543_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg544_1, (72, ), (1, ))
    assert_size_stride(arg545_1, (72, ), (1, ))
    assert_size_stride(arg546_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg547_1, (72, ), (1, ))
    assert_size_stride(arg548_1, (72, ), (1, ))
    assert_size_stride(arg549_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg550_1, (72, ), (1, ))
    assert_size_stride(arg551_1, (72, ), (1, ))
    assert_size_stride(arg552_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg553_1, (72, ), (1, ))
    assert_size_stride(arg554_1, (72, ), (1, ))
    assert_size_stride(arg555_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg556_1, (144, ), (1, ))
    assert_size_stride(arg557_1, (144, ), (1, ))
    assert_size_stride(arg558_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg559_1, (144, ), (1, ))
    assert_size_stride(arg560_1, (144, ), (1, ))
    assert_size_stride(arg561_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg562_1, (144, ), (1, ))
    assert_size_stride(arg563_1, (144, ), (1, ))
    assert_size_stride(arg564_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg565_1, (144, ), (1, ))
    assert_size_stride(arg566_1, (144, ), (1, ))
    assert_size_stride(arg567_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg568_1, (144, ), (1, ))
    assert_size_stride(arg569_1, (144, ), (1, ))
    assert_size_stride(arg570_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg571_1, (144, ), (1, ))
    assert_size_stride(arg572_1, (144, ), (1, ))
    assert_size_stride(arg573_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg574_1, (144, ), (1, ))
    assert_size_stride(arg575_1, (144, ), (1, ))
    assert_size_stride(arg576_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg577_1, (144, ), (1, ))
    assert_size_stride(arg578_1, (144, ), (1, ))
    assert_size_stride(arg579_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg580_1, (18, ), (1, ))
    assert_size_stride(arg581_1, (18, ), (1, ))
    assert_size_stride(arg582_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg583_1, (18, ), (1, ))
    assert_size_stride(arg584_1, (18, ), (1, ))
    assert_size_stride(arg585_1, (18, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg586_1, (18, ), (1, ))
    assert_size_stride(arg587_1, (18, ), (1, ))
    assert_size_stride(arg588_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg589_1, (36, ), (1, ))
    assert_size_stride(arg590_1, (36, ), (1, ))
    assert_size_stride(arg591_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg592_1, (36, ), (1, ))
    assert_size_stride(arg593_1, (36, ), (1, ))
    assert_size_stride(arg594_1, (36, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg595_1, (36, ), (1, ))
    assert_size_stride(arg596_1, (36, ), (1, ))
    assert_size_stride(arg597_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg598_1, (18, ), (1, ))
    assert_size_stride(arg599_1, (18, ), (1, ))
    assert_size_stride(arg600_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg601_1, (72, ), (1, ))
    assert_size_stride(arg602_1, (72, ), (1, ))
    assert_size_stride(arg603_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg604_1, (72, ), (1, ))
    assert_size_stride(arg605_1, (72, ), (1, ))
    assert_size_stride(arg606_1, (72, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg607_1, (72, ), (1, ))
    assert_size_stride(arg608_1, (72, ), (1, ))
    assert_size_stride(arg609_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg610_1, (18, ), (1, ))
    assert_size_stride(arg611_1, (18, ), (1, ))
    assert_size_stride(arg612_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg613_1, (18, ), (1, ))
    assert_size_stride(arg614_1, (18, ), (1, ))
    assert_size_stride(arg615_1, (144, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg616_1, (144, ), (1, ))
    assert_size_stride(arg617_1, (144, ), (1, ))
    assert_size_stride(arg618_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg619_1, (36, ), (1, ))
    assert_size_stride(arg620_1, (36, ), (1, ))
    assert_size_stride(arg621_1, (144, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg622_1, (144, ), (1, ))
    assert_size_stride(arg623_1, (144, ), (1, ))
    assert_size_stride(arg624_1, (144, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg625_1, (144, ), (1, ))
    assert_size_stride(arg626_1, (144, ), (1, ))
    assert_size_stride(arg627_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg628_1, (18, ), (1, ))
    assert_size_stride(arg629_1, (18, ), (1, ))
    assert_size_stride(arg630_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg631_1, (18, ), (1, ))
    assert_size_stride(arg632_1, (18, ), (1, ))
    assert_size_stride(arg633_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg634_1, (18, ), (1, ))
    assert_size_stride(arg635_1, (18, ), (1, ))
    assert_size_stride(arg636_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg637_1, (18, ), (1, ))
    assert_size_stride(arg638_1, (18, ), (1, ))
    assert_size_stride(arg639_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg640_1, (18, ), (1, ))
    assert_size_stride(arg641_1, (18, ), (1, ))
    assert_size_stride(arg642_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg643_1, (18, ), (1, ))
    assert_size_stride(arg644_1, (18, ), (1, ))
    assert_size_stride(arg645_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg646_1, (18, ), (1, ))
    assert_size_stride(arg647_1, (18, ), (1, ))
    assert_size_stride(arg648_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg649_1, (18, ), (1, ))
    assert_size_stride(arg650_1, (18, ), (1, ))
    assert_size_stride(arg651_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg652_1, (36, ), (1, ))
    assert_size_stride(arg653_1, (36, ), (1, ))
    assert_size_stride(arg654_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg655_1, (36, ), (1, ))
    assert_size_stride(arg656_1, (36, ), (1, ))
    assert_size_stride(arg657_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg658_1, (36, ), (1, ))
    assert_size_stride(arg659_1, (36, ), (1, ))
    assert_size_stride(arg660_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg661_1, (36, ), (1, ))
    assert_size_stride(arg662_1, (36, ), (1, ))
    assert_size_stride(arg663_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg664_1, (36, ), (1, ))
    assert_size_stride(arg665_1, (36, ), (1, ))
    assert_size_stride(arg666_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg667_1, (36, ), (1, ))
    assert_size_stride(arg668_1, (36, ), (1, ))
    assert_size_stride(arg669_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg670_1, (36, ), (1, ))
    assert_size_stride(arg671_1, (36, ), (1, ))
    assert_size_stride(arg672_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg673_1, (36, ), (1, ))
    assert_size_stride(arg674_1, (36, ), (1, ))
    assert_size_stride(arg675_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg676_1, (72, ), (1, ))
    assert_size_stride(arg677_1, (72, ), (1, ))
    assert_size_stride(arg678_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg679_1, (72, ), (1, ))
    assert_size_stride(arg680_1, (72, ), (1, ))
    assert_size_stride(arg681_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg682_1, (72, ), (1, ))
    assert_size_stride(arg683_1, (72, ), (1, ))
    assert_size_stride(arg684_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg685_1, (72, ), (1, ))
    assert_size_stride(arg686_1, (72, ), (1, ))
    assert_size_stride(arg687_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg688_1, (72, ), (1, ))
    assert_size_stride(arg689_1, (72, ), (1, ))
    assert_size_stride(arg690_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg691_1, (72, ), (1, ))
    assert_size_stride(arg692_1, (72, ), (1, ))
    assert_size_stride(arg693_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg694_1, (72, ), (1, ))
    assert_size_stride(arg695_1, (72, ), (1, ))
    assert_size_stride(arg696_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg697_1, (72, ), (1, ))
    assert_size_stride(arg698_1, (72, ), (1, ))
    assert_size_stride(arg699_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg700_1, (144, ), (1, ))
    assert_size_stride(arg701_1, (144, ), (1, ))
    assert_size_stride(arg702_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg703_1, (144, ), (1, ))
    assert_size_stride(arg704_1, (144, ), (1, ))
    assert_size_stride(arg705_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg706_1, (144, ), (1, ))
    assert_size_stride(arg707_1, (144, ), (1, ))
    assert_size_stride(arg708_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg709_1, (144, ), (1, ))
    assert_size_stride(arg710_1, (144, ), (1, ))
    assert_size_stride(arg711_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg712_1, (144, ), (1, ))
    assert_size_stride(arg713_1, (144, ), (1, ))
    assert_size_stride(arg714_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg715_1, (144, ), (1, ))
    assert_size_stride(arg716_1, (144, ), (1, ))
    assert_size_stride(arg717_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg718_1, (144, ), (1, ))
    assert_size_stride(arg719_1, (144, ), (1, ))
    assert_size_stride(arg720_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg721_1, (144, ), (1, ))
    assert_size_stride(arg722_1, (144, ), (1, ))
    assert_size_stride(arg723_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg724_1, (18, ), (1, ))
    assert_size_stride(arg725_1, (18, ), (1, ))
    assert_size_stride(arg726_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg727_1, (18, ), (1, ))
    assert_size_stride(arg728_1, (18, ), (1, ))
    assert_size_stride(arg729_1, (18, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg730_1, (18, ), (1, ))
    assert_size_stride(arg731_1, (18, ), (1, ))
    assert_size_stride(arg732_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg733_1, (36, ), (1, ))
    assert_size_stride(arg734_1, (36, ), (1, ))
    assert_size_stride(arg735_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg736_1, (36, ), (1, ))
    assert_size_stride(arg737_1, (36, ), (1, ))
    assert_size_stride(arg738_1, (36, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg739_1, (36, ), (1, ))
    assert_size_stride(arg740_1, (36, ), (1, ))
    assert_size_stride(arg741_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg742_1, (18, ), (1, ))
    assert_size_stride(arg743_1, (18, ), (1, ))
    assert_size_stride(arg744_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg745_1, (72, ), (1, ))
    assert_size_stride(arg746_1, (72, ), (1, ))
    assert_size_stride(arg747_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg748_1, (72, ), (1, ))
    assert_size_stride(arg749_1, (72, ), (1, ))
    assert_size_stride(arg750_1, (72, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg751_1, (72, ), (1, ))
    assert_size_stride(arg752_1, (72, ), (1, ))
    assert_size_stride(arg753_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg754_1, (18, ), (1, ))
    assert_size_stride(arg755_1, (18, ), (1, ))
    assert_size_stride(arg756_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg757_1, (18, ), (1, ))
    assert_size_stride(arg758_1, (18, ), (1, ))
    assert_size_stride(arg759_1, (144, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg760_1, (144, ), (1, ))
    assert_size_stride(arg761_1, (144, ), (1, ))
    assert_size_stride(arg762_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg763_1, (36, ), (1, ))
    assert_size_stride(arg764_1, (36, ), (1, ))
    assert_size_stride(arg765_1, (144, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg766_1, (144, ), (1, ))
    assert_size_stride(arg767_1, (144, ), (1, ))
    assert_size_stride(arg768_1, (144, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg769_1, (144, ), (1, ))
    assert_size_stride(arg770_1, (144, ), (1, ))
    assert_size_stride(arg771_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg772_1, (18, ), (1, ))
    assert_size_stride(arg773_1, (18, ), (1, ))
    assert_size_stride(arg774_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg775_1, (18, ), (1, ))
    assert_size_stride(arg776_1, (18, ), (1, ))
    assert_size_stride(arg777_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg778_1, (18, ), (1, ))
    assert_size_stride(arg779_1, (18, ), (1, ))
    assert_size_stride(arg780_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg781_1, (18, ), (1, ))
    assert_size_stride(arg782_1, (18, ), (1, ))
    assert_size_stride(arg783_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg784_1, (18, ), (1, ))
    assert_size_stride(arg785_1, (18, ), (1, ))
    assert_size_stride(arg786_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg787_1, (18, ), (1, ))
    assert_size_stride(arg788_1, (18, ), (1, ))
    assert_size_stride(arg789_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg790_1, (18, ), (1, ))
    assert_size_stride(arg791_1, (18, ), (1, ))
    assert_size_stride(arg792_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg793_1, (18, ), (1, ))
    assert_size_stride(arg794_1, (18, ), (1, ))
    assert_size_stride(arg795_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg796_1, (36, ), (1, ))
    assert_size_stride(arg797_1, (36, ), (1, ))
    assert_size_stride(arg798_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg799_1, (36, ), (1, ))
    assert_size_stride(arg800_1, (36, ), (1, ))
    assert_size_stride(arg801_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg802_1, (36, ), (1, ))
    assert_size_stride(arg803_1, (36, ), (1, ))
    assert_size_stride(arg804_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg805_1, (36, ), (1, ))
    assert_size_stride(arg806_1, (36, ), (1, ))
    assert_size_stride(arg807_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg808_1, (36, ), (1, ))
    assert_size_stride(arg809_1, (36, ), (1, ))
    assert_size_stride(arg810_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg811_1, (36, ), (1, ))
    assert_size_stride(arg812_1, (36, ), (1, ))
    assert_size_stride(arg813_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg814_1, (36, ), (1, ))
    assert_size_stride(arg815_1, (36, ), (1, ))
    assert_size_stride(arg816_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg817_1, (36, ), (1, ))
    assert_size_stride(arg818_1, (36, ), (1, ))
    assert_size_stride(arg819_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg820_1, (72, ), (1, ))
    assert_size_stride(arg821_1, (72, ), (1, ))
    assert_size_stride(arg822_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg823_1, (72, ), (1, ))
    assert_size_stride(arg824_1, (72, ), (1, ))
    assert_size_stride(arg825_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg826_1, (72, ), (1, ))
    assert_size_stride(arg827_1, (72, ), (1, ))
    assert_size_stride(arg828_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg829_1, (72, ), (1, ))
    assert_size_stride(arg830_1, (72, ), (1, ))
    assert_size_stride(arg831_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg832_1, (72, ), (1, ))
    assert_size_stride(arg833_1, (72, ), (1, ))
    assert_size_stride(arg834_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg835_1, (72, ), (1, ))
    assert_size_stride(arg836_1, (72, ), (1, ))
    assert_size_stride(arg837_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg838_1, (72, ), (1, ))
    assert_size_stride(arg839_1, (72, ), (1, ))
    assert_size_stride(arg840_1, (72, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg841_1, (72, ), (1, ))
    assert_size_stride(arg842_1, (72, ), (1, ))
    assert_size_stride(arg843_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg844_1, (144, ), (1, ))
    assert_size_stride(arg845_1, (144, ), (1, ))
    assert_size_stride(arg846_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg847_1, (144, ), (1, ))
    assert_size_stride(arg848_1, (144, ), (1, ))
    assert_size_stride(arg849_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg850_1, (144, ), (1, ))
    assert_size_stride(arg851_1, (144, ), (1, ))
    assert_size_stride(arg852_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg853_1, (144, ), (1, ))
    assert_size_stride(arg854_1, (144, ), (1, ))
    assert_size_stride(arg855_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg856_1, (144, ), (1, ))
    assert_size_stride(arg857_1, (144, ), (1, ))
    assert_size_stride(arg858_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg859_1, (144, ), (1, ))
    assert_size_stride(arg860_1, (144, ), (1, ))
    assert_size_stride(arg861_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg862_1, (144, ), (1, ))
    assert_size_stride(arg863_1, (144, ), (1, ))
    assert_size_stride(arg864_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg865_1, (144, ), (1, ))
    assert_size_stride(arg866_1, (144, ), (1, ))
    assert_size_stride(arg867_1, (18, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg868_1, (18, ), (1, ))
    assert_size_stride(arg869_1, (18, ), (1, ))
    assert_size_stride(arg870_1, (18, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg871_1, (18, ), (1, ))
    assert_size_stride(arg872_1, (18, ), (1, ))
    assert_size_stride(arg873_1, (18, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg874_1, (18, ), (1, ))
    assert_size_stride(arg875_1, (18, ), (1, ))
    assert_size_stride(arg876_1, (36, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg877_1, (36, ), (1, ))
    assert_size_stride(arg878_1, (36, ), (1, ))
    assert_size_stride(arg879_1, (36, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg880_1, (36, ), (1, ))
    assert_size_stride(arg881_1, (36, ), (1, ))
    assert_size_stride(arg882_1, (36, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg883_1, (36, ), (1, ))
    assert_size_stride(arg884_1, (36, ), (1, ))
    assert_size_stride(arg885_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg886_1, (18, ), (1, ))
    assert_size_stride(arg887_1, (18, ), (1, ))
    assert_size_stride(arg888_1, (72, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg889_1, (72, ), (1, ))
    assert_size_stride(arg890_1, (72, ), (1, ))
    assert_size_stride(arg891_1, (72, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg892_1, (72, ), (1, ))
    assert_size_stride(arg893_1, (72, ), (1, ))
    assert_size_stride(arg894_1, (72, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg895_1, (72, ), (1, ))
    assert_size_stride(arg896_1, (72, ), (1, ))
    assert_size_stride(arg897_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg898_1, (18, ), (1, ))
    assert_size_stride(arg899_1, (18, ), (1, ))
    assert_size_stride(arg900_1, (18, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg901_1, (18, ), (1, ))
    assert_size_stride(arg902_1, (18, ), (1, ))
    assert_size_stride(arg903_1, (144, 18, 3, 3), (162, 9, 3, 1))
    assert_size_stride(arg904_1, (144, ), (1, ))
    assert_size_stride(arg905_1, (144, ), (1, ))
    assert_size_stride(arg906_1, (36, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg907_1, (36, ), (1, ))
    assert_size_stride(arg908_1, (36, ), (1, ))
    assert_size_stride(arg909_1, (144, 36, 3, 3), (324, 9, 3, 1))
    assert_size_stride(arg910_1, (144, ), (1, ))
    assert_size_stride(arg911_1, (144, ), (1, ))
    assert_size_stride(arg912_1, (144, 72, 3, 3), (648, 9, 3, 1))
    assert_size_stride(arg913_1, (144, ), (1, ))
    assert_size_stride(arg914_1, (144, ), (1, ))
    assert_size_stride(arg915_1, (32, 18, 1, 1), (18, 1, 1, 1))
    assert_size_stride(arg916_1, (32, ), (1, ))
    assert_size_stride(arg917_1, (32, ), (1, ))
    assert_size_stride(arg918_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg919_1, (32, ), (1, ))
    assert_size_stride(arg920_1, (32, ), (1, ))
    assert_size_stride(arg921_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg922_1, (128, ), (1, ))
    assert_size_stride(arg923_1, (128, ), (1, ))
    assert_size_stride(arg924_1, (128, 18, 1, 1), (18, 1, 1, 1))
    assert_size_stride(arg925_1, (128, ), (1, ))
    assert_size_stride(arg926_1, (128, ), (1, ))
    assert_size_stride(arg927_1, (64, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg928_1, (64, ), (1, ))
    assert_size_stride(arg929_1, (64, ), (1, ))
    assert_size_stride(arg930_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg931_1, (64, ), (1, ))
    assert_size_stride(arg932_1, (64, ), (1, ))
    assert_size_stride(arg933_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg934_1, (256, ), (1, ))
    assert_size_stride(arg935_1, (256, ), (1, ))
    assert_size_stride(arg936_1, (256, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg937_1, (256, ), (1, ))
    assert_size_stride(arg938_1, (256, ), (1, ))
    assert_size_stride(arg939_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg940_1, (256, ), (1, ))
    assert_size_stride(arg941_1, (256, ), (1, ))
    assert_size_stride(arg942_1, (256, ), (1, ))
    assert_size_stride(arg943_1, (128, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg944_1, (128, ), (1, ))
    assert_size_stride(arg945_1, (128, ), (1, ))
    assert_size_stride(arg946_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg947_1, (128, ), (1, ))
    assert_size_stride(arg948_1, (128, ), (1, ))
    assert_size_stride(arg949_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg950_1, (512, ), (1, ))
    assert_size_stride(arg951_1, (512, ), (1, ))
    assert_size_stride(arg952_1, (512, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg953_1, (512, ), (1, ))
    assert_size_stride(arg954_1, (512, ), (1, ))
    assert_size_stride(arg955_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg956_1, (512, ), (1, ))
    assert_size_stride(arg957_1, (512, ), (1, ))
    assert_size_stride(arg958_1, (512, ), (1, ))
    assert_size_stride(arg959_1, (256, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg960_1, (256, ), (1, ))
    assert_size_stride(arg961_1, (256, ), (1, ))
    assert_size_stride(arg962_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg963_1, (256, ), (1, ))
    assert_size_stride(arg964_1, (256, ), (1, ))
    assert_size_stride(arg965_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg966_1, (1024, ), (1, ))
    assert_size_stride(arg967_1, (1024, ), (1, ))
    assert_size_stride(arg968_1, (1024, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg969_1, (1024, ), (1, ))
    assert_size_stride(arg970_1, (1024, ), (1, ))
    assert_size_stride(arg971_1, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg972_1, (1024, ), (1, ))
    assert_size_stride(arg973_1, (1024, ), (1, ))
    assert_size_stride(arg974_1, (1024, ), (1, ))
    assert_size_stride(arg975_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg976_1, (2048, ), (1, ))
    assert_size_stride(arg977_1, (2048, ), (1, ))
    assert_size_stride(arg978_1, (2048, ), (1, ))
    assert_size_stride(arg979_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg980_1, (1000, ), (1, ))
    assert_size_stride(arg981_1, (64, ), (1, ))
    assert_size_stride(arg982_1, (64, ), (1, ))
    assert_size_stride(arg983_1, (), ())
    assert_size_stride(arg984_1, (64, ), (1, ))
    assert_size_stride(arg985_1, (64, ), (1, ))
    assert_size_stride(arg986_1, (), ())
    assert_size_stride(arg987_1, (64, ), (1, ))
    assert_size_stride(arg988_1, (64, ), (1, ))
    assert_size_stride(arg989_1, (), ())
    assert_size_stride(arg990_1, (64, ), (1, ))
    assert_size_stride(arg991_1, (64, ), (1, ))
    assert_size_stride(arg992_1, (), ())
    assert_size_stride(arg993_1, (256, ), (1, ))
    assert_size_stride(arg994_1, (256, ), (1, ))
    assert_size_stride(arg995_1, (), ())
    assert_size_stride(arg996_1, (256, ), (1, ))
    assert_size_stride(arg997_1, (256, ), (1, ))
    assert_size_stride(arg998_1, (), ())
    assert_size_stride(arg999_1, (64, ), (1, ))
    assert_size_stride(arg1000_1, (64, ), (1, ))
    assert_size_stride(arg1001_1, (), ())
    assert_size_stride(arg1002_1, (64, ), (1, ))
    assert_size_stride(arg1003_1, (64, ), (1, ))
    assert_size_stride(arg1004_1, (), ())
    assert_size_stride(arg1005_1, (256, ), (1, ))
    assert_size_stride(arg1006_1, (256, ), (1, ))
    assert_size_stride(arg1007_1, (), ())
    assert_size_stride(arg1008_1, (64, ), (1, ))
    assert_size_stride(arg1009_1, (64, ), (1, ))
    assert_size_stride(arg1010_1, (), ())
    assert_size_stride(arg1011_1, (64, ), (1, ))
    assert_size_stride(arg1012_1, (64, ), (1, ))
    assert_size_stride(arg1013_1, (), ())
    assert_size_stride(arg1014_1, (256, ), (1, ))
    assert_size_stride(arg1015_1, (256, ), (1, ))
    assert_size_stride(arg1016_1, (), ())
    assert_size_stride(arg1017_1, (64, ), (1, ))
    assert_size_stride(arg1018_1, (64, ), (1, ))
    assert_size_stride(arg1019_1, (), ())
    assert_size_stride(arg1020_1, (64, ), (1, ))
    assert_size_stride(arg1021_1, (64, ), (1, ))
    assert_size_stride(arg1022_1, (), ())
    assert_size_stride(arg1023_1, (256, ), (1, ))
    assert_size_stride(arg1024_1, (256, ), (1, ))
    assert_size_stride(arg1025_1, (), ())
    assert_size_stride(arg1026_1, (18, ), (1, ))
    assert_size_stride(arg1027_1, (18, ), (1, ))
    assert_size_stride(arg1028_1, (), ())
    assert_size_stride(arg1029_1, (36, ), (1, ))
    assert_size_stride(arg1030_1, (36, ), (1, ))
    assert_size_stride(arg1031_1, (), ())
    assert_size_stride(arg1032_1, (18, ), (1, ))
    assert_size_stride(arg1033_1, (18, ), (1, ))
    assert_size_stride(arg1034_1, (), ())
    assert_size_stride(arg1035_1, (18, ), (1, ))
    assert_size_stride(arg1036_1, (18, ), (1, ))
    assert_size_stride(arg1037_1, (), ())
    assert_size_stride(arg1038_1, (18, ), (1, ))
    assert_size_stride(arg1039_1, (18, ), (1, ))
    assert_size_stride(arg1040_1, (), ())
    assert_size_stride(arg1041_1, (18, ), (1, ))
    assert_size_stride(arg1042_1, (18, ), (1, ))
    assert_size_stride(arg1043_1, (), ())
    assert_size_stride(arg1044_1, (18, ), (1, ))
    assert_size_stride(arg1045_1, (18, ), (1, ))
    assert_size_stride(arg1046_1, (), ())
    assert_size_stride(arg1047_1, (18, ), (1, ))
    assert_size_stride(arg1048_1, (18, ), (1, ))
    assert_size_stride(arg1049_1, (), ())
    assert_size_stride(arg1050_1, (18, ), (1, ))
    assert_size_stride(arg1051_1, (18, ), (1, ))
    assert_size_stride(arg1052_1, (), ())
    assert_size_stride(arg1053_1, (18, ), (1, ))
    assert_size_stride(arg1054_1, (18, ), (1, ))
    assert_size_stride(arg1055_1, (), ())
    assert_size_stride(arg1056_1, (36, ), (1, ))
    assert_size_stride(arg1057_1, (36, ), (1, ))
    assert_size_stride(arg1058_1, (), ())
    assert_size_stride(arg1059_1, (36, ), (1, ))
    assert_size_stride(arg1060_1, (36, ), (1, ))
    assert_size_stride(arg1061_1, (), ())
    assert_size_stride(arg1062_1, (36, ), (1, ))
    assert_size_stride(arg1063_1, (36, ), (1, ))
    assert_size_stride(arg1064_1, (), ())
    assert_size_stride(arg1065_1, (36, ), (1, ))
    assert_size_stride(arg1066_1, (36, ), (1, ))
    assert_size_stride(arg1067_1, (), ())
    assert_size_stride(arg1068_1, (36, ), (1, ))
    assert_size_stride(arg1069_1, (36, ), (1, ))
    assert_size_stride(arg1070_1, (), ())
    assert_size_stride(arg1071_1, (36, ), (1, ))
    assert_size_stride(arg1072_1, (36, ), (1, ))
    assert_size_stride(arg1073_1, (), ())
    assert_size_stride(arg1074_1, (36, ), (1, ))
    assert_size_stride(arg1075_1, (36, ), (1, ))
    assert_size_stride(arg1076_1, (), ())
    assert_size_stride(arg1077_1, (36, ), (1, ))
    assert_size_stride(arg1078_1, (36, ), (1, ))
    assert_size_stride(arg1079_1, (), ())
    assert_size_stride(arg1080_1, (18, ), (1, ))
    assert_size_stride(arg1081_1, (18, ), (1, ))
    assert_size_stride(arg1082_1, (), ())
    assert_size_stride(arg1083_1, (36, ), (1, ))
    assert_size_stride(arg1084_1, (36, ), (1, ))
    assert_size_stride(arg1085_1, (), ())
    assert_size_stride(arg1086_1, (72, ), (1, ))
    assert_size_stride(arg1087_1, (72, ), (1, ))
    assert_size_stride(arg1088_1, (), ())
    assert_size_stride(arg1089_1, (18, ), (1, ))
    assert_size_stride(arg1090_1, (18, ), (1, ))
    assert_size_stride(arg1091_1, (), ())
    assert_size_stride(arg1092_1, (18, ), (1, ))
    assert_size_stride(arg1093_1, (18, ), (1, ))
    assert_size_stride(arg1094_1, (), ())
    assert_size_stride(arg1095_1, (18, ), (1, ))
    assert_size_stride(arg1096_1, (18, ), (1, ))
    assert_size_stride(arg1097_1, (), ())
    assert_size_stride(arg1098_1, (18, ), (1, ))
    assert_size_stride(arg1099_1, (18, ), (1, ))
    assert_size_stride(arg1100_1, (), ())
    assert_size_stride(arg1101_1, (18, ), (1, ))
    assert_size_stride(arg1102_1, (18, ), (1, ))
    assert_size_stride(arg1103_1, (), ())
    assert_size_stride(arg1104_1, (18, ), (1, ))
    assert_size_stride(arg1105_1, (18, ), (1, ))
    assert_size_stride(arg1106_1, (), ())
    assert_size_stride(arg1107_1, (18, ), (1, ))
    assert_size_stride(arg1108_1, (18, ), (1, ))
    assert_size_stride(arg1109_1, (), ())
    assert_size_stride(arg1110_1, (18, ), (1, ))
    assert_size_stride(arg1111_1, (18, ), (1, ))
    assert_size_stride(arg1112_1, (), ())
    assert_size_stride(arg1113_1, (36, ), (1, ))
    assert_size_stride(arg1114_1, (36, ), (1, ))
    assert_size_stride(arg1115_1, (), ())
    assert_size_stride(arg1116_1, (36, ), (1, ))
    assert_size_stride(arg1117_1, (36, ), (1, ))
    assert_size_stride(arg1118_1, (), ())
    assert_size_stride(arg1119_1, (36, ), (1, ))
    assert_size_stride(arg1120_1, (36, ), (1, ))
    assert_size_stride(arg1121_1, (), ())
    assert_size_stride(arg1122_1, (36, ), (1, ))
    assert_size_stride(arg1123_1, (36, ), (1, ))
    assert_size_stride(arg1124_1, (), ())
    assert_size_stride(arg1125_1, (36, ), (1, ))
    assert_size_stride(arg1126_1, (36, ), (1, ))
    assert_size_stride(arg1127_1, (), ())
    assert_size_stride(arg1128_1, (36, ), (1, ))
    assert_size_stride(arg1129_1, (36, ), (1, ))
    assert_size_stride(arg1130_1, (), ())
    assert_size_stride(arg1131_1, (36, ), (1, ))
    assert_size_stride(arg1132_1, (36, ), (1, ))
    assert_size_stride(arg1133_1, (), ())
    assert_size_stride(arg1134_1, (36, ), (1, ))
    assert_size_stride(arg1135_1, (36, ), (1, ))
    assert_size_stride(arg1136_1, (), ())
    assert_size_stride(arg1137_1, (72, ), (1, ))
    assert_size_stride(arg1138_1, (72, ), (1, ))
    assert_size_stride(arg1139_1, (), ())
    assert_size_stride(arg1140_1, (72, ), (1, ))
    assert_size_stride(arg1141_1, (72, ), (1, ))
    assert_size_stride(arg1142_1, (), ())
    assert_size_stride(arg1143_1, (72, ), (1, ))
    assert_size_stride(arg1144_1, (72, ), (1, ))
    assert_size_stride(arg1145_1, (), ())
    assert_size_stride(arg1146_1, (72, ), (1, ))
    assert_size_stride(arg1147_1, (72, ), (1, ))
    assert_size_stride(arg1148_1, (), ())
    assert_size_stride(arg1149_1, (72, ), (1, ))
    assert_size_stride(arg1150_1, (72, ), (1, ))
    assert_size_stride(arg1151_1, (), ())
    assert_size_stride(arg1152_1, (72, ), (1, ))
    assert_size_stride(arg1153_1, (72, ), (1, ))
    assert_size_stride(arg1154_1, (), ())
    assert_size_stride(arg1155_1, (72, ), (1, ))
    assert_size_stride(arg1156_1, (72, ), (1, ))
    assert_size_stride(arg1157_1, (), ())
    assert_size_stride(arg1158_1, (72, ), (1, ))
    assert_size_stride(arg1159_1, (72, ), (1, ))
    assert_size_stride(arg1160_1, (), ())
    assert_size_stride(arg1161_1, (18, ), (1, ))
    assert_size_stride(arg1162_1, (18, ), (1, ))
    assert_size_stride(arg1163_1, (), ())
    assert_size_stride(arg1164_1, (18, ), (1, ))
    assert_size_stride(arg1165_1, (18, ), (1, ))
    assert_size_stride(arg1166_1, (), ())
    assert_size_stride(arg1167_1, (36, ), (1, ))
    assert_size_stride(arg1168_1, (36, ), (1, ))
    assert_size_stride(arg1169_1, (), ())
    assert_size_stride(arg1170_1, (36, ), (1, ))
    assert_size_stride(arg1171_1, (36, ), (1, ))
    assert_size_stride(arg1172_1, (), ())
    assert_size_stride(arg1173_1, (18, ), (1, ))
    assert_size_stride(arg1174_1, (18, ), (1, ))
    assert_size_stride(arg1175_1, (), ())
    assert_size_stride(arg1176_1, (72, ), (1, ))
    assert_size_stride(arg1177_1, (72, ), (1, ))
    assert_size_stride(arg1178_1, (), ())
    assert_size_stride(arg1179_1, (72, ), (1, ))
    assert_size_stride(arg1180_1, (72, ), (1, ))
    assert_size_stride(arg1181_1, (), ())
    assert_size_stride(arg1182_1, (18, ), (1, ))
    assert_size_stride(arg1183_1, (18, ), (1, ))
    assert_size_stride(arg1184_1, (), ())
    assert_size_stride(arg1185_1, (18, ), (1, ))
    assert_size_stride(arg1186_1, (18, ), (1, ))
    assert_size_stride(arg1187_1, (), ())
    assert_size_stride(arg1188_1, (18, ), (1, ))
    assert_size_stride(arg1189_1, (18, ), (1, ))
    assert_size_stride(arg1190_1, (), ())
    assert_size_stride(arg1191_1, (18, ), (1, ))
    assert_size_stride(arg1192_1, (18, ), (1, ))
    assert_size_stride(arg1193_1, (), ())
    assert_size_stride(arg1194_1, (18, ), (1, ))
    assert_size_stride(arg1195_1, (18, ), (1, ))
    assert_size_stride(arg1196_1, (), ())
    assert_size_stride(arg1197_1, (18, ), (1, ))
    assert_size_stride(arg1198_1, (18, ), (1, ))
    assert_size_stride(arg1199_1, (), ())
    assert_size_stride(arg1200_1, (18, ), (1, ))
    assert_size_stride(arg1201_1, (18, ), (1, ))
    assert_size_stride(arg1202_1, (), ())
    assert_size_stride(arg1203_1, (18, ), (1, ))
    assert_size_stride(arg1204_1, (18, ), (1, ))
    assert_size_stride(arg1205_1, (), ())
    assert_size_stride(arg1206_1, (36, ), (1, ))
    assert_size_stride(arg1207_1, (36, ), (1, ))
    assert_size_stride(arg1208_1, (), ())
    assert_size_stride(arg1209_1, (36, ), (1, ))
    assert_size_stride(arg1210_1, (36, ), (1, ))
    assert_size_stride(arg1211_1, (), ())
    assert_size_stride(arg1212_1, (36, ), (1, ))
    assert_size_stride(arg1213_1, (36, ), (1, ))
    assert_size_stride(arg1214_1, (), ())
    assert_size_stride(arg1215_1, (36, ), (1, ))
    assert_size_stride(arg1216_1, (36, ), (1, ))
    assert_size_stride(arg1217_1, (), ())
    assert_size_stride(arg1218_1, (36, ), (1, ))
    assert_size_stride(arg1219_1, (36, ), (1, ))
    assert_size_stride(arg1220_1, (), ())
    assert_size_stride(arg1221_1, (36, ), (1, ))
    assert_size_stride(arg1222_1, (36, ), (1, ))
    assert_size_stride(arg1223_1, (), ())
    assert_size_stride(arg1224_1, (36, ), (1, ))
    assert_size_stride(arg1225_1, (36, ), (1, ))
    assert_size_stride(arg1226_1, (), ())
    assert_size_stride(arg1227_1, (36, ), (1, ))
    assert_size_stride(arg1228_1, (36, ), (1, ))
    assert_size_stride(arg1229_1, (), ())
    assert_size_stride(arg1230_1, (72, ), (1, ))
    assert_size_stride(arg1231_1, (72, ), (1, ))
    assert_size_stride(arg1232_1, (), ())
    assert_size_stride(arg1233_1, (72, ), (1, ))
    assert_size_stride(arg1234_1, (72, ), (1, ))
    assert_size_stride(arg1235_1, (), ())
    assert_size_stride(arg1236_1, (72, ), (1, ))
    assert_size_stride(arg1237_1, (72, ), (1, ))
    assert_size_stride(arg1238_1, (), ())
    assert_size_stride(arg1239_1, (72, ), (1, ))
    assert_size_stride(arg1240_1, (72, ), (1, ))
    assert_size_stride(arg1241_1, (), ())
    assert_size_stride(arg1242_1, (72, ), (1, ))
    assert_size_stride(arg1243_1, (72, ), (1, ))
    assert_size_stride(arg1244_1, (), ())
    assert_size_stride(arg1245_1, (72, ), (1, ))
    assert_size_stride(arg1246_1, (72, ), (1, ))
    assert_size_stride(arg1247_1, (), ())
    assert_size_stride(arg1248_1, (72, ), (1, ))
    assert_size_stride(arg1249_1, (72, ), (1, ))
    assert_size_stride(arg1250_1, (), ())
    assert_size_stride(arg1251_1, (72, ), (1, ))
    assert_size_stride(arg1252_1, (72, ), (1, ))
    assert_size_stride(arg1253_1, (), ())
    assert_size_stride(arg1254_1, (18, ), (1, ))
    assert_size_stride(arg1255_1, (18, ), (1, ))
    assert_size_stride(arg1256_1, (), ())
    assert_size_stride(arg1257_1, (18, ), (1, ))
    assert_size_stride(arg1258_1, (18, ), (1, ))
    assert_size_stride(arg1259_1, (), ())
    assert_size_stride(arg1260_1, (36, ), (1, ))
    assert_size_stride(arg1261_1, (36, ), (1, ))
    assert_size_stride(arg1262_1, (), ())
    assert_size_stride(arg1263_1, (36, ), (1, ))
    assert_size_stride(arg1264_1, (36, ), (1, ))
    assert_size_stride(arg1265_1, (), ())
    assert_size_stride(arg1266_1, (18, ), (1, ))
    assert_size_stride(arg1267_1, (18, ), (1, ))
    assert_size_stride(arg1268_1, (), ())
    assert_size_stride(arg1269_1, (72, ), (1, ))
    assert_size_stride(arg1270_1, (72, ), (1, ))
    assert_size_stride(arg1271_1, (), ())
    assert_size_stride(arg1272_1, (72, ), (1, ))
    assert_size_stride(arg1273_1, (72, ), (1, ))
    assert_size_stride(arg1274_1, (), ())
    assert_size_stride(arg1275_1, (18, ), (1, ))
    assert_size_stride(arg1276_1, (18, ), (1, ))
    assert_size_stride(arg1277_1, (), ())
    assert_size_stride(arg1278_1, (18, ), (1, ))
    assert_size_stride(arg1279_1, (18, ), (1, ))
    assert_size_stride(arg1280_1, (), ())
    assert_size_stride(arg1281_1, (18, ), (1, ))
    assert_size_stride(arg1282_1, (18, ), (1, ))
    assert_size_stride(arg1283_1, (), ())
    assert_size_stride(arg1284_1, (18, ), (1, ))
    assert_size_stride(arg1285_1, (18, ), (1, ))
    assert_size_stride(arg1286_1, (), ())
    assert_size_stride(arg1287_1, (18, ), (1, ))
    assert_size_stride(arg1288_1, (18, ), (1, ))
    assert_size_stride(arg1289_1, (), ())
    assert_size_stride(arg1290_1, (18, ), (1, ))
    assert_size_stride(arg1291_1, (18, ), (1, ))
    assert_size_stride(arg1292_1, (), ())
    assert_size_stride(arg1293_1, (18, ), (1, ))
    assert_size_stride(arg1294_1, (18, ), (1, ))
    assert_size_stride(arg1295_1, (), ())
    assert_size_stride(arg1296_1, (18, ), (1, ))
    assert_size_stride(arg1297_1, (18, ), (1, ))
    assert_size_stride(arg1298_1, (), ())
    assert_size_stride(arg1299_1, (36, ), (1, ))
    assert_size_stride(arg1300_1, (36, ), (1, ))
    assert_size_stride(arg1301_1, (), ())
    assert_size_stride(arg1302_1, (36, ), (1, ))
    assert_size_stride(arg1303_1, (36, ), (1, ))
    assert_size_stride(arg1304_1, (), ())
    assert_size_stride(arg1305_1, (36, ), (1, ))
    assert_size_stride(arg1306_1, (36, ), (1, ))
    assert_size_stride(arg1307_1, (), ())
    assert_size_stride(arg1308_1, (36, ), (1, ))
    assert_size_stride(arg1309_1, (36, ), (1, ))
    assert_size_stride(arg1310_1, (), ())
    assert_size_stride(arg1311_1, (36, ), (1, ))
    assert_size_stride(arg1312_1, (36, ), (1, ))
    assert_size_stride(arg1313_1, (), ())
    assert_size_stride(arg1314_1, (36, ), (1, ))
    assert_size_stride(arg1315_1, (36, ), (1, ))
    assert_size_stride(arg1316_1, (), ())
    assert_size_stride(arg1317_1, (36, ), (1, ))
    assert_size_stride(arg1318_1, (36, ), (1, ))
    assert_size_stride(arg1319_1, (), ())
    assert_size_stride(arg1320_1, (36, ), (1, ))
    assert_size_stride(arg1321_1, (36, ), (1, ))
    assert_size_stride(arg1322_1, (), ())
    assert_size_stride(arg1323_1, (72, ), (1, ))
    assert_size_stride(arg1324_1, (72, ), (1, ))
    assert_size_stride(arg1325_1, (), ())
    assert_size_stride(arg1326_1, (72, ), (1, ))
    assert_size_stride(arg1327_1, (72, ), (1, ))
    assert_size_stride(arg1328_1, (), ())
    assert_size_stride(arg1329_1, (72, ), (1, ))
    assert_size_stride(arg1330_1, (72, ), (1, ))
    assert_size_stride(arg1331_1, (), ())
    assert_size_stride(arg1332_1, (72, ), (1, ))
    assert_size_stride(arg1333_1, (72, ), (1, ))
    assert_size_stride(arg1334_1, (), ())
    assert_size_stride(arg1335_1, (72, ), (1, ))
    assert_size_stride(arg1336_1, (72, ), (1, ))
    assert_size_stride(arg1337_1, (), ())
    assert_size_stride(arg1338_1, (72, ), (1, ))
    assert_size_stride(arg1339_1, (72, ), (1, ))
    assert_size_stride(arg1340_1, (), ())
    assert_size_stride(arg1341_1, (72, ), (1, ))
    assert_size_stride(arg1342_1, (72, ), (1, ))
    assert_size_stride(arg1343_1, (), ())
    assert_size_stride(arg1344_1, (72, ), (1, ))
    assert_size_stride(arg1345_1, (72, ), (1, ))
    assert_size_stride(arg1346_1, (), ())
    assert_size_stride(arg1347_1, (18, ), (1, ))
    assert_size_stride(arg1348_1, (18, ), (1, ))
    assert_size_stride(arg1349_1, (), ())
    assert_size_stride(arg1350_1, (18, ), (1, ))
    assert_size_stride(arg1351_1, (18, ), (1, ))
    assert_size_stride(arg1352_1, (), ())
    assert_size_stride(arg1353_1, (36, ), (1, ))
    assert_size_stride(arg1354_1, (36, ), (1, ))
    assert_size_stride(arg1355_1, (), ())
    assert_size_stride(arg1356_1, (36, ), (1, ))
    assert_size_stride(arg1357_1, (36, ), (1, ))
    assert_size_stride(arg1358_1, (), ())
    assert_size_stride(arg1359_1, (18, ), (1, ))
    assert_size_stride(arg1360_1, (18, ), (1, ))
    assert_size_stride(arg1361_1, (), ())
    assert_size_stride(arg1362_1, (72, ), (1, ))
    assert_size_stride(arg1363_1, (72, ), (1, ))
    assert_size_stride(arg1364_1, (), ())
    assert_size_stride(arg1365_1, (72, ), (1, ))
    assert_size_stride(arg1366_1, (72, ), (1, ))
    assert_size_stride(arg1367_1, (), ())
    assert_size_stride(arg1368_1, (18, ), (1, ))
    assert_size_stride(arg1369_1, (18, ), (1, ))
    assert_size_stride(arg1370_1, (), ())
    assert_size_stride(arg1371_1, (18, ), (1, ))
    assert_size_stride(arg1372_1, (18, ), (1, ))
    assert_size_stride(arg1373_1, (), ())
    assert_size_stride(arg1374_1, (18, ), (1, ))
    assert_size_stride(arg1375_1, (18, ), (1, ))
    assert_size_stride(arg1376_1, (), ())
    assert_size_stride(arg1377_1, (18, ), (1, ))
    assert_size_stride(arg1378_1, (18, ), (1, ))
    assert_size_stride(arg1379_1, (), ())
    assert_size_stride(arg1380_1, (18, ), (1, ))
    assert_size_stride(arg1381_1, (18, ), (1, ))
    assert_size_stride(arg1382_1, (), ())
    assert_size_stride(arg1383_1, (18, ), (1, ))
    assert_size_stride(arg1384_1, (18, ), (1, ))
    assert_size_stride(arg1385_1, (), ())
    assert_size_stride(arg1386_1, (18, ), (1, ))
    assert_size_stride(arg1387_1, (18, ), (1, ))
    assert_size_stride(arg1388_1, (), ())
    assert_size_stride(arg1389_1, (18, ), (1, ))
    assert_size_stride(arg1390_1, (18, ), (1, ))
    assert_size_stride(arg1391_1, (), ())
    assert_size_stride(arg1392_1, (36, ), (1, ))
    assert_size_stride(arg1393_1, (36, ), (1, ))
    assert_size_stride(arg1394_1, (), ())
    assert_size_stride(arg1395_1, (36, ), (1, ))
    assert_size_stride(arg1396_1, (36, ), (1, ))
    assert_size_stride(arg1397_1, (), ())
    assert_size_stride(arg1398_1, (36, ), (1, ))
    assert_size_stride(arg1399_1, (36, ), (1, ))
    assert_size_stride(arg1400_1, (), ())
    assert_size_stride(arg1401_1, (36, ), (1, ))
    assert_size_stride(arg1402_1, (36, ), (1, ))
    assert_size_stride(arg1403_1, (), ())
    assert_size_stride(arg1404_1, (36, ), (1, ))
    assert_size_stride(arg1405_1, (36, ), (1, ))
    assert_size_stride(arg1406_1, (), ())
    assert_size_stride(arg1407_1, (36, ), (1, ))
    assert_size_stride(arg1408_1, (36, ), (1, ))
    assert_size_stride(arg1409_1, (), ())
    assert_size_stride(arg1410_1, (36, ), (1, ))
    assert_size_stride(arg1411_1, (36, ), (1, ))
    assert_size_stride(arg1412_1, (), ())
    assert_size_stride(arg1413_1, (36, ), (1, ))
    assert_size_stride(arg1414_1, (36, ), (1, ))
    assert_size_stride(arg1415_1, (), ())
    assert_size_stride(arg1416_1, (72, ), (1, ))
    assert_size_stride(arg1417_1, (72, ), (1, ))
    assert_size_stride(arg1418_1, (), ())
    assert_size_stride(arg1419_1, (72, ), (1, ))
    assert_size_stride(arg1420_1, (72, ), (1, ))
    assert_size_stride(arg1421_1, (), ())
    assert_size_stride(arg1422_1, (72, ), (1, ))
    assert_size_stride(arg1423_1, (72, ), (1, ))
    assert_size_stride(arg1424_1, (), ())
    assert_size_stride(arg1425_1, (72, ), (1, ))
    assert_size_stride(arg1426_1, (72, ), (1, ))
    assert_size_stride(arg1427_1, (), ())
    assert_size_stride(arg1428_1, (72, ), (1, ))
    assert_size_stride(arg1429_1, (72, ), (1, ))
    assert_size_stride(arg1430_1, (), ())
    assert_size_stride(arg1431_1, (72, ), (1, ))
    assert_size_stride(arg1432_1, (72, ), (1, ))
    assert_size_stride(arg1433_1, (), ())
    assert_size_stride(arg1434_1, (72, ), (1, ))
    assert_size_stride(arg1435_1, (72, ), (1, ))
    assert_size_stride(arg1436_1, (), ())
    assert_size_stride(arg1437_1, (72, ), (1, ))
    assert_size_stride(arg1438_1, (72, ), (1, ))
    assert_size_stride(arg1439_1, (), ())
    assert_size_stride(arg1440_1, (18, ), (1, ))
    assert_size_stride(arg1441_1, (18, ), (1, ))
    assert_size_stride(arg1442_1, (), ())
    assert_size_stride(arg1443_1, (18, ), (1, ))
    assert_size_stride(arg1444_1, (18, ), (1, ))
    assert_size_stride(arg1445_1, (), ())
    assert_size_stride(arg1446_1, (36, ), (1, ))
    assert_size_stride(arg1447_1, (36, ), (1, ))
    assert_size_stride(arg1448_1, (), ())
    assert_size_stride(arg1449_1, (36, ), (1, ))
    assert_size_stride(arg1450_1, (36, ), (1, ))
    assert_size_stride(arg1451_1, (), ())
    assert_size_stride(arg1452_1, (18, ), (1, ))
    assert_size_stride(arg1453_1, (18, ), (1, ))
    assert_size_stride(arg1454_1, (), ())
    assert_size_stride(arg1455_1, (72, ), (1, ))
    assert_size_stride(arg1456_1, (72, ), (1, ))
    assert_size_stride(arg1457_1, (), ())
    assert_size_stride(arg1458_1, (72, ), (1, ))
    assert_size_stride(arg1459_1, (72, ), (1, ))
    assert_size_stride(arg1460_1, (), ())
    assert_size_stride(arg1461_1, (144, ), (1, ))
    assert_size_stride(arg1462_1, (144, ), (1, ))
    assert_size_stride(arg1463_1, (), ())
    assert_size_stride(arg1464_1, (18, ), (1, ))
    assert_size_stride(arg1465_1, (18, ), (1, ))
    assert_size_stride(arg1466_1, (), ())
    assert_size_stride(arg1467_1, (18, ), (1, ))
    assert_size_stride(arg1468_1, (18, ), (1, ))
    assert_size_stride(arg1469_1, (), ())
    assert_size_stride(arg1470_1, (18, ), (1, ))
    assert_size_stride(arg1471_1, (18, ), (1, ))
    assert_size_stride(arg1472_1, (), ())
    assert_size_stride(arg1473_1, (18, ), (1, ))
    assert_size_stride(arg1474_1, (18, ), (1, ))
    assert_size_stride(arg1475_1, (), ())
    assert_size_stride(arg1476_1, (18, ), (1, ))
    assert_size_stride(arg1477_1, (18, ), (1, ))
    assert_size_stride(arg1478_1, (), ())
    assert_size_stride(arg1479_1, (18, ), (1, ))
    assert_size_stride(arg1480_1, (18, ), (1, ))
    assert_size_stride(arg1481_1, (), ())
    assert_size_stride(arg1482_1, (18, ), (1, ))
    assert_size_stride(arg1483_1, (18, ), (1, ))
    assert_size_stride(arg1484_1, (), ())
    assert_size_stride(arg1485_1, (18, ), (1, ))
    assert_size_stride(arg1486_1, (18, ), (1, ))
    assert_size_stride(arg1487_1, (), ())
    assert_size_stride(arg1488_1, (36, ), (1, ))
    assert_size_stride(arg1489_1, (36, ), (1, ))
    assert_size_stride(arg1490_1, (), ())
    assert_size_stride(arg1491_1, (36, ), (1, ))
    assert_size_stride(arg1492_1, (36, ), (1, ))
    assert_size_stride(arg1493_1, (), ())
    assert_size_stride(arg1494_1, (36, ), (1, ))
    assert_size_stride(arg1495_1, (36, ), (1, ))
    assert_size_stride(arg1496_1, (), ())
    assert_size_stride(arg1497_1, (36, ), (1, ))
    assert_size_stride(arg1498_1, (36, ), (1, ))
    assert_size_stride(arg1499_1, (), ())
    assert_size_stride(arg1500_1, (36, ), (1, ))
    assert_size_stride(arg1501_1, (36, ), (1, ))
    assert_size_stride(arg1502_1, (), ())
    assert_size_stride(arg1503_1, (36, ), (1, ))
    assert_size_stride(arg1504_1, (36, ), (1, ))
    assert_size_stride(arg1505_1, (), ())
    assert_size_stride(arg1506_1, (36, ), (1, ))
    assert_size_stride(arg1507_1, (36, ), (1, ))
    assert_size_stride(arg1508_1, (), ())
    assert_size_stride(arg1509_1, (36, ), (1, ))
    assert_size_stride(arg1510_1, (36, ), (1, ))
    assert_size_stride(arg1511_1, (), ())
    assert_size_stride(arg1512_1, (72, ), (1, ))
    assert_size_stride(arg1513_1, (72, ), (1, ))
    assert_size_stride(arg1514_1, (), ())
    assert_size_stride(arg1515_1, (72, ), (1, ))
    assert_size_stride(arg1516_1, (72, ), (1, ))
    assert_size_stride(arg1517_1, (), ())
    assert_size_stride(arg1518_1, (72, ), (1, ))
    assert_size_stride(arg1519_1, (72, ), (1, ))
    assert_size_stride(arg1520_1, (), ())
    assert_size_stride(arg1521_1, (72, ), (1, ))
    assert_size_stride(arg1522_1, (72, ), (1, ))
    assert_size_stride(arg1523_1, (), ())
    assert_size_stride(arg1524_1, (72, ), (1, ))
    assert_size_stride(arg1525_1, (72, ), (1, ))
    assert_size_stride(arg1526_1, (), ())
    assert_size_stride(arg1527_1, (72, ), (1, ))
    assert_size_stride(arg1528_1, (72, ), (1, ))
    assert_size_stride(arg1529_1, (), ())
    assert_size_stride(arg1530_1, (72, ), (1, ))
    assert_size_stride(arg1531_1, (72, ), (1, ))
    assert_size_stride(arg1532_1, (), ())
    assert_size_stride(arg1533_1, (72, ), (1, ))
    assert_size_stride(arg1534_1, (72, ), (1, ))
    assert_size_stride(arg1535_1, (), ())
    assert_size_stride(arg1536_1, (144, ), (1, ))
    assert_size_stride(arg1537_1, (144, ), (1, ))
    assert_size_stride(arg1538_1, (), ())
    assert_size_stride(arg1539_1, (144, ), (1, ))
    assert_size_stride(arg1540_1, (144, ), (1, ))
    assert_size_stride(arg1541_1, (), ())
    assert_size_stride(arg1542_1, (144, ), (1, ))
    assert_size_stride(arg1543_1, (144, ), (1, ))
    assert_size_stride(arg1544_1, (), ())
    assert_size_stride(arg1545_1, (144, ), (1, ))
    assert_size_stride(arg1546_1, (144, ), (1, ))
    assert_size_stride(arg1547_1, (), ())
    assert_size_stride(arg1548_1, (144, ), (1, ))
    assert_size_stride(arg1549_1, (144, ), (1, ))
    assert_size_stride(arg1550_1, (), ())
    assert_size_stride(arg1551_1, (144, ), (1, ))
    assert_size_stride(arg1552_1, (144, ), (1, ))
    assert_size_stride(arg1553_1, (), ())
    assert_size_stride(arg1554_1, (144, ), (1, ))
    assert_size_stride(arg1555_1, (144, ), (1, ))
    assert_size_stride(arg1556_1, (), ())
    assert_size_stride(arg1557_1, (144, ), (1, ))
    assert_size_stride(arg1558_1, (144, ), (1, ))
    assert_size_stride(arg1559_1, (), ())
    assert_size_stride(arg1560_1, (18, ), (1, ))
    assert_size_stride(arg1561_1, (18, ), (1, ))
    assert_size_stride(arg1562_1, (), ())
    assert_size_stride(arg1563_1, (18, ), (1, ))
    assert_size_stride(arg1564_1, (18, ), (1, ))
    assert_size_stride(arg1565_1, (), ())
    assert_size_stride(arg1566_1, (18, ), (1, ))
    assert_size_stride(arg1567_1, (18, ), (1, ))
    assert_size_stride(arg1568_1, (), ())
    assert_size_stride(arg1569_1, (36, ), (1, ))
    assert_size_stride(arg1570_1, (36, ), (1, ))
    assert_size_stride(arg1571_1, (), ())
    assert_size_stride(arg1572_1, (36, ), (1, ))
    assert_size_stride(arg1573_1, (36, ), (1, ))
    assert_size_stride(arg1574_1, (), ())
    assert_size_stride(arg1575_1, (36, ), (1, ))
    assert_size_stride(arg1576_1, (36, ), (1, ))
    assert_size_stride(arg1577_1, (), ())
    assert_size_stride(arg1578_1, (18, ), (1, ))
    assert_size_stride(arg1579_1, (18, ), (1, ))
    assert_size_stride(arg1580_1, (), ())
    assert_size_stride(arg1581_1, (72, ), (1, ))
    assert_size_stride(arg1582_1, (72, ), (1, ))
    assert_size_stride(arg1583_1, (), ())
    assert_size_stride(arg1584_1, (72, ), (1, ))
    assert_size_stride(arg1585_1, (72, ), (1, ))
    assert_size_stride(arg1586_1, (), ())
    assert_size_stride(arg1587_1, (72, ), (1, ))
    assert_size_stride(arg1588_1, (72, ), (1, ))
    assert_size_stride(arg1589_1, (), ())
    assert_size_stride(arg1590_1, (18, ), (1, ))
    assert_size_stride(arg1591_1, (18, ), (1, ))
    assert_size_stride(arg1592_1, (), ())
    assert_size_stride(arg1593_1, (18, ), (1, ))
    assert_size_stride(arg1594_1, (18, ), (1, ))
    assert_size_stride(arg1595_1, (), ())
    assert_size_stride(arg1596_1, (144, ), (1, ))
    assert_size_stride(arg1597_1, (144, ), (1, ))
    assert_size_stride(arg1598_1, (), ())
    assert_size_stride(arg1599_1, (36, ), (1, ))
    assert_size_stride(arg1600_1, (36, ), (1, ))
    assert_size_stride(arg1601_1, (), ())
    assert_size_stride(arg1602_1, (144, ), (1, ))
    assert_size_stride(arg1603_1, (144, ), (1, ))
    assert_size_stride(arg1604_1, (), ())
    assert_size_stride(arg1605_1, (144, ), (1, ))
    assert_size_stride(arg1606_1, (144, ), (1, ))
    assert_size_stride(arg1607_1, (), ())
    assert_size_stride(arg1608_1, (18, ), (1, ))
    assert_size_stride(arg1609_1, (18, ), (1, ))
    assert_size_stride(arg1610_1, (), ())
    assert_size_stride(arg1611_1, (18, ), (1, ))
    assert_size_stride(arg1612_1, (18, ), (1, ))
    assert_size_stride(arg1613_1, (), ())
    assert_size_stride(arg1614_1, (18, ), (1, ))
    assert_size_stride(arg1615_1, (18, ), (1, ))
    assert_size_stride(arg1616_1, (), ())
    assert_size_stride(arg1617_1, (18, ), (1, ))
    assert_size_stride(arg1618_1, (18, ), (1, ))
    assert_size_stride(arg1619_1, (), ())
    assert_size_stride(arg1620_1, (18, ), (1, ))
    assert_size_stride(arg1621_1, (18, ), (1, ))
    assert_size_stride(arg1622_1, (), ())
    assert_size_stride(arg1623_1, (18, ), (1, ))
    assert_size_stride(arg1624_1, (18, ), (1, ))
    assert_size_stride(arg1625_1, (), ())
    assert_size_stride(arg1626_1, (18, ), (1, ))
    assert_size_stride(arg1627_1, (18, ), (1, ))
    assert_size_stride(arg1628_1, (), ())
    assert_size_stride(arg1629_1, (18, ), (1, ))
    assert_size_stride(arg1630_1, (18, ), (1, ))
    assert_size_stride(arg1631_1, (), ())
    assert_size_stride(arg1632_1, (36, ), (1, ))
    assert_size_stride(arg1633_1, (36, ), (1, ))
    assert_size_stride(arg1634_1, (), ())
    assert_size_stride(arg1635_1, (36, ), (1, ))
    assert_size_stride(arg1636_1, (36, ), (1, ))
    assert_size_stride(arg1637_1, (), ())
    assert_size_stride(arg1638_1, (36, ), (1, ))
    assert_size_stride(arg1639_1, (36, ), (1, ))
    assert_size_stride(arg1640_1, (), ())
    assert_size_stride(arg1641_1, (36, ), (1, ))
    assert_size_stride(arg1642_1, (36, ), (1, ))
    assert_size_stride(arg1643_1, (), ())
    assert_size_stride(arg1644_1, (36, ), (1, ))
    assert_size_stride(arg1645_1, (36, ), (1, ))
    assert_size_stride(arg1646_1, (), ())
    assert_size_stride(arg1647_1, (36, ), (1, ))
    assert_size_stride(arg1648_1, (36, ), (1, ))
    assert_size_stride(arg1649_1, (), ())
    assert_size_stride(arg1650_1, (36, ), (1, ))
    assert_size_stride(arg1651_1, (36, ), (1, ))
    assert_size_stride(arg1652_1, (), ())
    assert_size_stride(arg1653_1, (36, ), (1, ))
    assert_size_stride(arg1654_1, (36, ), (1, ))
    assert_size_stride(arg1655_1, (), ())
    assert_size_stride(arg1656_1, (72, ), (1, ))
    assert_size_stride(arg1657_1, (72, ), (1, ))
    assert_size_stride(arg1658_1, (), ())
    assert_size_stride(arg1659_1, (72, ), (1, ))
    assert_size_stride(arg1660_1, (72, ), (1, ))
    assert_size_stride(arg1661_1, (), ())
    assert_size_stride(arg1662_1, (72, ), (1, ))
    assert_size_stride(arg1663_1, (72, ), (1, ))
    assert_size_stride(arg1664_1, (), ())
    assert_size_stride(arg1665_1, (72, ), (1, ))
    assert_size_stride(arg1666_1, (72, ), (1, ))
    assert_size_stride(arg1667_1, (), ())
    assert_size_stride(arg1668_1, (72, ), (1, ))
    assert_size_stride(arg1669_1, (72, ), (1, ))
    assert_size_stride(arg1670_1, (), ())
    assert_size_stride(arg1671_1, (72, ), (1, ))
    assert_size_stride(arg1672_1, (72, ), (1, ))
    assert_size_stride(arg1673_1, (), ())
    assert_size_stride(arg1674_1, (72, ), (1, ))
    assert_size_stride(arg1675_1, (72, ), (1, ))
    assert_size_stride(arg1676_1, (), ())
    assert_size_stride(arg1677_1, (72, ), (1, ))
    assert_size_stride(arg1678_1, (72, ), (1, ))
    assert_size_stride(arg1679_1, (), ())
    assert_size_stride(arg1680_1, (144, ), (1, ))
    assert_size_stride(arg1681_1, (144, ), (1, ))
    assert_size_stride(arg1682_1, (), ())
    assert_size_stride(arg1683_1, (144, ), (1, ))
    assert_size_stride(arg1684_1, (144, ), (1, ))
    assert_size_stride(arg1685_1, (), ())
    assert_size_stride(arg1686_1, (144, ), (1, ))
    assert_size_stride(arg1687_1, (144, ), (1, ))
    assert_size_stride(arg1688_1, (), ())
    assert_size_stride(arg1689_1, (144, ), (1, ))
    assert_size_stride(arg1690_1, (144, ), (1, ))
    assert_size_stride(arg1691_1, (), ())
    assert_size_stride(arg1692_1, (144, ), (1, ))
    assert_size_stride(arg1693_1, (144, ), (1, ))
    assert_size_stride(arg1694_1, (), ())
    assert_size_stride(arg1695_1, (144, ), (1, ))
    assert_size_stride(arg1696_1, (144, ), (1, ))
    assert_size_stride(arg1697_1, (), ())
    assert_size_stride(arg1698_1, (144, ), (1, ))
    assert_size_stride(arg1699_1, (144, ), (1, ))
    assert_size_stride(arg1700_1, (), ())
    assert_size_stride(arg1701_1, (144, ), (1, ))
    assert_size_stride(arg1702_1, (144, ), (1, ))
    assert_size_stride(arg1703_1, (), ())
    assert_size_stride(arg1704_1, (18, ), (1, ))
    assert_size_stride(arg1705_1, (18, ), (1, ))
    assert_size_stride(arg1706_1, (), ())
    assert_size_stride(arg1707_1, (18, ), (1, ))
    assert_size_stride(arg1708_1, (18, ), (1, ))
    assert_size_stride(arg1709_1, (), ())
    assert_size_stride(arg1710_1, (18, ), (1, ))
    assert_size_stride(arg1711_1, (18, ), (1, ))
    assert_size_stride(arg1712_1, (), ())
    assert_size_stride(arg1713_1, (36, ), (1, ))
    assert_size_stride(arg1714_1, (36, ), (1, ))
    assert_size_stride(arg1715_1, (), ())
    assert_size_stride(arg1716_1, (36, ), (1, ))
    assert_size_stride(arg1717_1, (36, ), (1, ))
    assert_size_stride(arg1718_1, (), ())
    assert_size_stride(arg1719_1, (36, ), (1, ))
    assert_size_stride(arg1720_1, (36, ), (1, ))
    assert_size_stride(arg1721_1, (), ())
    assert_size_stride(arg1722_1, (18, ), (1, ))
    assert_size_stride(arg1723_1, (18, ), (1, ))
    assert_size_stride(arg1724_1, (), ())
    assert_size_stride(arg1725_1, (72, ), (1, ))
    assert_size_stride(arg1726_1, (72, ), (1, ))
    assert_size_stride(arg1727_1, (), ())
    assert_size_stride(arg1728_1, (72, ), (1, ))
    assert_size_stride(arg1729_1, (72, ), (1, ))
    assert_size_stride(arg1730_1, (), ())
    assert_size_stride(arg1731_1, (72, ), (1, ))
    assert_size_stride(arg1732_1, (72, ), (1, ))
    assert_size_stride(arg1733_1, (), ())
    assert_size_stride(arg1734_1, (18, ), (1, ))
    assert_size_stride(arg1735_1, (18, ), (1, ))
    assert_size_stride(arg1736_1, (), ())
    assert_size_stride(arg1737_1, (18, ), (1, ))
    assert_size_stride(arg1738_1, (18, ), (1, ))
    assert_size_stride(arg1739_1, (), ())
    assert_size_stride(arg1740_1, (144, ), (1, ))
    assert_size_stride(arg1741_1, (144, ), (1, ))
    assert_size_stride(arg1742_1, (), ())
    assert_size_stride(arg1743_1, (36, ), (1, ))
    assert_size_stride(arg1744_1, (36, ), (1, ))
    assert_size_stride(arg1745_1, (), ())
    assert_size_stride(arg1746_1, (144, ), (1, ))
    assert_size_stride(arg1747_1, (144, ), (1, ))
    assert_size_stride(arg1748_1, (), ())
    assert_size_stride(arg1749_1, (144, ), (1, ))
    assert_size_stride(arg1750_1, (144, ), (1, ))
    assert_size_stride(arg1751_1, (), ())
    assert_size_stride(arg1752_1, (18, ), (1, ))
    assert_size_stride(arg1753_1, (18, ), (1, ))
    assert_size_stride(arg1754_1, (), ())
    assert_size_stride(arg1755_1, (18, ), (1, ))
    assert_size_stride(arg1756_1, (18, ), (1, ))
    assert_size_stride(arg1757_1, (), ())
    assert_size_stride(arg1758_1, (18, ), (1, ))
    assert_size_stride(arg1759_1, (18, ), (1, ))
    assert_size_stride(arg1760_1, (), ())
    assert_size_stride(arg1761_1, (18, ), (1, ))
    assert_size_stride(arg1762_1, (18, ), (1, ))
    assert_size_stride(arg1763_1, (), ())
    assert_size_stride(arg1764_1, (18, ), (1, ))
    assert_size_stride(arg1765_1, (18, ), (1, ))
    assert_size_stride(arg1766_1, (), ())
    assert_size_stride(arg1767_1, (18, ), (1, ))
    assert_size_stride(arg1768_1, (18, ), (1, ))
    assert_size_stride(arg1769_1, (), ())
    assert_size_stride(arg1770_1, (18, ), (1, ))
    assert_size_stride(arg1771_1, (18, ), (1, ))
    assert_size_stride(arg1772_1, (), ())
    assert_size_stride(arg1773_1, (18, ), (1, ))
    assert_size_stride(arg1774_1, (18, ), (1, ))
    assert_size_stride(arg1775_1, (), ())
    assert_size_stride(arg1776_1, (36, ), (1, ))
    assert_size_stride(arg1777_1, (36, ), (1, ))
    assert_size_stride(arg1778_1, (), ())
    assert_size_stride(arg1779_1, (36, ), (1, ))
    assert_size_stride(arg1780_1, (36, ), (1, ))
    assert_size_stride(arg1781_1, (), ())
    assert_size_stride(arg1782_1, (36, ), (1, ))
    assert_size_stride(arg1783_1, (36, ), (1, ))
    assert_size_stride(arg1784_1, (), ())
    assert_size_stride(arg1785_1, (36, ), (1, ))
    assert_size_stride(arg1786_1, (36, ), (1, ))
    assert_size_stride(arg1787_1, (), ())
    assert_size_stride(arg1788_1, (36, ), (1, ))
    assert_size_stride(arg1789_1, (36, ), (1, ))
    assert_size_stride(arg1790_1, (), ())
    assert_size_stride(arg1791_1, (36, ), (1, ))
    assert_size_stride(arg1792_1, (36, ), (1, ))
    assert_size_stride(arg1793_1, (), ())
    assert_size_stride(arg1794_1, (36, ), (1, ))
    assert_size_stride(arg1795_1, (36, ), (1, ))
    assert_size_stride(arg1796_1, (), ())
    assert_size_stride(arg1797_1, (36, ), (1, ))
    assert_size_stride(arg1798_1, (36, ), (1, ))
    assert_size_stride(arg1799_1, (), ())
    assert_size_stride(arg1800_1, (72, ), (1, ))
    assert_size_stride(arg1801_1, (72, ), (1, ))
    assert_size_stride(arg1802_1, (), ())
    assert_size_stride(arg1803_1, (72, ), (1, ))
    assert_size_stride(arg1804_1, (72, ), (1, ))
    assert_size_stride(arg1805_1, (), ())
    assert_size_stride(arg1806_1, (72, ), (1, ))
    assert_size_stride(arg1807_1, (72, ), (1, ))
    assert_size_stride(arg1808_1, (), ())
    assert_size_stride(arg1809_1, (72, ), (1, ))
    assert_size_stride(arg1810_1, (72, ), (1, ))
    assert_size_stride(arg1811_1, (), ())
    assert_size_stride(arg1812_1, (72, ), (1, ))
    assert_size_stride(arg1813_1, (72, ), (1, ))
    assert_size_stride(arg1814_1, (), ())
    assert_size_stride(arg1815_1, (72, ), (1, ))
    assert_size_stride(arg1816_1, (72, ), (1, ))
    assert_size_stride(arg1817_1, (), ())
    assert_size_stride(arg1818_1, (72, ), (1, ))
    assert_size_stride(arg1819_1, (72, ), (1, ))
    assert_size_stride(arg1820_1, (), ())
    assert_size_stride(arg1821_1, (72, ), (1, ))
    assert_size_stride(arg1822_1, (72, ), (1, ))
    assert_size_stride(arg1823_1, (), ())
    assert_size_stride(arg1824_1, (144, ), (1, ))
    assert_size_stride(arg1825_1, (144, ), (1, ))
    assert_size_stride(arg1826_1, (), ())
    assert_size_stride(arg1827_1, (144, ), (1, ))
    assert_size_stride(arg1828_1, (144, ), (1, ))
    assert_size_stride(arg1829_1, (), ())
    assert_size_stride(arg1830_1, (144, ), (1, ))
    assert_size_stride(arg1831_1, (144, ), (1, ))
    assert_size_stride(arg1832_1, (), ())
    assert_size_stride(arg1833_1, (144, ), (1, ))
    assert_size_stride(arg1834_1, (144, ), (1, ))
    assert_size_stride(arg1835_1, (), ())
    assert_size_stride(arg1836_1, (144, ), (1, ))
    assert_size_stride(arg1837_1, (144, ), (1, ))
    assert_size_stride(arg1838_1, (), ())
    assert_size_stride(arg1839_1, (144, ), (1, ))
    assert_size_stride(arg1840_1, (144, ), (1, ))
    assert_size_stride(arg1841_1, (), ())
    assert_size_stride(arg1842_1, (144, ), (1, ))
    assert_size_stride(arg1843_1, (144, ), (1, ))
    assert_size_stride(arg1844_1, (), ())
    assert_size_stride(arg1845_1, (144, ), (1, ))
    assert_size_stride(arg1846_1, (144, ), (1, ))
    assert_size_stride(arg1847_1, (), ())
    assert_size_stride(arg1848_1, (18, ), (1, ))
    assert_size_stride(arg1849_1, (18, ), (1, ))
    assert_size_stride(arg1850_1, (), ())
    assert_size_stride(arg1851_1, (18, ), (1, ))
    assert_size_stride(arg1852_1, (18, ), (1, ))
    assert_size_stride(arg1853_1, (), ())
    assert_size_stride(arg1854_1, (18, ), (1, ))
    assert_size_stride(arg1855_1, (18, ), (1, ))
    assert_size_stride(arg1856_1, (), ())
    assert_size_stride(arg1857_1, (36, ), (1, ))
    assert_size_stride(arg1858_1, (36, ), (1, ))
    assert_size_stride(arg1859_1, (), ())
    assert_size_stride(arg1860_1, (36, ), (1, ))
    assert_size_stride(arg1861_1, (36, ), (1, ))
    assert_size_stride(arg1862_1, (), ())
    assert_size_stride(arg1863_1, (36, ), (1, ))
    assert_size_stride(arg1864_1, (36, ), (1, ))
    assert_size_stride(arg1865_1, (), ())
    assert_size_stride(arg1866_1, (18, ), (1, ))
    assert_size_stride(arg1867_1, (18, ), (1, ))
    assert_size_stride(arg1868_1, (), ())
    assert_size_stride(arg1869_1, (72, ), (1, ))
    assert_size_stride(arg1870_1, (72, ), (1, ))
    assert_size_stride(arg1871_1, (), ())
    assert_size_stride(arg1872_1, (72, ), (1, ))
    assert_size_stride(arg1873_1, (72, ), (1, ))
    assert_size_stride(arg1874_1, (), ())
    assert_size_stride(arg1875_1, (72, ), (1, ))
    assert_size_stride(arg1876_1, (72, ), (1, ))
    assert_size_stride(arg1877_1, (), ())
    assert_size_stride(arg1878_1, (18, ), (1, ))
    assert_size_stride(arg1879_1, (18, ), (1, ))
    assert_size_stride(arg1880_1, (), ())
    assert_size_stride(arg1881_1, (18, ), (1, ))
    assert_size_stride(arg1882_1, (18, ), (1, ))
    assert_size_stride(arg1883_1, (), ())
    assert_size_stride(arg1884_1, (144, ), (1, ))
    assert_size_stride(arg1885_1, (144, ), (1, ))
    assert_size_stride(arg1886_1, (), ())
    assert_size_stride(arg1887_1, (36, ), (1, ))
    assert_size_stride(arg1888_1, (36, ), (1, ))
    assert_size_stride(arg1889_1, (), ())
    assert_size_stride(arg1890_1, (144, ), (1, ))
    assert_size_stride(arg1891_1, (144, ), (1, ))
    assert_size_stride(arg1892_1, (), ())
    assert_size_stride(arg1893_1, (144, ), (1, ))
    assert_size_stride(arg1894_1, (144, ), (1, ))
    assert_size_stride(arg1895_1, (), ())
    assert_size_stride(arg1896_1, (32, ), (1, ))
    assert_size_stride(arg1897_1, (32, ), (1, ))
    assert_size_stride(arg1898_1, (), ())
    assert_size_stride(arg1899_1, (32, ), (1, ))
    assert_size_stride(arg1900_1, (32, ), (1, ))
    assert_size_stride(arg1901_1, (), ())
    assert_size_stride(arg1902_1, (128, ), (1, ))
    assert_size_stride(arg1903_1, (128, ), (1, ))
    assert_size_stride(arg1904_1, (), ())
    assert_size_stride(arg1905_1, (128, ), (1, ))
    assert_size_stride(arg1906_1, (128, ), (1, ))
    assert_size_stride(arg1907_1, (), ())
    assert_size_stride(arg1908_1, (64, ), (1, ))
    assert_size_stride(arg1909_1, (64, ), (1, ))
    assert_size_stride(arg1910_1, (), ())
    assert_size_stride(arg1911_1, (64, ), (1, ))
    assert_size_stride(arg1912_1, (64, ), (1, ))
    assert_size_stride(arg1913_1, (), ())
    assert_size_stride(arg1914_1, (256, ), (1, ))
    assert_size_stride(arg1915_1, (256, ), (1, ))
    assert_size_stride(arg1916_1, (), ())
    assert_size_stride(arg1917_1, (256, ), (1, ))
    assert_size_stride(arg1918_1, (256, ), (1, ))
    assert_size_stride(arg1919_1, (), ())
    assert_size_stride(arg1920_1, (256, ), (1, ))
    assert_size_stride(arg1921_1, (256, ), (1, ))
    assert_size_stride(arg1922_1, (), ())
    assert_size_stride(arg1923_1, (128, ), (1, ))
    assert_size_stride(arg1924_1, (128, ), (1, ))
    assert_size_stride(arg1925_1, (), ())
    assert_size_stride(arg1926_1, (128, ), (1, ))
    assert_size_stride(arg1927_1, (128, ), (1, ))
    assert_size_stride(arg1928_1, (), ())
    assert_size_stride(arg1929_1, (512, ), (1, ))
    assert_size_stride(arg1930_1, (512, ), (1, ))
    assert_size_stride(arg1931_1, (), ())
    assert_size_stride(arg1932_1, (512, ), (1, ))
    assert_size_stride(arg1933_1, (512, ), (1, ))
    assert_size_stride(arg1934_1, (), ())
    assert_size_stride(arg1935_1, (512, ), (1, ))
    assert_size_stride(arg1936_1, (512, ), (1, ))
    assert_size_stride(arg1937_1, (), ())
    assert_size_stride(arg1938_1, (256, ), (1, ))
    assert_size_stride(arg1939_1, (256, ), (1, ))
    assert_size_stride(arg1940_1, (), ())
    assert_size_stride(arg1941_1, (256, ), (1, ))
    assert_size_stride(arg1942_1, (256, ), (1, ))
    assert_size_stride(arg1943_1, (), ())
    assert_size_stride(arg1944_1, (1024, ), (1, ))
    assert_size_stride(arg1945_1, (1024, ), (1, ))
    assert_size_stride(arg1946_1, (), ())
    assert_size_stride(arg1947_1, (1024, ), (1, ))
    assert_size_stride(arg1948_1, (1024, ), (1, ))
    assert_size_stride(arg1949_1, (), ())
    assert_size_stride(arg1950_1, (1024, ), (1, ))
    assert_size_stride(arg1951_1, (1024, ), (1, ))
    assert_size_stride(arg1952_1, (), ())
    assert_size_stride(arg1953_1, (2048, ), (1, ))
    assert_size_stride(arg1954_1, (2048, ), (1, ))
    assert_size_stride(arg1955_1, (), ())
    assert_size_stride(arg1956_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg1956_1, arg0_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg0_1
        del arg1956_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg981_1, arg982_1, arg1_1, arg2_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg1_1
        del arg2_1
        del arg981_1
        del arg982_1
        # Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf2 = extern_kernels.convolution(buf1, arg3_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg3_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [shortcut, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf3, arg984_1, arg985_1, arg4_1, arg5_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg4_1
        del arg5_1
        del arg984_1
        del arg985_1
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg6_1
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [x_7, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf5, arg987_1, arg988_1, arg7_1, arg8_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg7_1
        del arg8_1
        del arg987_1
        del arg988_1
        # Source Nodes: [x_7, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg9_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg9_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [x_10, x_12, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf7, arg990_1, arg991_1, arg10_1, arg11_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg10_1
        del arg11_1
        del arg990_1
        del arg991_1
        # Source Nodes: [x_10, x_12, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf7, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg12_1
        del buf7
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf3, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg15_1
        del buf3
        buf10 = buf8; del buf8  # reuse
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [shortcut_1, shortcut_2, x_15, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2.run(buf11, arg993_1, arg994_1, arg13_1, arg14_1, buf9, arg996_1, arg997_1, arg16_1, arg17_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg13_1
        del arg14_1
        del arg16_1
        del arg17_1
        del arg993_1
        del arg994_1
        del arg996_1
        del arg997_1
        del buf9
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg18_1
        buf13 = buf12; del buf12  # reuse
        # Source Nodes: [x_19, x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf13, arg999_1, arg1000_1, arg19_1, arg20_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg1000_1
        del arg19_1
        del arg20_1
        del arg999_1
        # Source Nodes: [x_19, x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf14 = extern_kernels.convolution(buf13, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg21_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Source Nodes: [x_22, x_24, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf15, arg1002_1, arg1003_1, arg22_1, arg23_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg1002_1
        del arg1003_1
        del arg22_1
        del arg23_1
        # Source Nodes: [x_22, x_24, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf16 = extern_kernels.convolution(buf15, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg24_1
        del buf15
        buf17 = buf11; del buf11  # reuse
        # Source Nodes: [shortcut_3, x_27, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf17, buf16, arg1005_1, arg1006_1, arg25_1, arg26_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg1005_1
        del arg1006_1
        del arg25_1
        del arg26_1
        del buf16
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg27_1
        buf19 = buf18; del buf18  # reuse
        # Source Nodes: [x_31, x_32, x_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf19, arg1008_1, arg1009_1, arg28_1, arg29_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg1008_1
        del arg1009_1
        del arg28_1
        del arg29_1
        # Source Nodes: [x_31, x_32, x_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf20 = extern_kernels.convolution(buf19, arg30_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg30_1
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Source Nodes: [x_34, x_36, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf21, arg1011_1, arg1012_1, arg31_1, arg32_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg1011_1
        del arg1012_1
        del arg31_1
        del arg32_1
        # Source Nodes: [x_34, x_36, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf22 = extern_kernels.convolution(buf21, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg33_1
        del buf21
        buf23 = buf17; del buf17  # reuse
        # Source Nodes: [shortcut_4, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf23, buf22, arg1014_1, arg1015_1, arg34_1, arg35_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg1014_1
        del arg1015_1
        del arg34_1
        del arg35_1
        del buf22
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg36_1
        buf25 = buf24; del buf24  # reuse
        # Source Nodes: [x_43, x_44, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf25, arg1017_1, arg1018_1, arg37_1, arg38_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg1017_1
        del arg1018_1
        del arg37_1
        del arg38_1
        # Source Nodes: [x_43, x_44, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf26 = extern_kernels.convolution(buf25, arg39_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg39_1
        del buf25
        buf27 = buf26; del buf26  # reuse
        # Source Nodes: [x_46, x_48, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf27, arg1020_1, arg1021_1, arg40_1, arg41_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg1020_1
        del arg1021_1
        del arg40_1
        del arg41_1
        # Source Nodes: [x_46, x_48, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf28 = extern_kernels.convolution(buf27, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg42_1
        del buf27
        buf29 = buf23; del buf23  # reuse
        # Source Nodes: [x_51, x_52, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf29, buf28, arg1023_1, arg1024_1, arg43_1, arg44_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg1023_1
        del arg1024_1
        del arg43_1
        del arg44_1
        del buf28
        # Source Nodes: [l__mod___transition1_0_0], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg45_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg45_1
        buf31 = buf30; del buf30  # reuse
        # Source Nodes: [l__mod___transition1_0_1, shortcut_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf31, arg1026_1, arg1027_1, arg46_1, arg47_1, 451584, grid=grid(451584), stream=stream0)
        del arg1026_1
        del arg1027_1
        del arg46_1
        del arg47_1
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg51_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg51_1
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [x_55, x_57, x_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf33, arg1032_1, arg1033_1, arg52_1, arg53_1, 451584, grid=grid(451584), stream=stream0)
        del arg1032_1
        del arg1033_1
        del arg52_1
        del arg53_1
        # Source Nodes: [x_55, x_57, x_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf34 = extern_kernels.convolution(buf33, arg54_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg54_1
        del buf33
        buf35 = buf31; del buf31  # reuse
        # Source Nodes: [shortcut_6, x_60, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf35, buf34, arg1035_1, arg1036_1, arg55_1, arg56_1, 451584, grid=grid(451584), stream=stream0)
        del arg1035_1
        del arg1036_1
        del arg55_1
        del arg56_1
        del buf34
        # Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg57_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg57_1
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [x_64, x_66, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf37, arg1038_1, arg1039_1, arg58_1, arg59_1, 451584, grid=grid(451584), stream=stream0)
        del arg1038_1
        del arg1039_1
        del arg58_1
        del arg59_1
        # Source Nodes: [x_64, x_66, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf38 = extern_kernels.convolution(buf37, arg60_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg60_1
        del buf37
        buf39 = buf35; del buf35  # reuse
        # Source Nodes: [shortcut_7, x_69, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf39, buf38, arg1041_1, arg1042_1, arg61_1, arg62_1, 451584, grid=grid(451584), stream=stream0)
        del arg1041_1
        del arg1042_1
        del arg61_1
        del arg62_1
        del buf38
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg63_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg63_1
        buf41 = buf40; del buf40  # reuse
        # Source Nodes: [x_73, x_75, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf41, arg1044_1, arg1045_1, arg64_1, arg65_1, 451584, grid=grid(451584), stream=stream0)
        del arg1044_1
        del arg1045_1
        del arg64_1
        del arg65_1
        # Source Nodes: [x_73, x_75, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg66_1
        del buf41
        buf43 = buf39; del buf39  # reuse
        # Source Nodes: [shortcut_8, x_78, x_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf43, buf42, arg1047_1, arg1048_1, arg67_1, arg68_1, 451584, grid=grid(451584), stream=stream0)
        del arg1047_1
        del arg1048_1
        del arg67_1
        del arg68_1
        del buf42
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg69_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg69_1
        buf45 = buf44; del buf44  # reuse
        # Source Nodes: [x_82, x_84, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf45, arg1050_1, arg1051_1, arg70_1, arg71_1, 451584, grid=grid(451584), stream=stream0)
        del arg1050_1
        del arg1051_1
        del arg70_1
        del arg71_1
        # Source Nodes: [x_82, x_84, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg72_1
        # Source Nodes: [l__mod___transition1_1_0_0], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf29, arg48_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg48_1
        del buf29
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [l__mod___transition1_1_0_1, shortcut_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf49, arg1029_1, arg1030_1, arg49_1, arg50_1, 225792, grid=grid(225792), stream=stream0)
        del arg1029_1
        del arg1030_1
        del arg49_1
        del arg50_1
        # Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg75_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg75_1
        buf51 = buf50; del buf50  # reuse
        # Source Nodes: [x_91, x_93, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf51, arg1056_1, arg1057_1, arg76_1, arg77_1, 225792, grid=grid(225792), stream=stream0)
        del arg1056_1
        del arg1057_1
        del arg76_1
        del arg77_1
        # Source Nodes: [x_91, x_93, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf52 = extern_kernels.convolution(buf51, arg78_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg78_1
        del buf51
        buf53 = buf49; del buf49  # reuse
        # Source Nodes: [shortcut_10, x_96, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf53, buf52, arg1059_1, arg1060_1, arg79_1, arg80_1, 225792, grid=grid(225792), stream=stream0)
        del arg1059_1
        del arg1060_1
        del arg79_1
        del arg80_1
        del buf52
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg81_1
        buf55 = buf54; del buf54  # reuse
        # Source Nodes: [x_100, x_102, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf55, arg1062_1, arg1063_1, arg82_1, arg83_1, 225792, grid=grid(225792), stream=stream0)
        del arg1062_1
        del arg1063_1
        del arg82_1
        del arg83_1
        # Source Nodes: [x_100, x_102, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg84_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg84_1
        del buf55
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [shortcut_11, x_105, x_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf57, buf56, arg1065_1, arg1066_1, arg85_1, arg86_1, 225792, grid=grid(225792), stream=stream0)
        del arg1065_1
        del arg1066_1
        del arg85_1
        del arg86_1
        del buf56
        # Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg87_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg87_1
        buf59 = buf58; del buf58  # reuse
        # Source Nodes: [x_109, x_111, x_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf59, arg1068_1, arg1069_1, arg88_1, arg89_1, 225792, grid=grid(225792), stream=stream0)
        del arg1068_1
        del arg1069_1
        del arg88_1
        del arg89_1
        # Source Nodes: [x_109, x_111, x_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg90_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg90_1
        del buf59
        buf61 = buf57; del buf57  # reuse
        # Source Nodes: [shortcut_12, x_114, x_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf61, buf60, arg1071_1, arg1072_1, arg91_1, arg92_1, 225792, grid=grid(225792), stream=stream0)
        del arg1071_1
        del arg1072_1
        del arg91_1
        del arg92_1
        del buf60
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg93_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg93_1
        buf63 = buf62; del buf62  # reuse
        # Source Nodes: [x_118, x_120, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf63, arg1074_1, arg1075_1, arg94_1, arg95_1, 225792, grid=grid(225792), stream=stream0)
        del arg1074_1
        del arg1075_1
        del arg94_1
        del arg95_1
        # Source Nodes: [x_118, x_120, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf64 = extern_kernels.convolution(buf63, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg96_1
        del buf63
        buf65 = buf61; del buf61  # reuse
        # Source Nodes: [x_123, x_124, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf65, buf64, arg1077_1, arg1078_1, arg97_1, arg98_1, 225792, grid=grid(225792), stream=stream0)
        del arg1077_1
        del arg1078_1
        del arg97_1
        del arg98_1
        del buf64
        # Source Nodes: [l__mod___stage2_0_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg99_1
        buf47 = buf43; del buf43  # reuse
        buf67 = buf45; del buf45  # reuse
        # Source Nodes: [l__mod___stage2_0_fuse_layers_0_1_1, l__mod___stage2_0_fuse_layers_0_1_2, shortcut_13, x_87, x_88, x_89, y_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_8.run(buf47, buf46, arg1053_1, arg1054_1, arg73_1, arg74_1, buf66, arg1080_1, arg1081_1, arg100_1, arg101_1, buf67, 451584, grid=grid(451584), stream=stream0)
        del arg100_1
        del arg101_1
        del arg1053_1
        del arg1054_1
        del arg1080_1
        del arg1081_1
        del arg73_1
        del arg74_1
        del buf46
        del buf66
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg108_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg108_1
        buf69 = buf68; del buf68  # reuse
        # Source Nodes: [x_127, x_129, x_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf69, arg1089_1, arg1090_1, arg109_1, arg110_1, 451584, grid=grid(451584), stream=stream0)
        del arg1089_1
        del arg1090_1
        del arg109_1
        del arg110_1
        # Source Nodes: [x_127, x_129, x_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf70 = extern_kernels.convolution(buf69, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg111_1
        del buf69
        buf71 = buf67; del buf67  # reuse
        # Source Nodes: [shortcut_14, x_132, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf71, buf70, arg1092_1, arg1093_1, arg112_1, arg113_1, 451584, grid=grid(451584), stream=stream0)
        del arg1092_1
        del arg1093_1
        del arg112_1
        del arg113_1
        del buf70
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg114_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg114_1
        buf73 = buf72; del buf72  # reuse
        # Source Nodes: [x_136, x_138, x_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf73, arg1095_1, arg1096_1, arg115_1, arg116_1, 451584, grid=grid(451584), stream=stream0)
        del arg1095_1
        del arg1096_1
        del arg115_1
        del arg116_1
        # Source Nodes: [x_136, x_138, x_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf74 = extern_kernels.convolution(buf73, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg117_1
        del buf73
        buf75 = buf71; del buf71  # reuse
        # Source Nodes: [shortcut_15, x_141, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf75, buf74, arg1098_1, arg1099_1, arg118_1, arg119_1, 451584, grid=grid(451584), stream=stream0)
        del arg1098_1
        del arg1099_1
        del arg118_1
        del arg119_1
        del buf74
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg120_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg120_1
        buf77 = buf76; del buf76  # reuse
        # Source Nodes: [x_145, x_147, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf77, arg1101_1, arg1102_1, arg121_1, arg122_1, 451584, grid=grid(451584), stream=stream0)
        del arg1101_1
        del arg1102_1
        del arg121_1
        del arg122_1
        # Source Nodes: [x_145, x_147, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf78 = extern_kernels.convolution(buf77, arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg123_1
        del buf77
        buf79 = buf75; del buf75  # reuse
        # Source Nodes: [shortcut_16, x_150, x_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf79, buf78, arg1104_1, arg1105_1, arg124_1, arg125_1, 451584, grid=grid(451584), stream=stream0)
        del arg1104_1
        del arg1105_1
        del arg124_1
        del arg125_1
        del buf78
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg126_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg126_1
        buf81 = buf80; del buf80  # reuse
        # Source Nodes: [x_154, x_156, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf81, arg1107_1, arg1108_1, arg127_1, arg128_1, 451584, grid=grid(451584), stream=stream0)
        del arg1107_1
        del arg1108_1
        del arg127_1
        del arg128_1
        # Source Nodes: [x_154, x_156, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf82 = extern_kernels.convolution(buf81, arg129_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg129_1
        del buf81
        # Source Nodes: [l__mod___stage2_0_fuse_layers_1_0_0_0], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf47, arg102_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg102_1
        buf85 = buf65; del buf65  # reuse
        # Source Nodes: [shortcut_17, y_2, y_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf85, buf84, arg1083_1, arg1084_1, arg103_1, arg104_1, 225792, grid=grid(225792), stream=stream0)
        del arg103_1
        del arg104_1
        del arg1083_1
        del arg1084_1
        del buf84
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg132_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg132_1
        buf87 = buf86; del buf86  # reuse
        # Source Nodes: [x_163, x_165, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf87, arg1113_1, arg1114_1, arg133_1, arg134_1, 225792, grid=grid(225792), stream=stream0)
        del arg1113_1
        del arg1114_1
        del arg133_1
        del arg134_1
        # Source Nodes: [x_163, x_165, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf88 = extern_kernels.convolution(buf87, arg135_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg135_1
        del buf87
        buf89 = buf88; del buf88  # reuse
        # Source Nodes: [shortcut_18, x_168, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf89, arg1116_1, arg1117_1, arg136_1, arg137_1, buf85, 225792, grid=grid(225792), stream=stream0)
        del arg1116_1
        del arg1117_1
        del arg136_1
        del arg137_1
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg138_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg138_1
        buf91 = buf90; del buf90  # reuse
        # Source Nodes: [x_172, x_174, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf91, arg1119_1, arg1120_1, arg139_1, arg140_1, 225792, grid=grid(225792), stream=stream0)
        del arg1119_1
        del arg1120_1
        del arg139_1
        del arg140_1
        # Source Nodes: [x_172, x_174, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf92 = extern_kernels.convolution(buf91, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg141_1
        del buf91
        buf93 = buf89; del buf89  # reuse
        # Source Nodes: [shortcut_19, x_177, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf93, buf92, arg1122_1, arg1123_1, arg142_1, arg143_1, 225792, grid=grid(225792), stream=stream0)
        del arg1122_1
        del arg1123_1
        del arg142_1
        del arg143_1
        del buf92
        # Source Nodes: [x_180], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg144_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg144_1
        buf95 = buf94; del buf94  # reuse
        # Source Nodes: [x_181, x_183, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf95, arg1125_1, arg1126_1, arg145_1, arg146_1, 225792, grid=grid(225792), stream=stream0)
        del arg1125_1
        del arg1126_1
        del arg145_1
        del arg146_1
        # Source Nodes: [x_181, x_183, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf96 = extern_kernels.convolution(buf95, arg147_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg147_1
        del buf95
        buf97 = buf93; del buf93  # reuse
        # Source Nodes: [shortcut_20, x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf97, buf96, arg1128_1, arg1129_1, arg148_1, arg149_1, 225792, grid=grid(225792), stream=stream0)
        del arg1128_1
        del arg1129_1
        del arg148_1
        del arg149_1
        del buf96
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg150_1
        buf99 = buf98; del buf98  # reuse
        # Source Nodes: [x_190, x_192, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf99, arg1131_1, arg1132_1, arg151_1, arg152_1, 225792, grid=grid(225792), stream=stream0)
        del arg1131_1
        del arg1132_1
        del arg151_1
        del arg152_1
        # Source Nodes: [x_190, x_192, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf100 = extern_kernels.convolution(buf99, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg153_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Source Nodes: [x_195, x_196, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf101, arg1134_1, arg1135_1, arg154_1, arg155_1, buf97, 225792, grid=grid(225792), stream=stream0)
        del arg1134_1
        del arg1135_1
        del arg154_1
        del arg155_1
        del buf97
        # Source Nodes: [l__mod___stage3_0_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg180_1
        # Source Nodes: [l__mod___transition2_2_0_0], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf85, arg105_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg105_1
        del buf85
        buf104 = buf103; del buf103  # reuse
        # Source Nodes: [l__mod___transition2_2_0_1, shortcut_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf104, arg1086_1, arg1087_1, arg106_1, arg107_1, 112896, grid=grid(112896), stream=stream0)
        del arg106_1
        del arg107_1
        del arg1086_1
        del arg1087_1
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg156_1
        buf106 = buf105; del buf105  # reuse
        # Source Nodes: [x_199, x_201, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf106, arg1137_1, arg1138_1, arg157_1, arg158_1, 112896, grid=grid(112896), stream=stream0)
        del arg1137_1
        del arg1138_1
        del arg157_1
        del arg158_1
        # Source Nodes: [x_199, x_201, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf107 = extern_kernels.convolution(buf106, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg159_1
        del buf106
        buf108 = buf104; del buf104  # reuse
        # Source Nodes: [shortcut_22, x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf108, buf107, arg1140_1, arg1141_1, arg160_1, arg161_1, 112896, grid=grid(112896), stream=stream0)
        del arg1140_1
        del arg1141_1
        del arg160_1
        del arg161_1
        del buf107
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg162_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg162_1
        buf110 = buf109; del buf109  # reuse
        # Source Nodes: [x_208, x_210, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf110, arg1143_1, arg1144_1, arg163_1, arg164_1, 112896, grid=grid(112896), stream=stream0)
        del arg1143_1
        del arg1144_1
        del arg163_1
        del arg164_1
        # Source Nodes: [x_208, x_210, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf111 = extern_kernels.convolution(buf110, arg165_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg165_1
        del buf110
        buf112 = buf108; del buf108  # reuse
        # Source Nodes: [shortcut_23, x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf112, buf111, arg1146_1, arg1147_1, arg166_1, arg167_1, 112896, grid=grid(112896), stream=stream0)
        del arg1146_1
        del arg1147_1
        del arg166_1
        del arg167_1
        del buf111
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, arg168_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg168_1
        buf114 = buf113; del buf113  # reuse
        # Source Nodes: [x_217, x_219, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf114, arg1149_1, arg1150_1, arg169_1, arg170_1, 112896, grid=grid(112896), stream=stream0)
        del arg1149_1
        del arg1150_1
        del arg169_1
        del arg170_1
        # Source Nodes: [x_217, x_219, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf115 = extern_kernels.convolution(buf114, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg171_1
        del buf114
        buf116 = buf112; del buf112  # reuse
        # Source Nodes: [shortcut_24, x_222, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf116, buf115, arg1152_1, arg1153_1, arg172_1, arg173_1, 112896, grid=grid(112896), stream=stream0)
        del arg1152_1
        del arg1153_1
        del arg172_1
        del arg173_1
        del buf115
        # Source Nodes: [x_225], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg174_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg174_1
        buf118 = buf117; del buf117  # reuse
        # Source Nodes: [x_226, x_228, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf118, arg1155_1, arg1156_1, arg175_1, arg176_1, 112896, grid=grid(112896), stream=stream0)
        del arg1155_1
        del arg1156_1
        del arg175_1
        del arg176_1
        # Source Nodes: [x_226, x_228, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf119 = extern_kernels.convolution(buf118, arg177_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg177_1
        del buf118
        buf120 = buf116; del buf116  # reuse
        # Source Nodes: [x_231, x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf120, buf119, arg1158_1, arg1159_1, arg178_1, arg179_1, 112896, grid=grid(112896), stream=stream0)
        del arg1158_1
        del arg1159_1
        del arg178_1
        del arg179_1
        del buf119
        # Source Nodes: [l__mod___stage3_0_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg183_1
        buf83 = buf79; del buf79  # reuse
        buf122 = buf47; del buf47  # reuse
        buf123 = buf122; del buf122  # reuse
        # Source Nodes: [l__mod___stage3_0_fuse_layers_0_1_1, l__mod___stage3_0_fuse_layers_0_1_2, l__mod___stage3_0_fuse_layers_0_2_1, l__mod___stage3_0_fuse_layers_0_2_2, shortcut_25, x_159, x_160, x_161, y_5, y_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_12.run(buf83, buf123, buf82, arg1110_1, arg1111_1, arg130_1, arg131_1, buf102, arg1161_1, arg1162_1, arg181_1, arg182_1, buf121, arg1164_1, arg1165_1, arg184_1, arg185_1, 451584, grid=grid(451584), stream=stream0)
        del arg1110_1
        del arg1111_1
        del arg1161_1
        del arg1162_1
        del arg1164_1
        del arg1165_1
        del arg130_1
        del arg131_1
        del arg181_1
        del arg182_1
        del arg184_1
        del arg185_1
        del buf102
        del buf121
        del buf82
        # Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg201_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg201_1
        buf125 = buf124; del buf124  # reuse
        # Source Nodes: [x_235, x_237, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf125, arg1182_1, arg1183_1, arg202_1, arg203_1, 451584, grid=grid(451584), stream=stream0)
        del arg1182_1
        del arg1183_1
        del arg202_1
        del arg203_1
        # Source Nodes: [x_235, x_237, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg204_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg204_1
        del buf125
        buf127 = buf123; del buf123  # reuse
        # Source Nodes: [shortcut_26, x_240, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf127, buf126, arg1185_1, arg1186_1, arg205_1, arg206_1, 451584, grid=grid(451584), stream=stream0)
        del arg1185_1
        del arg1186_1
        del arg205_1
        del arg206_1
        del buf126
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg207_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg207_1
        buf129 = buf128; del buf128  # reuse
        # Source Nodes: [x_244, x_246, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf129, arg1188_1, arg1189_1, arg208_1, arg209_1, 451584, grid=grid(451584), stream=stream0)
        del arg1188_1
        del arg1189_1
        del arg208_1
        del arg209_1
        # Source Nodes: [x_244, x_246, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf130 = extern_kernels.convolution(buf129, arg210_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg210_1
        del buf129
        buf131 = buf127; del buf127  # reuse
        # Source Nodes: [shortcut_27, x_249, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf131, buf130, arg1191_1, arg1192_1, arg211_1, arg212_1, 451584, grid=grid(451584), stream=stream0)
        del arg1191_1
        del arg1192_1
        del arg211_1
        del arg212_1
        del buf130
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg213_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg213_1
        buf133 = buf132; del buf132  # reuse
        # Source Nodes: [x_253, x_255, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf133, arg1194_1, arg1195_1, arg214_1, arg215_1, 451584, grid=grid(451584), stream=stream0)
        del arg1194_1
        del arg1195_1
        del arg214_1
        del arg215_1
        # Source Nodes: [x_253, x_255, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf134 = extern_kernels.convolution(buf133, arg216_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg216_1
        del buf133
        buf135 = buf131; del buf131  # reuse
        # Source Nodes: [shortcut_28, x_258, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf135, buf134, arg1197_1, arg1198_1, arg217_1, arg218_1, 451584, grid=grid(451584), stream=stream0)
        del arg1197_1
        del arg1198_1
        del arg217_1
        del arg218_1
        del buf134
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg219_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg219_1
        buf137 = buf136; del buf136  # reuse
        # Source Nodes: [x_262, x_264, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf137, arg1200_1, arg1201_1, arg220_1, arg221_1, 451584, grid=grid(451584), stream=stream0)
        del arg1200_1
        del arg1201_1
        del arg220_1
        del arg221_1
        # Source Nodes: [x_262, x_264, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf138 = extern_kernels.convolution(buf137, arg222_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg222_1
        del buf137
        # Source Nodes: [l__mod___stage3_0_fuse_layers_1_0_0_0], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf83, arg186_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg186_1
        # Source Nodes: [l__mod___stage3_0_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf120, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg189_1
        buf142 = buf140; del buf140  # reuse
        buf143 = buf142; del buf142  # reuse
        # Source Nodes: [l__mod___stage3_0_fuse_layers_1_2_1, l__mod___stage3_0_fuse_layers_1_2_2, shortcut_29, y_7, y_8, y_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_13.run(buf143, arg1167_1, arg1168_1, arg187_1, arg188_1, buf101, buf141, arg1170_1, arg1171_1, arg190_1, arg191_1, 225792, grid=grid(225792), stream=stream0)
        del arg1167_1
        del arg1168_1
        del arg1170_1
        del arg1171_1
        del arg187_1
        del arg188_1
        del arg190_1
        del arg191_1
        del buf141
        # Source Nodes: [x_270], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, arg225_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg225_1
        buf145 = buf144; del buf144  # reuse
        # Source Nodes: [x_271, x_273, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf145, arg1206_1, arg1207_1, arg226_1, arg227_1, 225792, grid=grid(225792), stream=stream0)
        del arg1206_1
        del arg1207_1
        del arg226_1
        del arg227_1
        # Source Nodes: [x_271, x_273, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf146 = extern_kernels.convolution(buf145, arg228_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg228_1
        del buf145
        buf147 = buf143; del buf143  # reuse
        # Source Nodes: [shortcut_30, x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf147, buf146, arg1209_1, arg1210_1, arg229_1, arg230_1, 225792, grid=grid(225792), stream=stream0)
        del arg1209_1
        del arg1210_1
        del arg229_1
        del arg230_1
        del buf146
        # Source Nodes: [x_279], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg231_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg231_1
        buf149 = buf148; del buf148  # reuse
        # Source Nodes: [x_280, x_282, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf149, arg1212_1, arg1213_1, arg232_1, arg233_1, 225792, grid=grid(225792), stream=stream0)
        del arg1212_1
        del arg1213_1
        del arg232_1
        del arg233_1
        # Source Nodes: [x_280, x_282, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf150 = extern_kernels.convolution(buf149, arg234_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg234_1
        del buf149
        buf151 = buf147; del buf147  # reuse
        # Source Nodes: [shortcut_31, x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf151, buf150, arg1215_1, arg1216_1, arg235_1, arg236_1, 225792, grid=grid(225792), stream=stream0)
        del arg1215_1
        del arg1216_1
        del arg235_1
        del arg236_1
        del buf150
        # Source Nodes: [x_288], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg237_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg237_1
        buf153 = buf152; del buf152  # reuse
        # Source Nodes: [x_289, x_291, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf153, arg1218_1, arg1219_1, arg238_1, arg239_1, 225792, grid=grid(225792), stream=stream0)
        del arg1218_1
        del arg1219_1
        del arg238_1
        del arg239_1
        # Source Nodes: [x_289, x_291, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf154 = extern_kernels.convolution(buf153, arg240_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg240_1
        del buf153
        buf155 = buf151; del buf151  # reuse
        # Source Nodes: [shortcut_32, x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf155, buf154, arg1221_1, arg1222_1, arg241_1, arg242_1, 225792, grid=grid(225792), stream=stream0)
        del arg1221_1
        del arg1222_1
        del arg241_1
        del arg242_1
        del buf154
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, arg243_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg243_1
        buf157 = buf156; del buf156  # reuse
        # Source Nodes: [x_298, x_300, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf157, arg1224_1, arg1225_1, arg244_1, arg245_1, 225792, grid=grid(225792), stream=stream0)
        del arg1224_1
        del arg1225_1
        del arg244_1
        del arg245_1
        # Source Nodes: [x_298, x_300, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf158 = extern_kernels.convolution(buf157, arg246_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg246_1
        del buf157
        buf159 = buf155; del buf155  # reuse
        # Source Nodes: [x_303, x_304, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf159, buf158, arg1227_1, arg1228_1, arg247_1, arg248_1, 225792, grid=grid(225792), stream=stream0)
        del arg1227_1
        del arg1228_1
        del arg247_1
        del arg248_1
        del buf158
        # Source Nodes: [l__mod___stage3_1_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg273_1
        # Source Nodes: [l__mod___stage3_0_fuse_layers_2_0_0_0], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf83, arg192_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg192_1
        buf162 = buf161; del buf161  # reuse
        # Source Nodes: [l__mod___stage3_0_fuse_layers_2_0_0_1, l__mod___stage3_0_fuse_layers_2_0_0_2, l__mod___stage3_0_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf162, arg1173_1, arg1174_1, arg193_1, arg194_1, 112896, grid=grid(112896), stream=stream0)
        del arg1173_1
        del arg1174_1
        del arg193_1
        del arg194_1
        # Source Nodes: [l__mod___stage3_0_fuse_layers_2_0_0_1, l__mod___stage3_0_fuse_layers_2_0_0_2, l__mod___stage3_0_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf163 = extern_kernels.convolution(buf162, arg195_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg195_1
        del buf162
        # Source Nodes: [l__mod___stage3_0_fuse_layers_2_1_0_0], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf101, arg198_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg198_1
        del buf101
        buf165 = buf163; del buf163  # reuse
        buf166 = buf120; del buf120  # reuse
        # Source Nodes: [l__mod___stage3_0_fuse_layers_2_1_0_1, shortcut_33, y_10, y_11, y_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf165, buf166, arg1176_1, arg1177_1, arg196_1, arg197_1, buf164, arg1179_1, arg1180_1, arg199_1, arg200_1, 112896, grid=grid(112896), stream=stream0)
        del arg1176_1
        del arg1177_1
        del arg1179_1
        del arg1180_1
        del arg196_1
        del arg197_1
        del arg199_1
        del arg200_1
        del buf164
        del buf165
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg249_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg249_1
        buf168 = buf167; del buf167  # reuse
        # Source Nodes: [x_307, x_309, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf168, arg1230_1, arg1231_1, arg250_1, arg251_1, 112896, grid=grid(112896), stream=stream0)
        del arg1230_1
        del arg1231_1
        del arg250_1
        del arg251_1
        # Source Nodes: [x_307, x_309, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf169 = extern_kernels.convolution(buf168, arg252_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg252_1
        del buf168
        buf170 = buf166; del buf166  # reuse
        # Source Nodes: [shortcut_34, x_312, x_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf170, buf169, arg1233_1, arg1234_1, arg253_1, arg254_1, 112896, grid=grid(112896), stream=stream0)
        del arg1233_1
        del arg1234_1
        del arg253_1
        del arg254_1
        del buf169
        # Source Nodes: [x_315], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg255_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg255_1
        buf172 = buf171; del buf171  # reuse
        # Source Nodes: [x_316, x_318, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf172, arg1236_1, arg1237_1, arg256_1, arg257_1, 112896, grid=grid(112896), stream=stream0)
        del arg1236_1
        del arg1237_1
        del arg256_1
        del arg257_1
        # Source Nodes: [x_316, x_318, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf173 = extern_kernels.convolution(buf172, arg258_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg258_1
        del buf172
        buf174 = buf170; del buf170  # reuse
        # Source Nodes: [shortcut_35, x_321, x_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf174, buf173, arg1239_1, arg1240_1, arg259_1, arg260_1, 112896, grid=grid(112896), stream=stream0)
        del arg1239_1
        del arg1240_1
        del arg259_1
        del arg260_1
        del buf173
        # Source Nodes: [x_324], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, arg261_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg261_1
        buf176 = buf175; del buf175  # reuse
        # Source Nodes: [x_325, x_327, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf176, arg1242_1, arg1243_1, arg262_1, arg263_1, 112896, grid=grid(112896), stream=stream0)
        del arg1242_1
        del arg1243_1
        del arg262_1
        del arg263_1
        # Source Nodes: [x_325, x_327, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf177 = extern_kernels.convolution(buf176, arg264_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg264_1
        del buf176
        buf178 = buf174; del buf174  # reuse
        # Source Nodes: [shortcut_36, x_330, x_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf178, buf177, arg1245_1, arg1246_1, arg265_1, arg266_1, 112896, grid=grid(112896), stream=stream0)
        del arg1245_1
        del arg1246_1
        del arg265_1
        del arg266_1
        del buf177
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, arg267_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg267_1
        buf180 = buf179; del buf179  # reuse
        # Source Nodes: [x_334, x_336, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf180, arg1248_1, arg1249_1, arg268_1, arg269_1, 112896, grid=grid(112896), stream=stream0)
        del arg1248_1
        del arg1249_1
        del arg268_1
        del arg269_1
        # Source Nodes: [x_334, x_336, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf181 = extern_kernels.convolution(buf180, arg270_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg270_1
        del buf180
        buf182 = buf178; del buf178  # reuse
        # Source Nodes: [x_339, x_340, x_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf182, buf181, arg1251_1, arg1252_1, arg271_1, arg272_1, 112896, grid=grid(112896), stream=stream0)
        del arg1251_1
        del arg1252_1
        del arg271_1
        del arg272_1
        del buf181
        # Source Nodes: [l__mod___stage3_1_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg276_1
        buf139 = buf135; del buf135  # reuse
        buf184 = buf83; del buf83  # reuse
        buf185 = buf184; del buf184  # reuse
        # Source Nodes: [l__mod___stage3_1_fuse_layers_0_1_1, l__mod___stage3_1_fuse_layers_0_1_2, l__mod___stage3_1_fuse_layers_0_2_1, l__mod___stage3_1_fuse_layers_0_2_2, shortcut_37, x_267, x_268, x_269, y_14, y_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_12.run(buf139, buf185, buf138, arg1203_1, arg1204_1, arg223_1, arg224_1, buf160, arg1254_1, arg1255_1, arg274_1, arg275_1, buf183, arg1257_1, arg1258_1, arg277_1, arg278_1, 451584, grid=grid(451584), stream=stream0)
        del arg1203_1
        del arg1204_1
        del arg1254_1
        del arg1255_1
        del arg1257_1
        del arg1258_1
        del arg223_1
        del arg224_1
        del arg274_1
        del arg275_1
        del arg277_1
        del arg278_1
        del buf138
        del buf160
        del buf183
        # Source Nodes: [x_342], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg294_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg294_1
        buf187 = buf186; del buf186  # reuse
        # Source Nodes: [x_343, x_345, x_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf187, arg1275_1, arg1276_1, arg295_1, arg296_1, 451584, grid=grid(451584), stream=stream0)
        del arg1275_1
        del arg1276_1
        del arg295_1
        del arg296_1
        # Source Nodes: [x_343, x_345, x_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf188 = extern_kernels.convolution(buf187, arg297_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg297_1
        del buf187
        buf189 = buf185; del buf185  # reuse
        # Source Nodes: [shortcut_38, x_348, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf189, buf188, arg1278_1, arg1279_1, arg298_1, arg299_1, 451584, grid=grid(451584), stream=stream0)
        del arg1278_1
        del arg1279_1
        del arg298_1
        del arg299_1
        del buf188
        # Source Nodes: [x_351], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, arg300_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg300_1
        buf191 = buf190; del buf190  # reuse
        # Source Nodes: [x_352, x_354, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf191, arg1281_1, arg1282_1, arg301_1, arg302_1, 451584, grid=grid(451584), stream=stream0)
        del arg1281_1
        del arg1282_1
        del arg301_1
        del arg302_1
        # Source Nodes: [x_352, x_354, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf192 = extern_kernels.convolution(buf191, arg303_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg303_1
        del buf191
        buf193 = buf189; del buf189  # reuse
        # Source Nodes: [shortcut_39, x_357, x_358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf193, buf192, arg1284_1, arg1285_1, arg304_1, arg305_1, 451584, grid=grid(451584), stream=stream0)
        del arg1284_1
        del arg1285_1
        del arg304_1
        del arg305_1
        del buf192
        # Source Nodes: [x_360], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, arg306_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg306_1
        buf195 = buf194; del buf194  # reuse
        # Source Nodes: [x_361, x_363, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf195, arg1287_1, arg1288_1, arg307_1, arg308_1, 451584, grid=grid(451584), stream=stream0)
        del arg1287_1
        del arg1288_1
        del arg307_1
        del arg308_1
        # Source Nodes: [x_361, x_363, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf196 = extern_kernels.convolution(buf195, arg309_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg309_1
        del buf195
        buf197 = buf193; del buf193  # reuse
        # Source Nodes: [shortcut_40, x_366, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf197, buf196, arg1290_1, arg1291_1, arg310_1, arg311_1, 451584, grid=grid(451584), stream=stream0)
        del arg1290_1
        del arg1291_1
        del arg310_1
        del arg311_1
        del buf196
        # Source Nodes: [x_369], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, arg312_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg312_1
        buf199 = buf198; del buf198  # reuse
        # Source Nodes: [x_370, x_372, x_374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf199, arg1293_1, arg1294_1, arg313_1, arg314_1, 451584, grid=grid(451584), stream=stream0)
        del arg1293_1
        del arg1294_1
        del arg313_1
        del arg314_1
        # Source Nodes: [x_370, x_372, x_374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf200 = extern_kernels.convolution(buf199, arg315_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg315_1
        del buf199
        # Source Nodes: [l__mod___stage3_1_fuse_layers_1_0_0_0], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf139, arg279_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg279_1
        # Source Nodes: [l__mod___stage3_1_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf182, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg282_1
        buf204 = buf202; del buf202  # reuse
        buf205 = buf204; del buf204  # reuse
        # Source Nodes: [l__mod___stage3_1_fuse_layers_1_2_1, l__mod___stage3_1_fuse_layers_1_2_2, shortcut_41, y_16, y_17, y_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_13.run(buf205, arg1260_1, arg1261_1, arg280_1, arg281_1, buf159, buf203, arg1263_1, arg1264_1, arg283_1, arg284_1, 225792, grid=grid(225792), stream=stream0)
        del arg1260_1
        del arg1261_1
        del arg1263_1
        del arg1264_1
        del arg280_1
        del arg281_1
        del arg283_1
        del arg284_1
        del buf203
        # Source Nodes: [x_378], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, arg318_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg318_1
        buf207 = buf206; del buf206  # reuse
        # Source Nodes: [x_379, x_381, x_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf207, arg1299_1, arg1300_1, arg319_1, arg320_1, 225792, grid=grid(225792), stream=stream0)
        del arg1299_1
        del arg1300_1
        del arg319_1
        del arg320_1
        # Source Nodes: [x_379, x_381, x_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf208 = extern_kernels.convolution(buf207, arg321_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg321_1
        del buf207
        buf209 = buf205; del buf205  # reuse
        # Source Nodes: [shortcut_42, x_384, x_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf209, buf208, arg1302_1, arg1303_1, arg322_1, arg323_1, 225792, grid=grid(225792), stream=stream0)
        del arg1302_1
        del arg1303_1
        del arg322_1
        del arg323_1
        del buf208
        # Source Nodes: [x_387], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, arg324_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg324_1
        buf211 = buf210; del buf210  # reuse
        # Source Nodes: [x_388, x_390, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf211, arg1305_1, arg1306_1, arg325_1, arg326_1, 225792, grid=grid(225792), stream=stream0)
        del arg1305_1
        del arg1306_1
        del arg325_1
        del arg326_1
        # Source Nodes: [x_388, x_390, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf212 = extern_kernels.convolution(buf211, arg327_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg327_1
        del buf211
        buf213 = buf209; del buf209  # reuse
        # Source Nodes: [shortcut_43, x_393, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf213, buf212, arg1308_1, arg1309_1, arg328_1, arg329_1, 225792, grid=grid(225792), stream=stream0)
        del arg1308_1
        del arg1309_1
        del arg328_1
        del arg329_1
        del buf212
        # Source Nodes: [x_396], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, arg330_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg330_1
        buf215 = buf214; del buf214  # reuse
        # Source Nodes: [x_397, x_399, x_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf215, arg1311_1, arg1312_1, arg331_1, arg332_1, 225792, grid=grid(225792), stream=stream0)
        del arg1311_1
        del arg1312_1
        del arg331_1
        del arg332_1
        # Source Nodes: [x_397, x_399, x_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf216 = extern_kernels.convolution(buf215, arg333_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg333_1
        del buf215
        buf217 = buf213; del buf213  # reuse
        # Source Nodes: [shortcut_44, x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf217, buf216, arg1314_1, arg1315_1, arg334_1, arg335_1, 225792, grid=grid(225792), stream=stream0)
        del arg1314_1
        del arg1315_1
        del arg334_1
        del arg335_1
        del buf216
        # Source Nodes: [x_405], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, arg336_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg336_1
        buf219 = buf218; del buf218  # reuse
        # Source Nodes: [x_406, x_408, x_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf219, arg1317_1, arg1318_1, arg337_1, arg338_1, 225792, grid=grid(225792), stream=stream0)
        del arg1317_1
        del arg1318_1
        del arg337_1
        del arg338_1
        # Source Nodes: [x_406, x_408, x_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf220 = extern_kernels.convolution(buf219, arg339_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg339_1
        del buf219
        buf221 = buf217; del buf217  # reuse
        # Source Nodes: [x_411, x_412, x_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf221, buf220, arg1320_1, arg1321_1, arg340_1, arg341_1, 225792, grid=grid(225792), stream=stream0)
        del arg1320_1
        del arg1321_1
        del arg340_1
        del arg341_1
        del buf220
        # Source Nodes: [l__mod___stage3_2_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, arg366_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg366_1
        # Source Nodes: [l__mod___stage3_1_fuse_layers_2_0_0_0], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf139, arg285_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg285_1
        buf224 = buf223; del buf223  # reuse
        # Source Nodes: [l__mod___stage3_1_fuse_layers_2_0_0_1, l__mod___stage3_1_fuse_layers_2_0_0_2, l__mod___stage3_1_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf224, arg1266_1, arg1267_1, arg286_1, arg287_1, 112896, grid=grid(112896), stream=stream0)
        del arg1266_1
        del arg1267_1
        del arg286_1
        del arg287_1
        # Source Nodes: [l__mod___stage3_1_fuse_layers_2_0_0_1, l__mod___stage3_1_fuse_layers_2_0_0_2, l__mod___stage3_1_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf225 = extern_kernels.convolution(buf224, arg288_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg288_1
        del buf224
        # Source Nodes: [l__mod___stage3_1_fuse_layers_2_1_0_0], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf159, arg291_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg291_1
        del buf159
        buf227 = buf225; del buf225  # reuse
        buf228 = buf182; del buf182  # reuse
        # Source Nodes: [l__mod___stage3_1_fuse_layers_2_1_0_1, shortcut_45, y_19, y_20, y_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf227, buf228, arg1269_1, arg1270_1, arg289_1, arg290_1, buf226, arg1272_1, arg1273_1, arg292_1, arg293_1, 112896, grid=grid(112896), stream=stream0)
        del arg1269_1
        del arg1270_1
        del arg1272_1
        del arg1273_1
        del arg289_1
        del arg290_1
        del arg292_1
        del arg293_1
        del buf226
        del buf227
        # Source Nodes: [x_414], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, arg342_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg342_1
        buf230 = buf229; del buf229  # reuse
        # Source Nodes: [x_415, x_417, x_419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf230, arg1323_1, arg1324_1, arg343_1, arg344_1, 112896, grid=grid(112896), stream=stream0)
        del arg1323_1
        del arg1324_1
        del arg343_1
        del arg344_1
        # Source Nodes: [x_415, x_417, x_419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf231 = extern_kernels.convolution(buf230, arg345_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg345_1
        del buf230
        buf232 = buf228; del buf228  # reuse
        # Source Nodes: [shortcut_46, x_420, x_421], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf232, buf231, arg1326_1, arg1327_1, arg346_1, arg347_1, 112896, grid=grid(112896), stream=stream0)
        del arg1326_1
        del arg1327_1
        del arg346_1
        del arg347_1
        del buf231
        # Source Nodes: [x_423], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, arg348_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg348_1
        buf234 = buf233; del buf233  # reuse
        # Source Nodes: [x_424, x_426, x_428], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf234, arg1329_1, arg1330_1, arg349_1, arg350_1, 112896, grid=grid(112896), stream=stream0)
        del arg1329_1
        del arg1330_1
        del arg349_1
        del arg350_1
        # Source Nodes: [x_424, x_426, x_428], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf235 = extern_kernels.convolution(buf234, arg351_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg351_1
        del buf234
        buf236 = buf232; del buf232  # reuse
        # Source Nodes: [shortcut_47, x_429, x_430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf236, buf235, arg1332_1, arg1333_1, arg352_1, arg353_1, 112896, grid=grid(112896), stream=stream0)
        del arg1332_1
        del arg1333_1
        del arg352_1
        del arg353_1
        del buf235
        # Source Nodes: [x_432], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg354_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg354_1
        buf238 = buf237; del buf237  # reuse
        # Source Nodes: [x_433, x_435, x_437], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf238, arg1335_1, arg1336_1, arg355_1, arg356_1, 112896, grid=grid(112896), stream=stream0)
        del arg1335_1
        del arg1336_1
        del arg355_1
        del arg356_1
        # Source Nodes: [x_433, x_435, x_437], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf239 = extern_kernels.convolution(buf238, arg357_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg357_1
        del buf238
        buf240 = buf236; del buf236  # reuse
        # Source Nodes: [shortcut_48, x_438, x_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf240, buf239, arg1338_1, arg1339_1, arg358_1, arg359_1, 112896, grid=grid(112896), stream=stream0)
        del arg1338_1
        del arg1339_1
        del arg358_1
        del arg359_1
        del buf239
        # Source Nodes: [x_441], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, arg360_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg360_1
        buf242 = buf241; del buf241  # reuse
        # Source Nodes: [x_442, x_444, x_446], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf242, arg1341_1, arg1342_1, arg361_1, arg362_1, 112896, grid=grid(112896), stream=stream0)
        del arg1341_1
        del arg1342_1
        del arg361_1
        del arg362_1
        # Source Nodes: [x_442, x_444, x_446], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf243 = extern_kernels.convolution(buf242, arg363_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg363_1
        del buf242
        buf244 = buf240; del buf240  # reuse
        # Source Nodes: [x_447, x_448, x_449], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf244, buf243, arg1344_1, arg1345_1, arg364_1, arg365_1, 112896, grid=grid(112896), stream=stream0)
        del arg1344_1
        del arg1345_1
        del arg364_1
        del arg365_1
        del buf243
        # Source Nodes: [l__mod___stage3_2_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, arg369_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg369_1
        buf201 = buf197; del buf197  # reuse
        buf246 = buf139; del buf139  # reuse
        buf247 = buf246; del buf246  # reuse
        # Source Nodes: [l__mod___stage3_2_fuse_layers_0_1_1, l__mod___stage3_2_fuse_layers_0_1_2, l__mod___stage3_2_fuse_layers_0_2_1, l__mod___stage3_2_fuse_layers_0_2_2, shortcut_49, x_375, x_376, x_377, y_23, y_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_12.run(buf201, buf247, buf200, arg1296_1, arg1297_1, arg316_1, arg317_1, buf222, arg1347_1, arg1348_1, arg367_1, arg368_1, buf245, arg1350_1, arg1351_1, arg370_1, arg371_1, 451584, grid=grid(451584), stream=stream0)
        del arg1296_1
        del arg1297_1
        del arg1347_1
        del arg1348_1
        del arg1350_1
        del arg1351_1
        del arg316_1
        del arg317_1
        del arg367_1
        del arg368_1
        del arg370_1
        del arg371_1
        del buf200
        del buf222
        del buf245
        # Source Nodes: [x_450], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, arg387_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg387_1
        buf249 = buf248; del buf248  # reuse
        # Source Nodes: [x_451, x_453, x_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf249, arg1368_1, arg1369_1, arg388_1, arg389_1, 451584, grid=grid(451584), stream=stream0)
        del arg1368_1
        del arg1369_1
        del arg388_1
        del arg389_1
        # Source Nodes: [x_451, x_453, x_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf250 = extern_kernels.convolution(buf249, arg390_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg390_1
        del buf249
        buf251 = buf247; del buf247  # reuse
        # Source Nodes: [shortcut_50, x_456, x_457], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf251, buf250, arg1371_1, arg1372_1, arg391_1, arg392_1, 451584, grid=grid(451584), stream=stream0)
        del arg1371_1
        del arg1372_1
        del arg391_1
        del arg392_1
        del buf250
        # Source Nodes: [x_459], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, arg393_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg393_1
        buf253 = buf252; del buf252  # reuse
        # Source Nodes: [x_460, x_462, x_464], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf253, arg1374_1, arg1375_1, arg394_1, arg395_1, 451584, grid=grid(451584), stream=stream0)
        del arg1374_1
        del arg1375_1
        del arg394_1
        del arg395_1
        # Source Nodes: [x_460, x_462, x_464], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf254 = extern_kernels.convolution(buf253, arg396_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg396_1
        del buf253
        buf255 = buf251; del buf251  # reuse
        # Source Nodes: [shortcut_51, x_465, x_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf255, buf254, arg1377_1, arg1378_1, arg397_1, arg398_1, 451584, grid=grid(451584), stream=stream0)
        del arg1377_1
        del arg1378_1
        del arg397_1
        del arg398_1
        del buf254
        # Source Nodes: [x_468], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, arg399_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg399_1
        buf257 = buf256; del buf256  # reuse
        # Source Nodes: [x_469, x_471, x_473], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf257, arg1380_1, arg1381_1, arg400_1, arg401_1, 451584, grid=grid(451584), stream=stream0)
        del arg1380_1
        del arg1381_1
        del arg400_1
        del arg401_1
        # Source Nodes: [x_469, x_471, x_473], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf258 = extern_kernels.convolution(buf257, arg402_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg402_1
        del buf257
        buf259 = buf255; del buf255  # reuse
        # Source Nodes: [shortcut_52, x_474, x_475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf259, buf258, arg1383_1, arg1384_1, arg403_1, arg404_1, 451584, grid=grid(451584), stream=stream0)
        del arg1383_1
        del arg1384_1
        del arg403_1
        del arg404_1
        del buf258
        # Source Nodes: [x_477], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, arg405_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg405_1
        buf261 = buf260; del buf260  # reuse
        # Source Nodes: [x_478, x_480, x_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf261, arg1386_1, arg1387_1, arg406_1, arg407_1, 451584, grid=grid(451584), stream=stream0)
        del arg1386_1
        del arg1387_1
        del arg406_1
        del arg407_1
        # Source Nodes: [x_478, x_480, x_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf262 = extern_kernels.convolution(buf261, arg408_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg408_1
        del buf261
        # Source Nodes: [l__mod___stage3_2_fuse_layers_1_0_0_0], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf201, arg372_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg372_1
        # Source Nodes: [l__mod___stage3_2_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf244, arg375_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg375_1
        buf266 = buf264; del buf264  # reuse
        buf267 = buf266; del buf266  # reuse
        # Source Nodes: [l__mod___stage3_2_fuse_layers_1_2_1, l__mod___stage3_2_fuse_layers_1_2_2, shortcut_53, y_25, y_26, y_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_13.run(buf267, arg1353_1, arg1354_1, arg373_1, arg374_1, buf221, buf265, arg1356_1, arg1357_1, arg376_1, arg377_1, 225792, grid=grid(225792), stream=stream0)
        del arg1353_1
        del arg1354_1
        del arg1356_1
        del arg1357_1
        del arg373_1
        del arg374_1
        del arg376_1
        del arg377_1
        del buf265
        # Source Nodes: [x_486], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, arg411_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg411_1
        buf269 = buf268; del buf268  # reuse
        # Source Nodes: [x_487, x_489, x_491], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf269, arg1392_1, arg1393_1, arg412_1, arg413_1, 225792, grid=grid(225792), stream=stream0)
        del arg1392_1
        del arg1393_1
        del arg412_1
        del arg413_1
        # Source Nodes: [x_487, x_489, x_491], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf270 = extern_kernels.convolution(buf269, arg414_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg414_1
        del buf269
        buf271 = buf267; del buf267  # reuse
        # Source Nodes: [shortcut_54, x_492, x_493], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf271, buf270, arg1395_1, arg1396_1, arg415_1, arg416_1, 225792, grid=grid(225792), stream=stream0)
        del arg1395_1
        del arg1396_1
        del arg415_1
        del arg416_1
        del buf270
        # Source Nodes: [x_495], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, arg417_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg417_1
        buf273 = buf272; del buf272  # reuse
        # Source Nodes: [x_496, x_498, x_500], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf273, arg1398_1, arg1399_1, arg418_1, arg419_1, 225792, grid=grid(225792), stream=stream0)
        del arg1398_1
        del arg1399_1
        del arg418_1
        del arg419_1
        # Source Nodes: [x_496, x_498, x_500], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf274 = extern_kernels.convolution(buf273, arg420_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg420_1
        del buf273
        buf275 = buf271; del buf271  # reuse
        # Source Nodes: [shortcut_55, x_501, x_502], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf275, buf274, arg1401_1, arg1402_1, arg421_1, arg422_1, 225792, grid=grid(225792), stream=stream0)
        del arg1401_1
        del arg1402_1
        del arg421_1
        del arg422_1
        del buf274
        # Source Nodes: [x_504], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, arg423_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg423_1
        buf277 = buf276; del buf276  # reuse
        # Source Nodes: [x_505, x_507, x_509], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf277, arg1404_1, arg1405_1, arg424_1, arg425_1, 225792, grid=grid(225792), stream=stream0)
        del arg1404_1
        del arg1405_1
        del arg424_1
        del arg425_1
        # Source Nodes: [x_505, x_507, x_509], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf278 = extern_kernels.convolution(buf277, arg426_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg426_1
        del buf277
        buf279 = buf275; del buf275  # reuse
        # Source Nodes: [shortcut_56, x_510, x_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf279, buf278, arg1407_1, arg1408_1, arg427_1, arg428_1, 225792, grid=grid(225792), stream=stream0)
        del arg1407_1
        del arg1408_1
        del arg427_1
        del arg428_1
        del buf278
        # Source Nodes: [x_513], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, arg429_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg429_1
        buf281 = buf280; del buf280  # reuse
        # Source Nodes: [x_514, x_516, x_518], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf281, arg1410_1, arg1411_1, arg430_1, arg431_1, 225792, grid=grid(225792), stream=stream0)
        del arg1410_1
        del arg1411_1
        del arg430_1
        del arg431_1
        # Source Nodes: [x_514, x_516, x_518], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf282 = extern_kernels.convolution(buf281, arg432_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg432_1
        del buf281
        buf283 = buf279; del buf279  # reuse
        # Source Nodes: [x_519, x_520, x_521], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf283, buf282, arg1413_1, arg1414_1, arg433_1, arg434_1, 225792, grid=grid(225792), stream=stream0)
        del arg1413_1
        del arg1414_1
        del arg433_1
        del arg434_1
        del buf282
        # Source Nodes: [l__mod___stage3_3_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, arg459_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg459_1
        # Source Nodes: [l__mod___stage3_2_fuse_layers_2_0_0_0], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf201, arg378_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg378_1
        buf286 = buf285; del buf285  # reuse
        # Source Nodes: [l__mod___stage3_2_fuse_layers_2_0_0_1, l__mod___stage3_2_fuse_layers_2_0_0_2, l__mod___stage3_2_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf286, arg1359_1, arg1360_1, arg379_1, arg380_1, 112896, grid=grid(112896), stream=stream0)
        del arg1359_1
        del arg1360_1
        del arg379_1
        del arg380_1
        # Source Nodes: [l__mod___stage3_2_fuse_layers_2_0_0_1, l__mod___stage3_2_fuse_layers_2_0_0_2, l__mod___stage3_2_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf287 = extern_kernels.convolution(buf286, arg381_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg381_1
        del buf286
        # Source Nodes: [l__mod___stage3_2_fuse_layers_2_1_0_0], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf221, arg384_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg384_1
        del buf221
        buf289 = buf287; del buf287  # reuse
        buf290 = buf244; del buf244  # reuse
        # Source Nodes: [l__mod___stage3_2_fuse_layers_2_1_0_1, shortcut_57, y_28, y_29, y_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf289, buf290, arg1362_1, arg1363_1, arg382_1, arg383_1, buf288, arg1365_1, arg1366_1, arg385_1, arg386_1, 112896, grid=grid(112896), stream=stream0)
        del arg1362_1
        del arg1363_1
        del arg1365_1
        del arg1366_1
        del arg382_1
        del arg383_1
        del arg385_1
        del arg386_1
        del buf288
        del buf289
        # Source Nodes: [x_522], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, arg435_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg435_1
        buf292 = buf291; del buf291  # reuse
        # Source Nodes: [x_523, x_525, x_527], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf292, arg1416_1, arg1417_1, arg436_1, arg437_1, 112896, grid=grid(112896), stream=stream0)
        del arg1416_1
        del arg1417_1
        del arg436_1
        del arg437_1
        # Source Nodes: [x_523, x_525, x_527], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf293 = extern_kernels.convolution(buf292, arg438_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg438_1
        del buf292
        buf294 = buf290; del buf290  # reuse
        # Source Nodes: [shortcut_58, x_528, x_529], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf294, buf293, arg1419_1, arg1420_1, arg439_1, arg440_1, 112896, grid=grid(112896), stream=stream0)
        del arg1419_1
        del arg1420_1
        del arg439_1
        del arg440_1
        del buf293
        # Source Nodes: [x_531], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, arg441_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg441_1
        buf296 = buf295; del buf295  # reuse
        # Source Nodes: [x_532, x_534, x_536], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf296, arg1422_1, arg1423_1, arg442_1, arg443_1, 112896, grid=grid(112896), stream=stream0)
        del arg1422_1
        del arg1423_1
        del arg442_1
        del arg443_1
        # Source Nodes: [x_532, x_534, x_536], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf297 = extern_kernels.convolution(buf296, arg444_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg444_1
        del buf296
        buf298 = buf294; del buf294  # reuse
        # Source Nodes: [shortcut_59, x_537, x_538], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf298, buf297, arg1425_1, arg1426_1, arg445_1, arg446_1, 112896, grid=grid(112896), stream=stream0)
        del arg1425_1
        del arg1426_1
        del arg445_1
        del arg446_1
        del buf297
        # Source Nodes: [x_540], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, arg447_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg447_1
        buf300 = buf299; del buf299  # reuse
        # Source Nodes: [x_541, x_543, x_545], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf300, arg1428_1, arg1429_1, arg448_1, arg449_1, 112896, grid=grid(112896), stream=stream0)
        del arg1428_1
        del arg1429_1
        del arg448_1
        del arg449_1
        # Source Nodes: [x_541, x_543, x_545], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf301 = extern_kernels.convolution(buf300, arg450_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg450_1
        del buf300
        buf302 = buf298; del buf298  # reuse
        # Source Nodes: [shortcut_60, x_546, x_547], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf302, buf301, arg1431_1, arg1432_1, arg451_1, arg452_1, 112896, grid=grid(112896), stream=stream0)
        del arg1431_1
        del arg1432_1
        del arg451_1
        del arg452_1
        del buf301
        # Source Nodes: [x_549], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, arg453_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg453_1
        buf304 = buf303; del buf303  # reuse
        # Source Nodes: [x_550, x_552, x_554], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf304, arg1434_1, arg1435_1, arg454_1, arg455_1, 112896, grid=grid(112896), stream=stream0)
        del arg1434_1
        del arg1435_1
        del arg454_1
        del arg455_1
        # Source Nodes: [x_550, x_552, x_554], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf305 = extern_kernels.convolution(buf304, arg456_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg456_1
        del buf304
        buf306 = buf302; del buf302  # reuse
        # Source Nodes: [x_555, x_556, x_557], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf306, buf305, arg1437_1, arg1438_1, arg457_1, arg458_1, 112896, grid=grid(112896), stream=stream0)
        del arg1437_1
        del arg1438_1
        del arg457_1
        del arg458_1
        del buf305
        # Source Nodes: [l__mod___stage3_3_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, arg462_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg462_1
        buf263 = buf259; del buf259  # reuse
        buf308 = buf201; del buf201  # reuse
        buf309 = buf308; del buf308  # reuse
        # Source Nodes: [l__mod___stage3_3_fuse_layers_0_1_1, l__mod___stage3_3_fuse_layers_0_1_2, l__mod___stage3_3_fuse_layers_0_2_1, l__mod___stage3_3_fuse_layers_0_2_2, shortcut_61, x_483, x_484, x_485, y_32, y_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_12.run(buf263, buf309, buf262, arg1389_1, arg1390_1, arg409_1, arg410_1, buf284, arg1440_1, arg1441_1, arg460_1, arg461_1, buf307, arg1443_1, arg1444_1, arg463_1, arg464_1, 451584, grid=grid(451584), stream=stream0)
        del arg1389_1
        del arg1390_1
        del arg1440_1
        del arg1441_1
        del arg1443_1
        del arg1444_1
        del arg409_1
        del arg410_1
        del arg460_1
        del arg461_1
        del arg463_1
        del arg464_1
        del buf262
        del buf284
        del buf307
        # Source Nodes: [x_558], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, arg483_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg483_1
        buf311 = buf310; del buf310  # reuse
        # Source Nodes: [x_559, x_561, x_563], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf311, arg1464_1, arg1465_1, arg484_1, arg485_1, 451584, grid=grid(451584), stream=stream0)
        del arg1464_1
        del arg1465_1
        del arg484_1
        del arg485_1
        # Source Nodes: [x_559, x_561, x_563], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf312 = extern_kernels.convolution(buf311, arg486_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg486_1
        del buf311
        buf313 = buf309; del buf309  # reuse
        # Source Nodes: [shortcut_62, x_564, x_565], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf313, buf312, arg1467_1, arg1468_1, arg487_1, arg488_1, 451584, grid=grid(451584), stream=stream0)
        del arg1467_1
        del arg1468_1
        del arg487_1
        del arg488_1
        del buf312
        # Source Nodes: [x_567], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, arg489_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg489_1
        buf315 = buf314; del buf314  # reuse
        # Source Nodes: [x_568, x_570, x_572], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf315, arg1470_1, arg1471_1, arg490_1, arg491_1, 451584, grid=grid(451584), stream=stream0)
        del arg1470_1
        del arg1471_1
        del arg490_1
        del arg491_1
        # Source Nodes: [x_568, x_570, x_572], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf316 = extern_kernels.convolution(buf315, arg492_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg492_1
        del buf315
        buf317 = buf313; del buf313  # reuse
        # Source Nodes: [shortcut_63, x_573, x_574], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf317, buf316, arg1473_1, arg1474_1, arg493_1, arg494_1, 451584, grid=grid(451584), stream=stream0)
        del arg1473_1
        del arg1474_1
        del arg493_1
        del arg494_1
        del buf316
        # Source Nodes: [x_576], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, arg495_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg495_1
        buf319 = buf318; del buf318  # reuse
        # Source Nodes: [x_577, x_579, x_581], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf319, arg1476_1, arg1477_1, arg496_1, arg497_1, 451584, grid=grid(451584), stream=stream0)
        del arg1476_1
        del arg1477_1
        del arg496_1
        del arg497_1
        # Source Nodes: [x_577, x_579, x_581], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf320 = extern_kernels.convolution(buf319, arg498_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg498_1
        del buf319
        buf321 = buf317; del buf317  # reuse
        # Source Nodes: [shortcut_64, x_582, x_583], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf321, buf320, arg1479_1, arg1480_1, arg499_1, arg500_1, 451584, grid=grid(451584), stream=stream0)
        del arg1479_1
        del arg1480_1
        del arg499_1
        del arg500_1
        del buf320
        # Source Nodes: [x_585], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, arg501_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg501_1
        buf323 = buf322; del buf322  # reuse
        # Source Nodes: [x_586, x_588, x_590], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf323, arg1482_1, arg1483_1, arg502_1, arg503_1, 451584, grid=grid(451584), stream=stream0)
        del arg1482_1
        del arg1483_1
        del arg502_1
        del arg503_1
        # Source Nodes: [x_586, x_588, x_590], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf324 = extern_kernels.convolution(buf323, arg504_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg504_1
        del buf323
        # Source Nodes: [l__mod___stage3_3_fuse_layers_1_0_0_0], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf263, arg465_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg465_1
        # Source Nodes: [l__mod___stage3_3_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf306, arg468_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg468_1
        buf328 = buf326; del buf326  # reuse
        buf329 = buf328; del buf328  # reuse
        # Source Nodes: [l__mod___stage3_3_fuse_layers_1_2_1, l__mod___stage3_3_fuse_layers_1_2_2, shortcut_65, y_34, y_35, y_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_13.run(buf329, arg1446_1, arg1447_1, arg466_1, arg467_1, buf283, buf327, arg1449_1, arg1450_1, arg469_1, arg470_1, 225792, grid=grid(225792), stream=stream0)
        del arg1446_1
        del arg1447_1
        del arg1449_1
        del arg1450_1
        del arg466_1
        del arg467_1
        del arg469_1
        del arg470_1
        del buf327
        # Source Nodes: [x_594], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, arg507_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg507_1
        buf331 = buf330; del buf330  # reuse
        # Source Nodes: [x_595, x_597, x_599], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf331, arg1488_1, arg1489_1, arg508_1, arg509_1, 225792, grid=grid(225792), stream=stream0)
        del arg1488_1
        del arg1489_1
        del arg508_1
        del arg509_1
        # Source Nodes: [x_595, x_597, x_599], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf332 = extern_kernels.convolution(buf331, arg510_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg510_1
        del buf331
        buf333 = buf329; del buf329  # reuse
        # Source Nodes: [shortcut_66, x_600, x_601], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf333, buf332, arg1491_1, arg1492_1, arg511_1, arg512_1, 225792, grid=grid(225792), stream=stream0)
        del arg1491_1
        del arg1492_1
        del arg511_1
        del arg512_1
        del buf332
        # Source Nodes: [x_603], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, arg513_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg513_1
        buf335 = buf334; del buf334  # reuse
        # Source Nodes: [x_604, x_606, x_608], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf335, arg1494_1, arg1495_1, arg514_1, arg515_1, 225792, grid=grid(225792), stream=stream0)
        del arg1494_1
        del arg1495_1
        del arg514_1
        del arg515_1
        # Source Nodes: [x_604, x_606, x_608], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf336 = extern_kernels.convolution(buf335, arg516_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg516_1
        del buf335
        buf337 = buf333; del buf333  # reuse
        # Source Nodes: [shortcut_67, x_609, x_610], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf337, buf336, arg1497_1, arg1498_1, arg517_1, arg518_1, 225792, grid=grid(225792), stream=stream0)
        del arg1497_1
        del arg1498_1
        del arg517_1
        del arg518_1
        del buf336
        # Source Nodes: [x_612], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, arg519_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg519_1
        buf339 = buf338; del buf338  # reuse
        # Source Nodes: [x_613, x_615, x_617], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf339, arg1500_1, arg1501_1, arg520_1, arg521_1, 225792, grid=grid(225792), stream=stream0)
        del arg1500_1
        del arg1501_1
        del arg520_1
        del arg521_1
        # Source Nodes: [x_613, x_615, x_617], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf340 = extern_kernels.convolution(buf339, arg522_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg522_1
        del buf339
        buf341 = buf337; del buf337  # reuse
        # Source Nodes: [shortcut_68, x_618, x_619], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf341, buf340, arg1503_1, arg1504_1, arg523_1, arg524_1, 225792, grid=grid(225792), stream=stream0)
        del arg1503_1
        del arg1504_1
        del arg523_1
        del arg524_1
        del buf340
        # Source Nodes: [x_621], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, arg525_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg525_1
        buf343 = buf342; del buf342  # reuse
        # Source Nodes: [x_622, x_624, x_626], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf343, arg1506_1, arg1507_1, arg526_1, arg527_1, 225792, grid=grid(225792), stream=stream0)
        del arg1506_1
        del arg1507_1
        del arg526_1
        del arg527_1
        # Source Nodes: [x_622, x_624, x_626], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf344 = extern_kernels.convolution(buf343, arg528_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg528_1
        del buf343
        buf345 = buf341; del buf341  # reuse
        # Source Nodes: [x_627, x_628, x_629], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf345, buf344, arg1509_1, arg1510_1, arg529_1, arg530_1, 225792, grid=grid(225792), stream=stream0)
        del arg1509_1
        del arg1510_1
        del arg529_1
        del arg530_1
        del buf344
        # Source Nodes: [l__mod___stage4_0_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, arg579_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg579_1
        # Source Nodes: [l__mod___stage3_3_fuse_layers_2_0_0_0], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf263, arg471_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg471_1
        buf348 = buf347; del buf347  # reuse
        # Source Nodes: [l__mod___stage3_3_fuse_layers_2_0_0_1, l__mod___stage3_3_fuse_layers_2_0_0_2, l__mod___stage3_3_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf348, arg1452_1, arg1453_1, arg472_1, arg473_1, 112896, grid=grid(112896), stream=stream0)
        del arg1452_1
        del arg1453_1
        del arg472_1
        del arg473_1
        # Source Nodes: [l__mod___stage3_3_fuse_layers_2_0_0_1, l__mod___stage3_3_fuse_layers_2_0_0_2, l__mod___stage3_3_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf349 = extern_kernels.convolution(buf348, arg474_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg474_1
        del buf348
        # Source Nodes: [l__mod___stage3_3_fuse_layers_2_1_0_0], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf283, arg477_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg477_1
        del buf283
        buf351 = buf349; del buf349  # reuse
        buf352 = buf306; del buf306  # reuse
        # Source Nodes: [l__mod___stage3_3_fuse_layers_2_1_0_1, shortcut_69, y_37, y_38, y_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf351, buf352, arg1455_1, arg1456_1, arg475_1, arg476_1, buf350, arg1458_1, arg1459_1, arg478_1, arg479_1, 112896, grid=grid(112896), stream=stream0)
        del arg1455_1
        del arg1456_1
        del arg1458_1
        del arg1459_1
        del arg475_1
        del arg476_1
        del arg478_1
        del arg479_1
        del buf350
        del buf351
        # Source Nodes: [x_630], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, arg531_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg531_1
        buf354 = buf353; del buf353  # reuse
        # Source Nodes: [x_631, x_633, x_635], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf354, arg1512_1, arg1513_1, arg532_1, arg533_1, 112896, grid=grid(112896), stream=stream0)
        del arg1512_1
        del arg1513_1
        del arg532_1
        del arg533_1
        # Source Nodes: [x_631, x_633, x_635], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf355 = extern_kernels.convolution(buf354, arg534_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg534_1
        del buf354
        buf356 = buf355; del buf355  # reuse
        # Source Nodes: [shortcut_70, x_636, x_637], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf356, arg1515_1, arg1516_1, arg535_1, arg536_1, buf352, 112896, grid=grid(112896), stream=stream0)
        del arg1515_1
        del arg1516_1
        del arg535_1
        del arg536_1
        # Source Nodes: [x_639], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf356, arg537_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg537_1
        buf358 = buf357; del buf357  # reuse
        # Source Nodes: [x_640, x_642, x_644], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf358, arg1518_1, arg1519_1, arg538_1, arg539_1, 112896, grid=grid(112896), stream=stream0)
        del arg1518_1
        del arg1519_1
        del arg538_1
        del arg539_1
        # Source Nodes: [x_640, x_642, x_644], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf359 = extern_kernels.convolution(buf358, arg540_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg540_1
        del buf358
        buf360 = buf356; del buf356  # reuse
        # Source Nodes: [shortcut_71, x_645, x_646], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf360, buf359, arg1521_1, arg1522_1, arg541_1, arg542_1, 112896, grid=grid(112896), stream=stream0)
        del arg1521_1
        del arg1522_1
        del arg541_1
        del arg542_1
        del buf359
        # Source Nodes: [x_648], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, arg543_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg543_1
        buf362 = buf361; del buf361  # reuse
        # Source Nodes: [x_649, x_651, x_653], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf362, arg1524_1, arg1525_1, arg544_1, arg545_1, 112896, grid=grid(112896), stream=stream0)
        del arg1524_1
        del arg1525_1
        del arg544_1
        del arg545_1
        # Source Nodes: [x_649, x_651, x_653], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf363 = extern_kernels.convolution(buf362, arg546_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg546_1
        del buf362
        buf364 = buf360; del buf360  # reuse
        # Source Nodes: [shortcut_72, x_654, x_655], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf364, buf363, arg1527_1, arg1528_1, arg547_1, arg548_1, 112896, grid=grid(112896), stream=stream0)
        del arg1527_1
        del arg1528_1
        del arg547_1
        del arg548_1
        del buf363
        # Source Nodes: [x_657], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, arg549_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg549_1
        buf366 = buf365; del buf365  # reuse
        # Source Nodes: [x_658, x_660, x_662], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf366, arg1530_1, arg1531_1, arg550_1, arg551_1, 112896, grid=grid(112896), stream=stream0)
        del arg1530_1
        del arg1531_1
        del arg550_1
        del arg551_1
        # Source Nodes: [x_658, x_660, x_662], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf367 = extern_kernels.convolution(buf366, arg552_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg552_1
        del buf366
        buf368 = buf364; del buf364  # reuse
        # Source Nodes: [x_663, x_664, x_665], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf368, buf367, arg1533_1, arg1534_1, arg553_1, arg554_1, 112896, grid=grid(112896), stream=stream0)
        del arg1533_1
        del arg1534_1
        del arg553_1
        del arg554_1
        del buf367
        # Source Nodes: [l__mod___stage4_0_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf368, arg582_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg582_1
        # Source Nodes: [l__mod___transition3_3_0_0], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf352, arg480_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg480_1
        del buf352
        buf372 = buf371; del buf371  # reuse
        # Source Nodes: [l__mod___transition3_3_0_1, shortcut_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf372, arg1461_1, arg1462_1, arg481_1, arg482_1, 56448, grid=grid(56448), stream=stream0)
        del arg1461_1
        del arg1462_1
        del arg481_1
        del arg482_1
        # Source Nodes: [x_666], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, arg555_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg555_1
        buf374 = buf373; del buf373  # reuse
        # Source Nodes: [x_667, x_669, x_671], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf374, arg1536_1, arg1537_1, arg556_1, arg557_1, 56448, grid=grid(56448), stream=stream0)
        del arg1536_1
        del arg1537_1
        del arg556_1
        del arg557_1
        # Source Nodes: [x_667, x_669, x_671], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf375 = extern_kernels.convolution(buf374, arg558_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg558_1
        del buf374
        buf376 = buf372; del buf372  # reuse
        # Source Nodes: [shortcut_74, x_672, x_673], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf376, buf375, arg1539_1, arg1540_1, arg559_1, arg560_1, 56448, grid=grid(56448), stream=stream0)
        del arg1539_1
        del arg1540_1
        del arg559_1
        del arg560_1
        del buf375
        # Source Nodes: [x_675], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, arg561_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg561_1
        buf378 = buf377; del buf377  # reuse
        # Source Nodes: [x_676, x_678, x_680], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf378, arg1542_1, arg1543_1, arg562_1, arg563_1, 56448, grid=grid(56448), stream=stream0)
        del arg1542_1
        del arg1543_1
        del arg562_1
        del arg563_1
        # Source Nodes: [x_676, x_678, x_680], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf379 = extern_kernels.convolution(buf378, arg564_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg564_1
        del buf378
        buf380 = buf376; del buf376  # reuse
        # Source Nodes: [shortcut_75, x_681, x_682], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf380, buf379, arg1545_1, arg1546_1, arg565_1, arg566_1, 56448, grid=grid(56448), stream=stream0)
        del arg1545_1
        del arg1546_1
        del arg565_1
        del arg566_1
        del buf379
        # Source Nodes: [x_684], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, arg567_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg567_1
        buf382 = buf381; del buf381  # reuse
        # Source Nodes: [x_685, x_687, x_689], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf382, arg1548_1, arg1549_1, arg568_1, arg569_1, 56448, grid=grid(56448), stream=stream0)
        del arg1548_1
        del arg1549_1
        del arg568_1
        del arg569_1
        # Source Nodes: [x_685, x_687, x_689], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf383 = extern_kernels.convolution(buf382, arg570_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg570_1
        del buf382
        buf384 = buf380; del buf380  # reuse
        # Source Nodes: [shortcut_76, x_690, x_691], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf384, buf383, arg1551_1, arg1552_1, arg571_1, arg572_1, 56448, grid=grid(56448), stream=stream0)
        del arg1551_1
        del arg1552_1
        del arg571_1
        del arg572_1
        del buf383
        # Source Nodes: [x_693], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, arg573_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg573_1
        buf386 = buf385; del buf385  # reuse
        # Source Nodes: [x_694, x_696, x_698], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf386, arg1554_1, arg1555_1, arg574_1, arg575_1, 56448, grid=grid(56448), stream=stream0)
        del arg1554_1
        del arg1555_1
        del arg574_1
        del arg575_1
        # Source Nodes: [x_694, x_696, x_698], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf387 = extern_kernels.convolution(buf386, arg576_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg576_1
        del buf386
        buf388 = buf384; del buf384  # reuse
        # Source Nodes: [x_699, x_700, x_701], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf388, buf387, arg1557_1, arg1558_1, arg577_1, arg578_1, 56448, grid=grid(56448), stream=stream0)
        del arg1557_1
        del arg1558_1
        del arg577_1
        del arg578_1
        del buf387
        # Source Nodes: [l__mod___stage4_0_fuse_layers_0_3_0], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, arg585_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 18, 7, 7), (882, 49, 7, 1))
        del arg585_1
        buf325 = buf321; del buf321  # reuse
        buf370 = buf263; del buf263  # reuse
        buf390 = buf370; del buf370  # reuse
        # Source Nodes: [l__mod___stage4_0_fuse_layers_0_1_1, l__mod___stage4_0_fuse_layers_0_1_2, l__mod___stage4_0_fuse_layers_0_2_1, l__mod___stage4_0_fuse_layers_0_2_2, l__mod___stage4_0_fuse_layers_0_3_1, l__mod___stage4_0_fuse_layers_0_3_2, shortcut_77, x_591, x_592, x_593, y_41, y_42, y_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_19.run(buf325, buf390, buf324, arg1485_1, arg1486_1, arg505_1, arg506_1, buf346, arg1560_1, arg1561_1, arg580_1, arg581_1, buf369, arg1563_1, arg1564_1, arg583_1, arg584_1, buf389, arg1566_1, arg1567_1, arg586_1, arg587_1, 451584, grid=grid(451584), stream=stream0)
        del arg1485_1
        del arg1486_1
        del arg1560_1
        del arg1561_1
        del arg1563_1
        del arg1564_1
        del arg1566_1
        del arg1567_1
        del arg505_1
        del arg506_1
        del arg580_1
        del arg581_1
        del arg583_1
        del arg584_1
        del arg586_1
        del arg587_1
        del buf324
        del buf346
        del buf369
        del buf389
        # Source Nodes: [x_702], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, arg627_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg627_1
        buf392 = buf391; del buf391  # reuse
        # Source Nodes: [x_703, x_705, x_707], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf392, arg1608_1, arg1609_1, arg628_1, arg629_1, 451584, grid=grid(451584), stream=stream0)
        del arg1608_1
        del arg1609_1
        del arg628_1
        del arg629_1
        # Source Nodes: [x_703, x_705, x_707], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf393 = extern_kernels.convolution(buf392, arg630_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg630_1
        del buf392
        buf394 = buf390; del buf390  # reuse
        # Source Nodes: [shortcut_78, x_708, x_709], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf394, buf393, arg1611_1, arg1612_1, arg631_1, arg632_1, 451584, grid=grid(451584), stream=stream0)
        del arg1611_1
        del arg1612_1
        del arg631_1
        del arg632_1
        del buf393
        # Source Nodes: [x_711], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, arg633_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg633_1
        buf396 = buf395; del buf395  # reuse
        # Source Nodes: [x_712, x_714, x_716], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf396, arg1614_1, arg1615_1, arg634_1, arg635_1, 451584, grid=grid(451584), stream=stream0)
        del arg1614_1
        del arg1615_1
        del arg634_1
        del arg635_1
        # Source Nodes: [x_712, x_714, x_716], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf397 = extern_kernels.convolution(buf396, arg636_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg636_1
        del buf396
        buf398 = buf394; del buf394  # reuse
        # Source Nodes: [shortcut_79, x_717, x_718], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf398, buf397, arg1617_1, arg1618_1, arg637_1, arg638_1, 451584, grid=grid(451584), stream=stream0)
        del arg1617_1
        del arg1618_1
        del arg637_1
        del arg638_1
        del buf397
        # Source Nodes: [x_720], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, arg639_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg639_1
        buf400 = buf399; del buf399  # reuse
        # Source Nodes: [x_721, x_723, x_725], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf400, arg1620_1, arg1621_1, arg640_1, arg641_1, 451584, grid=grid(451584), stream=stream0)
        del arg1620_1
        del arg1621_1
        del arg640_1
        del arg641_1
        # Source Nodes: [x_721, x_723, x_725], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf401 = extern_kernels.convolution(buf400, arg642_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg642_1
        del buf400
        buf402 = buf398; del buf398  # reuse
        # Source Nodes: [shortcut_80, x_726, x_727], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf402, buf401, arg1623_1, arg1624_1, arg643_1, arg644_1, 451584, grid=grid(451584), stream=stream0)
        del arg1623_1
        del arg1624_1
        del arg643_1
        del arg644_1
        del buf401
        # Source Nodes: [x_729], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, arg645_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg645_1
        buf404 = buf403; del buf403  # reuse
        # Source Nodes: [x_730, x_732, x_734], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf404, arg1626_1, arg1627_1, arg646_1, arg647_1, 451584, grid=grid(451584), stream=stream0)
        del arg1626_1
        del arg1627_1
        del arg646_1
        del arg647_1
        # Source Nodes: [x_730, x_732, x_734], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf405 = extern_kernels.convolution(buf404, arg648_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg648_1
        del buf404
        # Source Nodes: [l__mod___stage4_0_fuse_layers_1_0_0_0], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf325, arg588_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg588_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf368, arg591_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg591_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_1_3_0], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf388, arg594_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (8, 36, 7, 7), (1764, 49, 7, 1))
        del arg594_1
        buf409 = buf407; del buf407  # reuse
        buf411 = buf409; del buf409  # reuse
        # Source Nodes: [l__mod___stage4_0_fuse_layers_1_2_1, l__mod___stage4_0_fuse_layers_1_2_2, l__mod___stage4_0_fuse_layers_1_3_1, l__mod___stage4_0_fuse_layers_1_3_2, shortcut_81, y_44, y_45, y_46, y_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_20.run(buf411, arg1569_1, arg1570_1, arg589_1, arg590_1, buf345, buf408, arg1572_1, arg1573_1, arg592_1, arg593_1, buf410, arg1575_1, arg1576_1, arg595_1, arg596_1, 225792, grid=grid(225792), stream=stream0)
        del arg1569_1
        del arg1570_1
        del arg1572_1
        del arg1573_1
        del arg1575_1
        del arg1576_1
        del arg589_1
        del arg590_1
        del arg592_1
        del arg593_1
        del arg595_1
        del arg596_1
        del buf408
        del buf410
        # Source Nodes: [x_738], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, arg651_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg651_1
        buf413 = buf412; del buf412  # reuse
        # Source Nodes: [x_739, x_741, x_743], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf413, arg1632_1, arg1633_1, arg652_1, arg653_1, 225792, grid=grid(225792), stream=stream0)
        del arg1632_1
        del arg1633_1
        del arg652_1
        del arg653_1
        # Source Nodes: [x_739, x_741, x_743], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf414 = extern_kernels.convolution(buf413, arg654_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg654_1
        del buf413
        buf415 = buf411; del buf411  # reuse
        # Source Nodes: [shortcut_82, x_744, x_745], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf415, buf414, arg1635_1, arg1636_1, arg655_1, arg656_1, 225792, grid=grid(225792), stream=stream0)
        del arg1635_1
        del arg1636_1
        del arg655_1
        del arg656_1
        del buf414
        # Source Nodes: [x_747], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, arg657_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg657_1
        buf417 = buf416; del buf416  # reuse
        # Source Nodes: [x_748, x_750, x_752], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf417, arg1638_1, arg1639_1, arg658_1, arg659_1, 225792, grid=grid(225792), stream=stream0)
        del arg1638_1
        del arg1639_1
        del arg658_1
        del arg659_1
        # Source Nodes: [x_748, x_750, x_752], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf418 = extern_kernels.convolution(buf417, arg660_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg660_1
        del buf417
        buf419 = buf415; del buf415  # reuse
        # Source Nodes: [shortcut_83, x_753, x_754], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf419, buf418, arg1641_1, arg1642_1, arg661_1, arg662_1, 225792, grid=grid(225792), stream=stream0)
        del arg1641_1
        del arg1642_1
        del arg661_1
        del arg662_1
        del buf418
        # Source Nodes: [x_756], Original ATen: [aten.convolution]
        buf420 = extern_kernels.convolution(buf419, arg663_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg663_1
        buf421 = buf420; del buf420  # reuse
        # Source Nodes: [x_757, x_759, x_761], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf421, arg1644_1, arg1645_1, arg664_1, arg665_1, 225792, grid=grid(225792), stream=stream0)
        del arg1644_1
        del arg1645_1
        del arg664_1
        del arg665_1
        # Source Nodes: [x_757, x_759, x_761], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf422 = extern_kernels.convolution(buf421, arg666_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg666_1
        del buf421
        buf423 = buf419; del buf419  # reuse
        # Source Nodes: [shortcut_84, x_762, x_763], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf423, buf422, arg1647_1, arg1648_1, arg667_1, arg668_1, 225792, grid=grid(225792), stream=stream0)
        del arg1647_1
        del arg1648_1
        del arg667_1
        del arg668_1
        del buf422
        # Source Nodes: [x_765], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, arg669_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg669_1
        buf425 = buf424; del buf424  # reuse
        # Source Nodes: [x_766, x_768, x_770], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf425, arg1650_1, arg1651_1, arg670_1, arg671_1, 225792, grid=grid(225792), stream=stream0)
        del arg1650_1
        del arg1651_1
        del arg670_1
        del arg671_1
        # Source Nodes: [x_766, x_768, x_770], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf426 = extern_kernels.convolution(buf425, arg672_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg672_1
        del buf425
        buf427 = buf423; del buf423  # reuse
        # Source Nodes: [x_771, x_772, x_773], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf427, buf426, arg1653_1, arg1654_1, arg673_1, arg674_1, 225792, grid=grid(225792), stream=stream0)
        del arg1653_1
        del arg1654_1
        del arg673_1
        del arg674_1
        del buf426
        # Source Nodes: [l__mod___stage4_1_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, arg723_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg723_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_2_0_0_0], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf325, arg597_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg597_1
        buf430 = buf429; del buf429  # reuse
        # Source Nodes: [l__mod___stage4_0_fuse_layers_2_0_0_1, l__mod___stage4_0_fuse_layers_2_0_0_2, l__mod___stage4_0_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf430, arg1578_1, arg1579_1, arg598_1, arg599_1, 112896, grid=grid(112896), stream=stream0)
        del arg1578_1
        del arg1579_1
        del arg598_1
        del arg599_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_2_0_0_1, l__mod___stage4_0_fuse_layers_2_0_0_2, l__mod___stage4_0_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf431 = extern_kernels.convolution(buf430, arg600_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg600_1
        del buf430
        # Source Nodes: [l__mod___stage4_0_fuse_layers_2_1_0_0], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf345, arg603_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg603_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_2_3_0], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf388, arg606_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (8, 72, 7, 7), (3528, 49, 7, 1))
        del arg606_1
        buf433 = buf431; del buf431  # reuse
        buf435 = buf433; del buf433  # reuse
        # Source Nodes: [l__mod___stage4_0_fuse_layers_2_1_0_1, l__mod___stage4_0_fuse_layers_2_3_1, l__mod___stage4_0_fuse_layers_2_3_2, shortcut_85, y_48, y_49, y_50, y_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_21.run(buf435, arg1581_1, arg1582_1, arg601_1, arg602_1, buf432, arg1584_1, arg1585_1, arg604_1, arg605_1, buf368, buf434, arg1587_1, arg1588_1, arg607_1, arg608_1, 112896, grid=grid(112896), stream=stream0)
        del arg1581_1
        del arg1582_1
        del arg1584_1
        del arg1585_1
        del arg1587_1
        del arg1588_1
        del arg601_1
        del arg602_1
        del arg604_1
        del arg605_1
        del arg607_1
        del arg608_1
        del buf432
        del buf434
        # Source Nodes: [x_774], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, arg675_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg675_1
        buf437 = buf436; del buf436  # reuse
        # Source Nodes: [x_775, x_777, x_779], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf437, arg1656_1, arg1657_1, arg676_1, arg677_1, 112896, grid=grid(112896), stream=stream0)
        del arg1656_1
        del arg1657_1
        del arg676_1
        del arg677_1
        # Source Nodes: [x_775, x_777, x_779], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf438 = extern_kernels.convolution(buf437, arg678_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg678_1
        del buf437
        buf439 = buf435; del buf435  # reuse
        # Source Nodes: [shortcut_86, x_780, x_781], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf439, buf438, arg1659_1, arg1660_1, arg679_1, arg680_1, 112896, grid=grid(112896), stream=stream0)
        del arg1659_1
        del arg1660_1
        del arg679_1
        del arg680_1
        del buf438
        # Source Nodes: [x_783], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf439, arg681_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg681_1
        buf441 = buf440; del buf440  # reuse
        # Source Nodes: [x_784, x_786, x_788], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf441, arg1662_1, arg1663_1, arg682_1, arg683_1, 112896, grid=grid(112896), stream=stream0)
        del arg1662_1
        del arg1663_1
        del arg682_1
        del arg683_1
        # Source Nodes: [x_784, x_786, x_788], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf442 = extern_kernels.convolution(buf441, arg684_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg684_1
        del buf441
        buf443 = buf439; del buf439  # reuse
        # Source Nodes: [shortcut_87, x_789, x_790], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf443, buf442, arg1665_1, arg1666_1, arg685_1, arg686_1, 112896, grid=grid(112896), stream=stream0)
        del arg1665_1
        del arg1666_1
        del arg685_1
        del arg686_1
        del buf442
        # Source Nodes: [x_792], Original ATen: [aten.convolution]
        buf444 = extern_kernels.convolution(buf443, arg687_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg687_1
        buf445 = buf444; del buf444  # reuse
        # Source Nodes: [x_793, x_795, x_797], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf445, arg1668_1, arg1669_1, arg688_1, arg689_1, 112896, grid=grid(112896), stream=stream0)
        del arg1668_1
        del arg1669_1
        del arg688_1
        del arg689_1
        # Source Nodes: [x_793, x_795, x_797], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf446 = extern_kernels.convolution(buf445, arg690_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg690_1
        del buf445
        buf447 = buf443; del buf443  # reuse
        # Source Nodes: [shortcut_88, x_798, x_799], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf447, buf446, arg1671_1, arg1672_1, arg691_1, arg692_1, 112896, grid=grid(112896), stream=stream0)
        del arg1671_1
        del arg1672_1
        del arg691_1
        del arg692_1
        del buf446
        # Source Nodes: [x_801], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, arg693_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg693_1
        buf449 = buf448; del buf448  # reuse
        # Source Nodes: [x_802, x_804, x_806], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf449, arg1674_1, arg1675_1, arg694_1, arg695_1, 112896, grid=grid(112896), stream=stream0)
        del arg1674_1
        del arg1675_1
        del arg694_1
        del arg695_1
        # Source Nodes: [x_802, x_804, x_806], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf450 = extern_kernels.convolution(buf449, arg696_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg696_1
        del buf449
        buf451 = buf447; del buf447  # reuse
        # Source Nodes: [x_807, x_808, x_809], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf451, buf450, arg1677_1, arg1678_1, arg697_1, arg698_1, 112896, grid=grid(112896), stream=stream0)
        del arg1677_1
        del arg1678_1
        del arg697_1
        del arg698_1
        del buf450
        # Source Nodes: [l__mod___stage4_1_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf451, arg726_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf452, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg726_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_0_0_0], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf325, arg609_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg609_1
        buf455 = buf454; del buf454  # reuse
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_0_0_1, l__mod___stage4_0_fuse_layers_3_0_0_2, l__mod___stage4_0_fuse_layers_3_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf455, arg1590_1, arg1591_1, arg610_1, arg611_1, 112896, grid=grid(112896), stream=stream0)
        del arg1590_1
        del arg1591_1
        del arg610_1
        del arg611_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_0_0_1, l__mod___stage4_0_fuse_layers_3_0_0_2, l__mod___stage4_0_fuse_layers_3_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf456 = extern_kernels.convolution(buf455, arg612_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg612_1
        del buf455
        buf457 = buf456; del buf456  # reuse
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_0_1_1, l__mod___stage4_0_fuse_layers_3_0_1_2, l__mod___stage4_0_fuse_layers_3_0_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf457, arg1593_1, arg1594_1, arg613_1, arg614_1, 28224, grid=grid(28224), stream=stream0)
        del arg1593_1
        del arg1594_1
        del arg613_1
        del arg614_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_0_1_1, l__mod___stage4_0_fuse_layers_3_0_1_2, l__mod___stage4_0_fuse_layers_3_0_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf458 = extern_kernels.convolution(buf457, arg615_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf458, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg615_1
        del buf457
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_1_0_0], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf345, arg618_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg618_1
        del buf345
        buf460 = buf459; del buf459  # reuse
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_1_0_1, l__mod___stage4_0_fuse_layers_3_1_0_2, l__mod___stage4_0_fuse_layers_3_1_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf460, arg1599_1, arg1600_1, arg619_1, arg620_1, 56448, grid=grid(56448), stream=stream0)
        del arg1599_1
        del arg1600_1
        del arg619_1
        del arg620_1
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_1_0_1, l__mod___stage4_0_fuse_layers_3_1_0_2, l__mod___stage4_0_fuse_layers_3_1_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf461 = extern_kernels.convolution(buf460, arg621_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg621_1
        del buf460
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_2_0_0], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf368, arg624_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg624_1
        del buf368
        buf462 = buf458; del buf458  # reuse
        buf464 = buf388; del buf388  # reuse
        # Source Nodes: [l__mod___stage4_0_fuse_layers_3_1_1_1, l__mod___stage4_0_fuse_layers_3_2_0_1, shortcut_89, y_52, y_53, y_54, y_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf462, buf464, arg1596_1, arg1597_1, arg616_1, arg617_1, buf461, arg1602_1, arg1603_1, arg622_1, arg623_1, buf463, arg1605_1, arg1606_1, arg625_1, arg626_1, 56448, grid=grid(56448), stream=stream0)
        del arg1596_1
        del arg1597_1
        del arg1602_1
        del arg1603_1
        del arg1605_1
        del arg1606_1
        del arg616_1
        del arg617_1
        del arg622_1
        del arg623_1
        del arg625_1
        del arg626_1
        del buf461
        del buf462
        del buf463
        # Source Nodes: [x_810], Original ATen: [aten.convolution]
        buf465 = extern_kernels.convolution(buf464, arg699_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg699_1
        buf466 = buf465; del buf465  # reuse
        # Source Nodes: [x_811, x_813, x_815], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf466, arg1680_1, arg1681_1, arg700_1, arg701_1, 56448, grid=grid(56448), stream=stream0)
        del arg1680_1
        del arg1681_1
        del arg700_1
        del arg701_1
        # Source Nodes: [x_811, x_813, x_815], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf467 = extern_kernels.convolution(buf466, arg702_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg702_1
        del buf466
        buf468 = buf464; del buf464  # reuse
        # Source Nodes: [shortcut_90, x_816, x_817], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf468, buf467, arg1683_1, arg1684_1, arg703_1, arg704_1, 56448, grid=grid(56448), stream=stream0)
        del arg1683_1
        del arg1684_1
        del arg703_1
        del arg704_1
        del buf467
        # Source Nodes: [x_819], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf468, arg705_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg705_1
        buf470 = buf469; del buf469  # reuse
        # Source Nodes: [x_820, x_822, x_824], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf470, arg1686_1, arg1687_1, arg706_1, arg707_1, 56448, grid=grid(56448), stream=stream0)
        del arg1686_1
        del arg1687_1
        del arg706_1
        del arg707_1
        # Source Nodes: [x_820, x_822, x_824], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf471 = extern_kernels.convolution(buf470, arg708_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg708_1
        del buf470
        buf472 = buf468; del buf468  # reuse
        # Source Nodes: [shortcut_91, x_825, x_826], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf472, buf471, arg1689_1, arg1690_1, arg709_1, arg710_1, 56448, grid=grid(56448), stream=stream0)
        del arg1689_1
        del arg1690_1
        del arg709_1
        del arg710_1
        del buf471
        # Source Nodes: [x_828], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, arg711_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg711_1
        buf474 = buf473; del buf473  # reuse
        # Source Nodes: [x_829, x_831, x_833], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf474, arg1692_1, arg1693_1, arg712_1, arg713_1, 56448, grid=grid(56448), stream=stream0)
        del arg1692_1
        del arg1693_1
        del arg712_1
        del arg713_1
        # Source Nodes: [x_829, x_831, x_833], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf475 = extern_kernels.convolution(buf474, arg714_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg714_1
        del buf474
        buf476 = buf472; del buf472  # reuse
        # Source Nodes: [shortcut_92, x_834, x_835], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf476, buf475, arg1695_1, arg1696_1, arg715_1, arg716_1, 56448, grid=grid(56448), stream=stream0)
        del arg1695_1
        del arg1696_1
        del arg715_1
        del arg716_1
        del buf475
        # Source Nodes: [x_837], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, arg717_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg717_1
        buf478 = buf477; del buf477  # reuse
        # Source Nodes: [x_838, x_840, x_842], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf478, arg1698_1, arg1699_1, arg718_1, arg719_1, 56448, grid=grid(56448), stream=stream0)
        del arg1698_1
        del arg1699_1
        del arg718_1
        del arg719_1
        # Source Nodes: [x_838, x_840, x_842], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf479 = extern_kernels.convolution(buf478, arg720_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg720_1
        del buf478
        buf480 = buf476; del buf476  # reuse
        # Source Nodes: [x_843, x_844, x_845], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf480, buf479, arg1701_1, arg1702_1, arg721_1, arg722_1, 56448, grid=grid(56448), stream=stream0)
        del arg1701_1
        del arg1702_1
        del arg721_1
        del arg722_1
        del buf479
        # Source Nodes: [l__mod___stage4_1_fuse_layers_0_3_0], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, arg729_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (8, 18, 7, 7), (882, 49, 7, 1))
        del arg729_1
        buf406 = buf402; del buf402  # reuse
        buf453 = buf325; del buf325  # reuse
        buf482 = buf453; del buf453  # reuse
        # Source Nodes: [l__mod___stage4_1_fuse_layers_0_1_1, l__mod___stage4_1_fuse_layers_0_1_2, l__mod___stage4_1_fuse_layers_0_2_1, l__mod___stage4_1_fuse_layers_0_2_2, l__mod___stage4_1_fuse_layers_0_3_1, l__mod___stage4_1_fuse_layers_0_3_2, shortcut_93, x_735, x_736, x_737, y_57, y_58, y_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_19.run(buf406, buf482, buf405, arg1629_1, arg1630_1, arg649_1, arg650_1, buf428, arg1704_1, arg1705_1, arg724_1, arg725_1, buf452, arg1707_1, arg1708_1, arg727_1, arg728_1, buf481, arg1710_1, arg1711_1, arg730_1, arg731_1, 451584, grid=grid(451584), stream=stream0)
        del arg1629_1
        del arg1630_1
        del arg1704_1
        del arg1705_1
        del arg1707_1
        del arg1708_1
        del arg1710_1
        del arg1711_1
        del arg649_1
        del arg650_1
        del arg724_1
        del arg725_1
        del arg727_1
        del arg728_1
        del arg730_1
        del arg731_1
        del buf405
        del buf428
        del buf452
        del buf481
        # Source Nodes: [x_846], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, arg771_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg771_1
        buf484 = buf483; del buf483  # reuse
        # Source Nodes: [x_847, x_849, x_851], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf484, arg1752_1, arg1753_1, arg772_1, arg773_1, 451584, grid=grid(451584), stream=stream0)
        del arg1752_1
        del arg1753_1
        del arg772_1
        del arg773_1
        # Source Nodes: [x_847, x_849, x_851], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf485 = extern_kernels.convolution(buf484, arg774_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf485, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg774_1
        del buf484
        buf486 = buf482; del buf482  # reuse
        # Source Nodes: [shortcut_94, x_852, x_853], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf486, buf485, arg1755_1, arg1756_1, arg775_1, arg776_1, 451584, grid=grid(451584), stream=stream0)
        del arg1755_1
        del arg1756_1
        del arg775_1
        del arg776_1
        del buf485
        # Source Nodes: [x_855], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf486, arg777_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg777_1
        buf488 = buf487; del buf487  # reuse
        # Source Nodes: [x_856, x_858, x_860], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf488, arg1758_1, arg1759_1, arg778_1, arg779_1, 451584, grid=grid(451584), stream=stream0)
        del arg1758_1
        del arg1759_1
        del arg778_1
        del arg779_1
        # Source Nodes: [x_856, x_858, x_860], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf489 = extern_kernels.convolution(buf488, arg780_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf489, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg780_1
        del buf488
        buf490 = buf486; del buf486  # reuse
        # Source Nodes: [shortcut_95, x_861, x_862], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf490, buf489, arg1761_1, arg1762_1, arg781_1, arg782_1, 451584, grid=grid(451584), stream=stream0)
        del arg1761_1
        del arg1762_1
        del arg781_1
        del arg782_1
        del buf489
        # Source Nodes: [x_864], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, arg783_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg783_1
        buf492 = buf491; del buf491  # reuse
        # Source Nodes: [x_865, x_867, x_869], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf492, arg1764_1, arg1765_1, arg784_1, arg785_1, 451584, grid=grid(451584), stream=stream0)
        del arg1764_1
        del arg1765_1
        del arg784_1
        del arg785_1
        # Source Nodes: [x_865, x_867, x_869], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf493 = extern_kernels.convolution(buf492, arg786_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf493, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg786_1
        del buf492
        buf494 = buf490; del buf490  # reuse
        # Source Nodes: [shortcut_96, x_870, x_871], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf494, buf493, arg1767_1, arg1768_1, arg787_1, arg788_1, 451584, grid=grid(451584), stream=stream0)
        del arg1767_1
        del arg1768_1
        del arg787_1
        del arg788_1
        del buf493
        # Source Nodes: [x_873], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, arg789_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg789_1
        buf496 = buf495; del buf495  # reuse
        # Source Nodes: [x_874, x_876, x_878], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf496, arg1770_1, arg1771_1, arg790_1, arg791_1, 451584, grid=grid(451584), stream=stream0)
        del arg1770_1
        del arg1771_1
        del arg790_1
        del arg791_1
        # Source Nodes: [x_874, x_876, x_878], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf497 = extern_kernels.convolution(buf496, arg792_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 18, 56, 56), (56448, 3136, 56, 1))
        del arg792_1
        del buf496
        # Source Nodes: [l__mod___stage4_1_fuse_layers_1_0_0_0], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf406, arg732_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg732_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf451, arg735_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg735_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_1_3_0], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf480, arg738_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (8, 36, 7, 7), (1764, 49, 7, 1))
        del arg738_1
        buf506 = buf504; del buf504  # reuse
        buf508 = buf506; del buf506  # reuse
        # Source Nodes: [l__mod___stage4_1_fuse_layers_1_2_1, l__mod___stage4_1_fuse_layers_1_2_2, l__mod___stage4_1_fuse_layers_1_3_1, l__mod___stage4_1_fuse_layers_1_3_2, shortcut_97, y_60, y_61, y_62, y_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_20.run(buf508, arg1713_1, arg1714_1, arg733_1, arg734_1, buf427, buf505, arg1716_1, arg1717_1, arg736_1, arg737_1, buf507, arg1719_1, arg1720_1, arg739_1, arg740_1, 225792, grid=grid(225792), stream=stream0)
        del arg1713_1
        del arg1714_1
        del arg1716_1
        del arg1717_1
        del arg1719_1
        del arg1720_1
        del arg733_1
        del arg734_1
        del arg736_1
        del arg737_1
        del arg739_1
        del arg740_1
        del buf505
        del buf507
        # Source Nodes: [x_882], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, arg795_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg795_1
        buf510 = buf509; del buf509  # reuse
        # Source Nodes: [x_883, x_885, x_887], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf510, arg1776_1, arg1777_1, arg796_1, arg797_1, 225792, grid=grid(225792), stream=stream0)
        del arg1776_1
        del arg1777_1
        del arg796_1
        del arg797_1
        # Source Nodes: [x_883, x_885, x_887], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf511 = extern_kernels.convolution(buf510, arg798_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg798_1
        del buf510
        buf512 = buf508; del buf508  # reuse
        # Source Nodes: [shortcut_98, x_888, x_889], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf512, buf511, arg1779_1, arg1780_1, arg799_1, arg800_1, 225792, grid=grid(225792), stream=stream0)
        del arg1779_1
        del arg1780_1
        del arg799_1
        del arg800_1
        del buf511
        # Source Nodes: [x_891], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, arg801_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg801_1
        buf514 = buf513; del buf513  # reuse
        # Source Nodes: [x_892, x_894, x_896], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf514, arg1782_1, arg1783_1, arg802_1, arg803_1, 225792, grid=grid(225792), stream=stream0)
        del arg1782_1
        del arg1783_1
        del arg802_1
        del arg803_1
        # Source Nodes: [x_892, x_894, x_896], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf515 = extern_kernels.convolution(buf514, arg804_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg804_1
        del buf514
        buf516 = buf512; del buf512  # reuse
        # Source Nodes: [shortcut_99, x_897, x_898], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf516, buf515, arg1785_1, arg1786_1, arg805_1, arg806_1, 225792, grid=grid(225792), stream=stream0)
        del arg1785_1
        del arg1786_1
        del arg805_1
        del arg806_1
        del buf515
        # Source Nodes: [x_900], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, arg807_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg807_1
        buf518 = buf517; del buf517  # reuse
        # Source Nodes: [x_901, x_903, x_905], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf518, arg1788_1, arg1789_1, arg808_1, arg809_1, 225792, grid=grid(225792), stream=stream0)
        del arg1788_1
        del arg1789_1
        del arg808_1
        del arg809_1
        # Source Nodes: [x_901, x_903, x_905], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf519 = extern_kernels.convolution(buf518, arg810_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg810_1
        del buf518
        buf520 = buf516; del buf516  # reuse
        # Source Nodes: [shortcut_100, x_906, x_907], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf520, buf519, arg1791_1, arg1792_1, arg811_1, arg812_1, 225792, grid=grid(225792), stream=stream0)
        del arg1791_1
        del arg1792_1
        del arg811_1
        del arg812_1
        del buf519
        # Source Nodes: [x_909], Original ATen: [aten.convolution]
        buf521 = extern_kernels.convolution(buf520, arg813_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf521, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg813_1
        buf522 = buf521; del buf521  # reuse
        # Source Nodes: [x_910, x_912, x_914], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf522, arg1794_1, arg1795_1, arg814_1, arg815_1, 225792, grid=grid(225792), stream=stream0)
        del arg1794_1
        del arg1795_1
        del arg814_1
        del arg815_1
        # Source Nodes: [x_910, x_912, x_914], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf523 = extern_kernels.convolution(buf522, arg816_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg816_1
        del buf522
        buf524 = buf520; del buf520  # reuse
        # Source Nodes: [x_915, x_916, x_917], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf524, buf523, arg1797_1, arg1798_1, arg817_1, arg818_1, 225792, grid=grid(225792), stream=stream0)
        del arg1797_1
        del arg1798_1
        del arg817_1
        del arg818_1
        del buf523
        # Source Nodes: [l__mod___stage4_2_fuse_layers_0_1_0], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf524, arg867_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg867_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_2_0_0_0], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf406, arg741_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg741_1
        buf530 = buf529; del buf529  # reuse
        # Source Nodes: [l__mod___stage4_1_fuse_layers_2_0_0_1, l__mod___stage4_1_fuse_layers_2_0_0_2, l__mod___stage4_1_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf530, arg1722_1, arg1723_1, arg742_1, arg743_1, 112896, grid=grid(112896), stream=stream0)
        del arg1722_1
        del arg1723_1
        del arg742_1
        del arg743_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_2_0_0_1, l__mod___stage4_1_fuse_layers_2_0_0_2, l__mod___stage4_1_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf531 = extern_kernels.convolution(buf530, arg744_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf531, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg744_1
        del buf530
        # Source Nodes: [l__mod___stage4_1_fuse_layers_2_1_0_0], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf427, arg747_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg747_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_2_3_0], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf480, arg750_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (8, 72, 7, 7), (3528, 49, 7, 1))
        del arg750_1
        buf533 = buf531; del buf531  # reuse
        buf535 = buf533; del buf533  # reuse
        # Source Nodes: [l__mod___stage4_1_fuse_layers_2_1_0_1, l__mod___stage4_1_fuse_layers_2_3_1, l__mod___stage4_1_fuse_layers_2_3_2, shortcut_101, y_64, y_65, y_66, y_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_21.run(buf535, arg1725_1, arg1726_1, arg745_1, arg746_1, buf532, arg1728_1, arg1729_1, arg748_1, arg749_1, buf451, buf534, arg1731_1, arg1732_1, arg751_1, arg752_1, 112896, grid=grid(112896), stream=stream0)
        del arg1725_1
        del arg1726_1
        del arg1728_1
        del arg1729_1
        del arg1731_1
        del arg1732_1
        del arg745_1
        del arg746_1
        del arg748_1
        del arg749_1
        del arg751_1
        del arg752_1
        del buf532
        del buf534
        # Source Nodes: [x_918], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, arg819_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg819_1
        buf537 = buf536; del buf536  # reuse
        # Source Nodes: [x_919, x_921, x_923], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf537, arg1800_1, arg1801_1, arg820_1, arg821_1, 112896, grid=grid(112896), stream=stream0)
        del arg1800_1
        del arg1801_1
        del arg820_1
        del arg821_1
        # Source Nodes: [x_919, x_921, x_923], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf538 = extern_kernels.convolution(buf537, arg822_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg822_1
        del buf537
        buf539 = buf535; del buf535  # reuse
        # Source Nodes: [shortcut_102, x_924, x_925], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf539, buf538, arg1803_1, arg1804_1, arg823_1, arg824_1, 112896, grid=grid(112896), stream=stream0)
        del arg1803_1
        del arg1804_1
        del arg823_1
        del arg824_1
        del buf538
        # Source Nodes: [x_927], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, arg825_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg825_1
        buf541 = buf540; del buf540  # reuse
        # Source Nodes: [x_928, x_930, x_932], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf541, arg1806_1, arg1807_1, arg826_1, arg827_1, 112896, grid=grid(112896), stream=stream0)
        del arg1806_1
        del arg1807_1
        del arg826_1
        del arg827_1
        # Source Nodes: [x_928, x_930, x_932], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf542 = extern_kernels.convolution(buf541, arg828_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg828_1
        del buf541
        buf543 = buf539; del buf539  # reuse
        # Source Nodes: [shortcut_103, x_933, x_934], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf543, buf542, arg1809_1, arg1810_1, arg829_1, arg830_1, 112896, grid=grid(112896), stream=stream0)
        del arg1809_1
        del arg1810_1
        del arg829_1
        del arg830_1
        del buf542
        # Source Nodes: [x_936], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, arg831_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf544, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg831_1
        buf545 = buf544; del buf544  # reuse
        # Source Nodes: [x_937, x_939, x_941], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf545, arg1812_1, arg1813_1, arg832_1, arg833_1, 112896, grid=grid(112896), stream=stream0)
        del arg1812_1
        del arg1813_1
        del arg832_1
        del arg833_1
        # Source Nodes: [x_937, x_939, x_941], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf546 = extern_kernels.convolution(buf545, arg834_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg834_1
        del buf545
        buf547 = buf543; del buf543  # reuse
        # Source Nodes: [shortcut_104, x_942, x_943], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf547, buf546, arg1815_1, arg1816_1, arg835_1, arg836_1, 112896, grid=grid(112896), stream=stream0)
        del arg1815_1
        del arg1816_1
        del arg835_1
        del arg836_1
        del buf546
        # Source Nodes: [x_945], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf547, arg837_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg837_1
        buf549 = buf548; del buf548  # reuse
        # Source Nodes: [x_946, x_948, x_950], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf549, arg1818_1, arg1819_1, arg838_1, arg839_1, 112896, grid=grid(112896), stream=stream0)
        del arg1818_1
        del arg1819_1
        del arg838_1
        del arg839_1
        # Source Nodes: [x_946, x_948, x_950], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf550 = extern_kernels.convolution(buf549, arg840_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg840_1
        del buf549
        buf551 = buf547; del buf547  # reuse
        # Source Nodes: [x_951, x_952, x_953], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf551, buf550, arg1821_1, arg1822_1, arg841_1, arg842_1, 112896, grid=grid(112896), stream=stream0)
        del arg1821_1
        del arg1822_1
        del arg841_1
        del arg842_1
        del buf550
        # Source Nodes: [l__mod___stage4_2_fuse_layers_0_2_0], Original ATen: [aten.convolution]
        buf615 = extern_kernels.convolution(buf551, arg870_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf615, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg870_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_0_0_0], Original ATen: [aten.convolution]
        buf553 = extern_kernels.convolution(buf406, arg753_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf553, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg753_1
        buf554 = buf553; del buf553  # reuse
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_0_0_1, l__mod___stage4_1_fuse_layers_3_0_0_2, l__mod___stage4_1_fuse_layers_3_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf554, arg1734_1, arg1735_1, arg754_1, arg755_1, 112896, grid=grid(112896), stream=stream0)
        del arg1734_1
        del arg1735_1
        del arg754_1
        del arg755_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_0_0_1, l__mod___stage4_1_fuse_layers_3_0_0_2, l__mod___stage4_1_fuse_layers_3_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf555 = extern_kernels.convolution(buf554, arg756_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf555, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg756_1
        del buf554
        buf556 = buf555; del buf555  # reuse
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_0_1_1, l__mod___stage4_1_fuse_layers_3_0_1_2, l__mod___stage4_1_fuse_layers_3_0_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf556, arg1737_1, arg1738_1, arg757_1, arg758_1, 28224, grid=grid(28224), stream=stream0)
        del arg1737_1
        del arg1738_1
        del arg757_1
        del arg758_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_0_1_1, l__mod___stage4_1_fuse_layers_3_0_1_2, l__mod___stage4_1_fuse_layers_3_0_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf557 = extern_kernels.convolution(buf556, arg759_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg759_1
        del buf556
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_1_0_0], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf427, arg762_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg762_1
        del buf427
        buf559 = buf558; del buf558  # reuse
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_1_0_1, l__mod___stage4_1_fuse_layers_3_1_0_2, l__mod___stage4_1_fuse_layers_3_1_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf559, arg1743_1, arg1744_1, arg763_1, arg764_1, 56448, grid=grid(56448), stream=stream0)
        del arg1743_1
        del arg1744_1
        del arg763_1
        del arg764_1
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_1_0_1, l__mod___stage4_1_fuse_layers_3_1_0_2, l__mod___stage4_1_fuse_layers_3_1_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf560 = extern_kernels.convolution(buf559, arg765_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg765_1
        del buf559
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_2_0_0], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf451, arg768_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg768_1
        del buf451
        buf561 = buf557; del buf557  # reuse
        buf563 = buf480; del buf480  # reuse
        # Source Nodes: [l__mod___stage4_1_fuse_layers_3_1_1_1, l__mod___stage4_1_fuse_layers_3_2_0_1, shortcut_105, y_68, y_69, y_70, y_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf561, buf563, arg1740_1, arg1741_1, arg760_1, arg761_1, buf560, arg1746_1, arg1747_1, arg766_1, arg767_1, buf562, arg1749_1, arg1750_1, arg769_1, arg770_1, 56448, grid=grid(56448), stream=stream0)
        del arg1740_1
        del arg1741_1
        del arg1746_1
        del arg1747_1
        del arg1749_1
        del arg1750_1
        del arg760_1
        del arg761_1
        del arg766_1
        del arg767_1
        del arg769_1
        del arg770_1
        del buf560
        del buf561
        del buf562
        # Source Nodes: [x_954], Original ATen: [aten.convolution]
        buf564 = extern_kernels.convolution(buf563, arg843_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf564, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg843_1
        buf565 = buf564; del buf564  # reuse
        # Source Nodes: [x_955, x_957, x_959], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf565, arg1824_1, arg1825_1, arg844_1, arg845_1, 56448, grid=grid(56448), stream=stream0)
        del arg1824_1
        del arg1825_1
        del arg844_1
        del arg845_1
        # Source Nodes: [x_955, x_957, x_959], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf566 = extern_kernels.convolution(buf565, arg846_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg846_1
        del buf565
        buf567 = buf563; del buf563  # reuse
        # Source Nodes: [shortcut_106, x_960, x_961], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf567, buf566, arg1827_1, arg1828_1, arg847_1, arg848_1, 56448, grid=grid(56448), stream=stream0)
        del arg1827_1
        del arg1828_1
        del arg847_1
        del arg848_1
        del buf566
        # Source Nodes: [x_963], Original ATen: [aten.convolution]
        buf568 = extern_kernels.convolution(buf567, arg849_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf568, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg849_1
        buf569 = buf568; del buf568  # reuse
        # Source Nodes: [x_964, x_966, x_968], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf569, arg1830_1, arg1831_1, arg850_1, arg851_1, 56448, grid=grid(56448), stream=stream0)
        del arg1830_1
        del arg1831_1
        del arg850_1
        del arg851_1
        # Source Nodes: [x_964, x_966, x_968], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf570 = extern_kernels.convolution(buf569, arg852_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg852_1
        del buf569
        buf571 = buf567; del buf567  # reuse
        # Source Nodes: [shortcut_107, x_969, x_970], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf571, buf570, arg1833_1, arg1834_1, arg853_1, arg854_1, 56448, grid=grid(56448), stream=stream0)
        del arg1833_1
        del arg1834_1
        del arg853_1
        del arg854_1
        del buf570
        # Source Nodes: [x_972], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, arg855_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg855_1
        buf573 = buf572; del buf572  # reuse
        # Source Nodes: [x_973, x_975, x_977], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf573, arg1836_1, arg1837_1, arg856_1, arg857_1, 56448, grid=grid(56448), stream=stream0)
        del arg1836_1
        del arg1837_1
        del arg856_1
        del arg857_1
        # Source Nodes: [x_973, x_975, x_977], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf574 = extern_kernels.convolution(buf573, arg858_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg858_1
        del buf573
        buf575 = buf571; del buf571  # reuse
        # Source Nodes: [shortcut_108, x_978, x_979], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf575, buf574, arg1839_1, arg1840_1, arg859_1, arg860_1, 56448, grid=grid(56448), stream=stream0)
        del arg1839_1
        del arg1840_1
        del arg859_1
        del arg860_1
        del buf574
        # Source Nodes: [x_981], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, arg861_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg861_1
        buf577 = buf576; del buf576  # reuse
        # Source Nodes: [x_982, x_984, x_986], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf577, arg1842_1, arg1843_1, arg862_1, arg863_1, 56448, grid=grid(56448), stream=stream0)
        del arg1842_1
        del arg1843_1
        del arg862_1
        del arg863_1
        # Source Nodes: [x_982, x_984, x_986], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf578 = extern_kernels.convolution(buf577, arg864_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg864_1
        del buf577
        buf579 = buf575; del buf575  # reuse
        # Source Nodes: [x_987, x_988, x_989], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf579, buf578, arg1845_1, arg1846_1, arg865_1, arg866_1, 56448, grid=grid(56448), stream=stream0)
        del arg1845_1
        del arg1846_1
        del arg865_1
        del arg866_1
        del buf578
        # Source Nodes: [l__mod___stage4_2_fuse_layers_0_3_0], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf579, arg873_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (8, 18, 7, 7), (882, 49, 7, 1))
        del arg873_1
        buf498 = buf494; del buf494  # reuse
        buf616 = buf406; del buf406  # reuse
        buf618 = buf616; del buf616  # reuse
        # Source Nodes: [l__mod___stage4_2_fuse_layers_0_1_1, l__mod___stage4_2_fuse_layers_0_1_2, l__mod___stage4_2_fuse_layers_0_2_1, l__mod___stage4_2_fuse_layers_0_2_2, l__mod___stage4_2_fuse_layers_0_3_1, l__mod___stage4_2_fuse_layers_0_3_2, shortcut_109, x_879, x_880, x_881, y_73, y_74, y_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_19.run(buf498, buf618, buf497, arg1773_1, arg1774_1, arg793_1, arg794_1, buf614, arg1848_1, arg1849_1, arg868_1, arg869_1, buf615, arg1851_1, arg1852_1, arg871_1, arg872_1, buf617, arg1854_1, arg1855_1, arg874_1, arg875_1, 451584, grid=grid(451584), stream=stream0)
        del arg1773_1
        del arg1774_1
        del arg1848_1
        del arg1849_1
        del arg1851_1
        del arg1852_1
        del arg1854_1
        del arg1855_1
        del arg793_1
        del arg794_1
        del arg868_1
        del arg869_1
        del arg871_1
        del arg872_1
        del arg874_1
        del arg875_1
        del buf497
        del buf614
        del buf615
        del buf617
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_0_0_0], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, arg897_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg897_1
        buf500 = buf499; del buf499  # reuse
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_0_0_1, l__mod___stage4_2_fuse_layers_3_0_0_2, l__mod___stage4_2_fuse_layers_3_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf500, arg1878_1, arg1879_1, arg898_1, arg899_1, 112896, grid=grid(112896), stream=stream0)
        del arg1878_1
        del arg1879_1
        del arg898_1
        del arg899_1
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_0_0_1, l__mod___stage4_2_fuse_layers_3_0_0_2, l__mod___stage4_2_fuse_layers_3_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf501 = extern_kernels.convolution(buf500, arg900_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (8, 18, 14, 14), (3528, 196, 14, 1))
        del arg900_1
        del buf500
        buf502 = buf501; del buf501  # reuse
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_0_1_1, l__mod___stage4_2_fuse_layers_3_0_1_2, l__mod___stage4_2_fuse_layers_3_0_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf502, arg1881_1, arg1882_1, arg901_1, arg902_1, 28224, grid=grid(28224), stream=stream0)
        del arg1881_1
        del arg1882_1
        del arg901_1
        del arg902_1
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_0_1_1, l__mod___stage4_2_fuse_layers_3_0_1_2, l__mod___stage4_2_fuse_layers_3_0_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf503 = extern_kernels.convolution(buf502, arg903_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf503, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg903_1
        del buf502
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_1_0_0], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf524, arg906_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf525, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg906_1
        buf526 = buf525; del buf525  # reuse
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_1_0_1, l__mod___stage4_2_fuse_layers_3_1_0_2, l__mod___stage4_2_fuse_layers_3_1_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_23.run(buf526, arg1887_1, arg1888_1, arg907_1, arg908_1, 56448, grid=grid(56448), stream=stream0)
        del arg1887_1
        del arg1888_1
        del arg907_1
        del arg908_1
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_1_0_1, l__mod___stage4_2_fuse_layers_3_1_0_2, l__mod___stage4_2_fuse_layers_3_1_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf527 = extern_kernels.convolution(buf526, arg909_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg909_1
        del buf526
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_2_0_0], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, arg912_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (8, 144, 7, 7), (7056, 49, 7, 1))
        del arg912_1
        buf528 = buf503; del buf503  # reuse
        buf580 = buf528; del buf528  # reuse
        # Source Nodes: [l__mod___stage4_2_fuse_layers_3_1_1_1, l__mod___stage4_2_fuse_layers_3_2_0_1, shortcut_115, y_84, y_85, y_86, y_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf580, arg1884_1, arg1885_1, arg904_1, arg905_1, buf527, arg1890_1, arg1891_1, arg910_1, arg911_1, buf552, arg1893_1, arg1894_1, arg913_1, arg914_1, buf579, 56448, grid=grid(56448), stream=stream0)
        del arg1884_1
        del arg1885_1
        del arg1890_1
        del arg1891_1
        del arg1893_1
        del arg1894_1
        del arg904_1
        del arg905_1
        del arg910_1
        del arg911_1
        del arg913_1
        del arg914_1
        del buf527
        del buf552
        # Source Nodes: [x_1026], Original ATen: [aten.convolution]
        buf581 = extern_kernels.convolution(buf580, arg959_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf581, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg959_1
        buf582 = buf581; del buf581  # reuse
        # Source Nodes: [x_1027, x_1028, x_1029], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(buf582, arg1938_1, arg1939_1, arg960_1, arg961_1, 100352, grid=grid(100352), stream=stream0)
        del arg1938_1
        del arg1939_1
        del arg960_1
        del arg961_1
        # Source Nodes: [x_1027, x_1028, x_1029], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf583 = extern_kernels.convolution(buf582, arg962_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf583, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg962_1
        del buf582
        buf584 = buf583; del buf583  # reuse
        # Source Nodes: [x_1030, x_1032, x_1034], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(buf584, arg1941_1, arg1942_1, arg963_1, arg964_1, 100352, grid=grid(100352), stream=stream0)
        del arg1941_1
        del arg1942_1
        del arg963_1
        del arg964_1
        # Source Nodes: [x_1030, x_1032, x_1034], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf585 = extern_kernels.convolution(buf584, arg965_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf585, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg965_1
        del buf584
        # Source Nodes: [getattr_l__mod___incre_modules_3___0___downsample_0], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf580, arg968_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg968_1
        del buf580
        # Source Nodes: [l__mod___stage4_2_fuse_layers_2_0_0_0], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(buf498, arg885_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (8, 18, 28, 28), (14112, 784, 28, 1))
        del arg885_1
        buf589 = buf588; del buf588  # reuse
        # Source Nodes: [l__mod___stage4_2_fuse_layers_2_0_0_1, l__mod___stage4_2_fuse_layers_2_0_0_2, l__mod___stage4_2_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf589, arg1866_1, arg1867_1, arg886_1, arg887_1, 112896, grid=grid(112896), stream=stream0)
        del arg1866_1
        del arg1867_1
        del arg886_1
        del arg887_1
        # Source Nodes: [l__mod___stage4_2_fuse_layers_2_0_0_1, l__mod___stage4_2_fuse_layers_2_0_0_2, l__mod___stage4_2_fuse_layers_2_0_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf590 = extern_kernels.convolution(buf589, arg888_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf590, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg888_1
        del buf589
        # Source Nodes: [l__mod___stage4_2_fuse_layers_2_1_0_0], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(buf524, arg891_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (8, 72, 14, 14), (14112, 196, 14, 1))
        del arg891_1
        # Source Nodes: [l__mod___stage4_2_fuse_layers_2_3_0], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf579, arg894_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (8, 72, 7, 7), (3528, 49, 7, 1))
        del arg894_1
        buf592 = buf590; del buf590  # reuse
        buf594 = buf592; del buf592  # reuse
        # Source Nodes: [l__mod___stage4_2_fuse_layers_2_1_0_1, l__mod___stage4_2_fuse_layers_2_3_1, l__mod___stage4_2_fuse_layers_2_3_2, shortcut_113, y_80, y_81, y_82, y_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_21.run(buf594, arg1869_1, arg1870_1, arg889_1, arg890_1, buf591, arg1872_1, arg1873_1, arg892_1, arg893_1, buf551, buf593, arg1875_1, arg1876_1, arg895_1, arg896_1, 112896, grid=grid(112896), stream=stream0)
        del arg1869_1
        del arg1870_1
        del arg1872_1
        del arg1873_1
        del arg1875_1
        del arg1876_1
        del arg889_1
        del arg890_1
        del arg892_1
        del arg893_1
        del arg895_1
        del arg896_1
        del buf591
        del buf593
        # Source Nodes: [x_1014], Original ATen: [aten.convolution]
        buf595 = extern_kernels.convolution(buf594, arg943_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf595, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg943_1
        buf596 = buf595; del buf595  # reuse
        # Source Nodes: [x_1015, x_1016, x_1017], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf596, arg1923_1, arg1924_1, arg944_1, arg945_1, 200704, grid=grid(200704), stream=stream0)
        del arg1923_1
        del arg1924_1
        del arg944_1
        del arg945_1
        # Source Nodes: [x_1015, x_1016, x_1017], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf597 = extern_kernels.convolution(buf596, arg946_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf597, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg946_1
        del buf596
        buf598 = buf597; del buf597  # reuse
        # Source Nodes: [x_1018, x_1020, x_1022], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_27.run(buf598, arg1926_1, arg1927_1, arg947_1, arg948_1, 200704, grid=grid(200704), stream=stream0)
        del arg1926_1
        del arg1927_1
        del arg947_1
        del arg948_1
        # Source Nodes: [x_1018, x_1020, x_1022], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf599 = extern_kernels.convolution(buf598, arg949_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf599, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg949_1
        del buf598
        # Source Nodes: [getattr_l__mod___incre_modules_2___0___downsample_0], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf594, arg952_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg952_1
        del buf594
        # Source Nodes: [l__mod___stage4_2_fuse_layers_1_0_0_0], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf498, arg876_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (8, 36, 28, 28), (28224, 784, 28, 1))
        del arg876_1
        del buf498
        # Source Nodes: [l__mod___stage4_2_fuse_layers_1_2_0], Original ATen: [aten.convolution]
        buf603 = extern_kernels.convolution(buf551, arg879_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf603, (8, 36, 14, 14), (7056, 196, 14, 1))
        del arg879_1
        del buf551
        # Source Nodes: [l__mod___stage4_2_fuse_layers_1_3_0], Original ATen: [aten.convolution]
        buf605 = extern_kernels.convolution(buf579, arg882_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf605, (8, 36, 7, 7), (1764, 49, 7, 1))
        del arg882_1
        del buf579
        buf604 = buf524; del buf524  # reuse
        buf606 = buf604; del buf604  # reuse
        # Source Nodes: [l__mod___stage4_2_fuse_layers_1_2_1, l__mod___stage4_2_fuse_layers_1_2_2, l__mod___stage4_2_fuse_layers_1_3_1, l__mod___stage4_2_fuse_layers_1_3_2, shortcut_111, y_76, y_77, y_78, y_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten._unsafe_index, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_28.run(buf606, buf602, arg1857_1, arg1858_1, arg877_1, arg878_1, buf603, arg1860_1, arg1861_1, arg880_1, arg881_1, buf605, arg1863_1, arg1864_1, arg883_1, arg884_1, 225792, grid=grid(225792), stream=stream0)
        del arg1857_1
        del arg1858_1
        del arg1860_1
        del arg1861_1
        del arg1863_1
        del arg1864_1
        del arg877_1
        del arg878_1
        del arg880_1
        del arg881_1
        del arg883_1
        del arg884_1
        del buf602
        del buf603
        del buf605
        # Source Nodes: [x_1002], Original ATen: [aten.convolution]
        buf607 = extern_kernels.convolution(buf606, arg927_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf607, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg927_1
        buf608 = buf607; del buf607  # reuse
        # Source Nodes: [x_1003, x_1004, x_1005], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29.run(buf608, arg1908_1, arg1909_1, arg928_1, arg929_1, 401408, grid=grid(401408), stream=stream0)
        del arg1908_1
        del arg1909_1
        del arg928_1
        del arg929_1
        # Source Nodes: [x_1003, x_1004, x_1005], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf609 = extern_kernels.convolution(buf608, arg930_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf609, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg930_1
        del buf608
        buf610 = buf609; del buf609  # reuse
        # Source Nodes: [x_1006, x_1008, x_1010], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29.run(buf610, arg1911_1, arg1912_1, arg931_1, arg932_1, 401408, grid=grid(401408), stream=stream0)
        del arg1911_1
        del arg1912_1
        del arg931_1
        del arg932_1
        # Source Nodes: [x_1006, x_1008, x_1010], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf611 = extern_kernels.convolution(buf610, arg933_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf611, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg933_1
        del buf610
        # Source Nodes: [getattr_l__mod___incre_modules_1___0___downsample_0], Original ATen: [aten.convolution]
        buf612 = extern_kernels.convolution(buf606, arg936_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf612, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg936_1
        del buf606
        # Source Nodes: [x_990], Original ATen: [aten.convolution]
        buf619 = extern_kernels.convolution(buf618, arg915_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf619, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg915_1
        buf620 = buf619; del buf619  # reuse
        # Source Nodes: [x_991, x_992, x_993], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(buf620, arg1896_1, arg1897_1, arg916_1, arg917_1, 802816, grid=grid(802816), stream=stream0)
        del arg1896_1
        del arg1897_1
        del arg916_1
        del arg917_1
        # Source Nodes: [x_991, x_992, x_993], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf621 = extern_kernels.convolution(buf620, arg918_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf621, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg918_1
        del buf620
        buf622 = buf621; del buf621  # reuse
        # Source Nodes: [x_994, x_996, x_998], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(buf622, arg1899_1, arg1900_1, arg919_1, arg920_1, 802816, grid=grid(802816), stream=stream0)
        del arg1899_1
        del arg1900_1
        del arg919_1
        del arg920_1
        # Source Nodes: [x_994, x_996, x_998], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf623 = extern_kernels.convolution(buf622, arg921_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf623, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg921_1
        del buf622
        # Source Nodes: [getattr_l__mod___incre_modules_0___0___downsample_0], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf618, arg924_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg924_1
        del buf618
        buf625 = buf623; del buf623  # reuse
        buf626 = buf625; del buf625  # reuse
        # Source Nodes: [forward, shortcut_110, x_1000, x_999, y_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_31.run(buf626, arg1902_1, arg1903_1, arg922_1, arg923_1, buf624, arg1905_1, arg1906_1, arg925_1, arg926_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg1902_1
        del arg1903_1
        del arg1905_1
        del arg1906_1
        del arg922_1
        del arg923_1
        del arg925_1
        del arg926_1
        del buf624
        # Source Nodes: [forward, y_88], Original ATen: [aten.convolution, aten.relu]
        buf627 = extern_kernels.convolution(buf626, arg939_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf627, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg939_1
        del buf626
        buf613 = buf611; del buf611  # reuse
        buf628 = buf613; del buf613  # reuse
        # Source Nodes: [forward, forward_1, shortcut_112, x_1011, x_1012, x_1013, y_88, y_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_32.run(buf628, arg1914_1, arg1915_1, arg934_1, arg935_1, buf612, arg1917_1, arg1918_1, arg937_1, arg938_1, buf627, arg940_1, arg1920_1, arg1921_1, arg941_1, arg942_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg1914_1
        del arg1915_1
        del arg1917_1
        del arg1918_1
        del arg1920_1
        del arg1921_1
        del arg934_1
        del arg935_1
        del arg937_1
        del arg938_1
        del arg940_1
        del arg941_1
        del arg942_1
        del buf612
        del buf627
        # Source Nodes: [forward, forward_1, x_1013, y_88, y_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        buf629 = extern_kernels.convolution(buf628, arg955_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf629, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg955_1
        del buf628
        buf601 = buf599; del buf599  # reuse
        buf630 = buf601; del buf601  # reuse
        # Source Nodes: [forward, forward_1, forward_2, shortcut_114, x_1013, x_1023, x_1024, x_1025, y_88, y_89, y_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_33.run(buf630, arg1929_1, arg1930_1, arg950_1, arg951_1, buf600, arg1932_1, arg1933_1, arg953_1, arg954_1, buf629, arg956_1, arg1935_1, arg1936_1, arg957_1, arg958_1, 802816, grid=grid(802816), stream=stream0)
        del arg1929_1
        del arg1930_1
        del arg1932_1
        del arg1933_1
        del arg1935_1
        del arg1936_1
        del arg950_1
        del arg951_1
        del arg953_1
        del arg954_1
        del arg956_1
        del arg957_1
        del arg958_1
        del buf600
        del buf629
        # Source Nodes: [forward, forward_1, forward_2, x_1013, x_1025, y_88, y_89, y_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        buf631 = extern_kernels.convolution(buf630, arg971_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf631, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg971_1
        del buf630
        buf587 = buf585; del buf585  # reuse
        buf632 = buf587; del buf587  # reuse
        # Source Nodes: [forward, forward_1, forward_2, l__mod___final_layer_0, shortcut_116, x_1013, x_1025, x_1035, x_1036, x_1037, y_88, y_89, y_90, y_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_34.run(buf632, arg1944_1, arg1945_1, arg966_1, arg967_1, buf586, arg1947_1, arg1948_1, arg969_1, arg970_1, buf631, arg972_1, arg1950_1, arg1951_1, arg973_1, arg974_1, 401408, grid=grid(401408), stream=stream0)
        del arg1944_1
        del arg1945_1
        del arg1947_1
        del arg1948_1
        del arg1950_1
        del arg1951_1
        del arg966_1
        del arg967_1
        del arg969_1
        del arg970_1
        del arg972_1
        del arg973_1
        del arg974_1
        del buf586
        del buf631
        # Source Nodes: [forward, forward_1, forward_2, l__mod___final_layer_0, x_1013, x_1025, x_1037, y_88, y_89, y_90, y_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        buf633 = extern_kernels.convolution(buf632, arg975_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg975_1
        del buf632
        buf634 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf635 = reinterpret_tensor(buf634, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf634  # reuse
        # Source Nodes: [forward, forward_1, forward_2, l__mod___final_layer_0, l__mod___final_layer_1, x_1013, x_1025, x_1037, x_1038, y_88, y_89, y_90, y_91, y_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_35.run(buf635, buf633, arg976_1, arg1953_1, arg1954_1, arg977_1, arg978_1, 16384, 49, grid=grid(16384), stream=stream0)
        del arg1953_1
        del arg1954_1
        del arg976_1
        del arg977_1
        del arg978_1
        del buf633
        buf636 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1042], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg980_1, reinterpret_tensor(buf635, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg979_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf636)
        del arg979_1
        del arg980_1
        return (buf636, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((18, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((36, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((144, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((18, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((36, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((72, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((144, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((144, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((144, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((18, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((36, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((72, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((144, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((144, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((144, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg797_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg800_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg803_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg806_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg809_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg812_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg815_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg818_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg821_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg824_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg827_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg830_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg833_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg836_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg839_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((72, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg842_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg845_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg848_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg851_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg854_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg857_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg860_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg863_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg866_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((18, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg869_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((18, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg872_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((18, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg875_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((36, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg878_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((36, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg881_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((36, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg884_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg887_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((72, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg890_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((72, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg893_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((72, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg896_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg899_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((18, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg902_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((144, 18, 3, 3), (162, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg905_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((36, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg908_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((144, 36, 3, 3), (324, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg911_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((144, 72, 3, 3), (648, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg914_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((32, 18, 1, 1), (18, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg917_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg920_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg923_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((128, 18, 1, 1), (18, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg926_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((64, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg929_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg932_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg935_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg936_1 = rand_strided((256, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg937_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg938_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg939_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg940_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg941_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg942_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg943_1 = rand_strided((128, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg944_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg945_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg946_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg947_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg948_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg949_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg950_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg951_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg952_1 = rand_strided((512, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg953_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg954_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg955_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg956_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg957_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg958_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg959_1 = rand_strided((256, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg960_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg961_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg962_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg963_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg964_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg965_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg966_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg967_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg968_1 = rand_strided((1024, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg969_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg970_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg971_1 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg972_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg973_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg974_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg975_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg976_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg977_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg978_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg979_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg980_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg981_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg982_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg983_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg984_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg985_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg986_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg987_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg988_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg989_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg990_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg991_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg992_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg993_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg994_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg995_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg996_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg997_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg998_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg999_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1000_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1001_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1002_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1003_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1004_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1005_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1006_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1007_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1008_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1009_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1010_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1011_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1012_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1013_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1014_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1015_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1016_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1017_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1018_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1019_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1020_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1021_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1022_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1023_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1024_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1025_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1026_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1027_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1028_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1029_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1030_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1031_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1032_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1033_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1034_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1035_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1036_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1037_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1038_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1039_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1040_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1041_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1042_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1043_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1044_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1045_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1046_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1047_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1048_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1049_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1050_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1051_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1052_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1053_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1054_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1055_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1056_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1057_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1058_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1059_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1060_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1061_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1062_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1063_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1064_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1065_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1066_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1067_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1068_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1069_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1070_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1071_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1072_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1073_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1074_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1075_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1076_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1077_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1078_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1079_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1080_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1081_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1082_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1083_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1084_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1085_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1086_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1087_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1088_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1089_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1090_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1091_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1092_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1093_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1094_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1095_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1096_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1097_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1098_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1099_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1100_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1101_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1102_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1103_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1104_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1105_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1106_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1107_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1108_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1109_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1110_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1111_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1112_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1113_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1114_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1115_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1116_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1117_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1118_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1119_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1120_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1121_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1122_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1123_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1124_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1125_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1126_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1127_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1128_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1129_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1130_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1131_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1132_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1133_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1134_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1135_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1136_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1137_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1138_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1139_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1140_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1141_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1142_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1143_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1144_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1145_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1146_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1147_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1148_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1149_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1150_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1151_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1152_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1153_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1154_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1155_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1156_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1157_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1158_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1159_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1160_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1161_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1162_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1163_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1164_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1165_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1166_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1167_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1168_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1169_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1170_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1171_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1172_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1173_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1174_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1175_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1176_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1177_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1178_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1179_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1180_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1181_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1182_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1183_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1184_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1185_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1186_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1187_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1188_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1189_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1190_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1191_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1192_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1193_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1194_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1195_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1196_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1197_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1198_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1199_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1200_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1201_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1202_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1203_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1204_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1205_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1206_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1207_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1208_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1209_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1210_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1211_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1212_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1213_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1214_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1215_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1216_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1217_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1218_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1219_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1220_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1221_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1222_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1223_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1224_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1225_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1226_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1227_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1228_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1229_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1230_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1231_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1232_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1233_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1234_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1235_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1236_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1237_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1238_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1239_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1240_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1241_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1242_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1243_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1244_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1245_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1246_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1247_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1248_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1249_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1250_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1251_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1252_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1253_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1254_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1255_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1256_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1257_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1258_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1259_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1260_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1261_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1262_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1263_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1264_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1265_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1266_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1267_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1268_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1269_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1270_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1271_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1272_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1273_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1274_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1275_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1276_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1277_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1278_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1279_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1280_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1281_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1282_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1283_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1284_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1285_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1286_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1287_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1288_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1289_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1290_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1291_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1292_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1293_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1294_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1295_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1296_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1297_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1298_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1299_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1300_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1301_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1302_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1303_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1304_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1305_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1306_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1307_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1308_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1309_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1310_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1311_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1312_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1313_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1314_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1315_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1316_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1317_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1318_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1319_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1320_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1321_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1322_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1323_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1324_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1325_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1326_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1327_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1328_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1329_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1330_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1331_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1332_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1333_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1334_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1335_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1336_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1337_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1338_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1339_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1340_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1341_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1342_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1343_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1344_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1345_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1346_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1347_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1348_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1349_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1350_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1351_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1352_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1353_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1354_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1355_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1356_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1357_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1358_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1359_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1360_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1361_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1362_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1363_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1364_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1365_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1366_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1367_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1368_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1369_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1370_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1371_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1372_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1373_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1374_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1375_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1376_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1377_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1378_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1379_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1380_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1381_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1382_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1383_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1384_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1385_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1386_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1387_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1388_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1389_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1390_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1391_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1392_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1393_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1394_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1395_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1396_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1397_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1398_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1399_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1400_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1401_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1402_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1403_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1404_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1405_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1406_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1407_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1408_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1409_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1410_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1411_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1412_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1413_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1414_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1415_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1416_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1417_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1418_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1419_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1420_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1421_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1422_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1423_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1424_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1425_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1426_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1427_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1428_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1429_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1430_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1431_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1432_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1433_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1434_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1435_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1436_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1437_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1438_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1439_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1440_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1441_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1442_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1443_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1444_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1445_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1446_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1447_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1448_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1449_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1450_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1451_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1452_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1453_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1454_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1455_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1456_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1457_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1458_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1459_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1460_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1461_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1462_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1463_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1464_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1465_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1466_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1467_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1468_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1469_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1470_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1471_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1472_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1473_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1474_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1475_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1476_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1477_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1478_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1479_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1480_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1481_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1482_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1483_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1484_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1485_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1486_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1487_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1488_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1489_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1490_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1491_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1492_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1493_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1494_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1495_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1496_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1497_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1498_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1499_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1500_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1501_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1502_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1503_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1504_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1505_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1506_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1507_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1508_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1509_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1510_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1511_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1512_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1513_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1514_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1515_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1516_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1517_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1518_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1519_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1520_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1521_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1522_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1523_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1524_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1525_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1526_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1527_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1528_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1529_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1530_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1531_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1532_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1533_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1534_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1535_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1536_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1537_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1538_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1539_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1540_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1541_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1542_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1543_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1544_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1545_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1546_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1547_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1548_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1549_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1550_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1551_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1552_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1553_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1554_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1555_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1556_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1557_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1558_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1559_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1560_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1561_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1562_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1563_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1564_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1565_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1566_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1567_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1568_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1569_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1570_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1571_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1572_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1573_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1574_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1575_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1576_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1577_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1578_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1579_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1580_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1581_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1582_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1583_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1584_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1585_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1586_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1587_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1588_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1589_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1590_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1591_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1592_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1593_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1594_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1595_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1596_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1597_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1598_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1599_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1600_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1601_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1602_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1603_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1604_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1605_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1606_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1607_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1608_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1609_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1610_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1611_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1612_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1613_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1614_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1615_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1616_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1617_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1618_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1619_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1620_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1621_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1622_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1623_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1624_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1625_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1626_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1627_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1628_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1629_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1630_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1631_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1632_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1633_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1634_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1635_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1636_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1637_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1638_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1639_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1640_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1641_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1642_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1643_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1644_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1645_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1646_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1647_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1648_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1649_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1650_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1651_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1652_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1653_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1654_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1655_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1656_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1657_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1658_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1659_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1660_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1661_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1662_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1663_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1664_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1665_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1666_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1667_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1668_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1669_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1670_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1671_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1672_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1673_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1674_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1675_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1676_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1677_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1678_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1679_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1680_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1681_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1682_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1683_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1684_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1685_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1686_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1687_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1688_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1689_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1690_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1691_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1692_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1693_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1694_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1695_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1696_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1697_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1698_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1699_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1700_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1701_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1702_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1703_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1704_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1705_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1706_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1707_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1708_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1709_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1710_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1711_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1712_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1713_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1714_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1715_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1716_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1717_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1718_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1719_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1720_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1721_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1722_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1723_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1724_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1725_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1726_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1727_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1728_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1729_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1730_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1731_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1732_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1733_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1734_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1735_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1736_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1737_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1738_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1739_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1740_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1741_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1742_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1743_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1744_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1745_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1746_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1747_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1748_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1749_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1750_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1751_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1752_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1753_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1754_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1755_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1756_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1757_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1758_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1759_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1760_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1761_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1762_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1763_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1764_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1765_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1766_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1767_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1768_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1769_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1770_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1771_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1772_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1773_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1774_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1775_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1776_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1777_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1778_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1779_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1780_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1781_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1782_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1783_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1784_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1785_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1786_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1787_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1788_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1789_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1790_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1791_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1792_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1793_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1794_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1795_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1796_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1797_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1798_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1799_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1800_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1801_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1802_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1803_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1804_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1805_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1806_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1807_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1808_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1809_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1810_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1811_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1812_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1813_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1814_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1815_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1816_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1817_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1818_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1819_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1820_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1821_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1822_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1823_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1824_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1825_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1826_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1827_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1828_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1829_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1830_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1831_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1832_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1833_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1834_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1835_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1836_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1837_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1838_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1839_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1840_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1841_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1842_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1843_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1844_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1845_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1846_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1847_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1848_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1849_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1850_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1851_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1852_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1853_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1854_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1855_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1856_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1857_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1858_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1859_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1860_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1861_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1862_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1863_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1864_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1865_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1866_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1867_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1868_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1869_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1870_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1871_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1872_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1873_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1874_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1875_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1876_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1877_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1878_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1879_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1880_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1881_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1882_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1883_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1884_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1885_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1886_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1887_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1888_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1889_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1890_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1891_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1892_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1893_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1894_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1895_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1896_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1897_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1898_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1899_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1900_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1901_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1902_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1903_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1904_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1905_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1906_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1907_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1908_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1909_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1910_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1911_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1912_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1913_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1914_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1915_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1916_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1917_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1918_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1919_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1920_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1921_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1922_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1923_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1924_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1925_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1926_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1927_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1928_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1929_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1930_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1931_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1932_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1933_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1934_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1935_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1936_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1937_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1938_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1939_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1940_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1941_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1942_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1943_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1944_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1945_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1946_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1947_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1948_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1949_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1950_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1951_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1952_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1953_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1954_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1955_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg1956_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1488_1, arg1489_1, arg1490_1, arg1491_1, arg1492_1, arg1493_1, arg1494_1, arg1495_1, arg1496_1, arg1497_1, arg1498_1, arg1499_1, arg1500_1, arg1501_1, arg1502_1, arg1503_1, arg1504_1, arg1505_1, arg1506_1, arg1507_1, arg1508_1, arg1509_1, arg1510_1, arg1511_1, arg1512_1, arg1513_1, arg1514_1, arg1515_1, arg1516_1, arg1517_1, arg1518_1, arg1519_1, arg1520_1, arg1521_1, arg1522_1, arg1523_1, arg1524_1, arg1525_1, arg1526_1, arg1527_1, arg1528_1, arg1529_1, arg1530_1, arg1531_1, arg1532_1, arg1533_1, arg1534_1, arg1535_1, arg1536_1, arg1537_1, arg1538_1, arg1539_1, arg1540_1, arg1541_1, arg1542_1, arg1543_1, arg1544_1, arg1545_1, arg1546_1, arg1547_1, arg1548_1, arg1549_1, arg1550_1, arg1551_1, arg1552_1, arg1553_1, arg1554_1, arg1555_1, arg1556_1, arg1557_1, arg1558_1, arg1559_1, arg1560_1, arg1561_1, arg1562_1, arg1563_1, arg1564_1, arg1565_1, arg1566_1, arg1567_1, arg1568_1, arg1569_1, arg1570_1, arg1571_1, arg1572_1, arg1573_1, arg1574_1, arg1575_1, arg1576_1, arg1577_1, arg1578_1, arg1579_1, arg1580_1, arg1581_1, arg1582_1, arg1583_1, arg1584_1, arg1585_1, arg1586_1, arg1587_1, arg1588_1, arg1589_1, arg1590_1, arg1591_1, arg1592_1, arg1593_1, arg1594_1, arg1595_1, arg1596_1, arg1597_1, arg1598_1, arg1599_1, arg1600_1, arg1601_1, arg1602_1, arg1603_1, arg1604_1, arg1605_1, arg1606_1, arg1607_1, arg1608_1, arg1609_1, arg1610_1, arg1611_1, arg1612_1, arg1613_1, arg1614_1, arg1615_1, arg1616_1, arg1617_1, arg1618_1, arg1619_1, arg1620_1, arg1621_1, arg1622_1, arg1623_1, arg1624_1, arg1625_1, arg1626_1, arg1627_1, arg1628_1, arg1629_1, arg1630_1, arg1631_1, arg1632_1, arg1633_1, arg1634_1, arg1635_1, arg1636_1, arg1637_1, arg1638_1, arg1639_1, arg1640_1, arg1641_1, arg1642_1, arg1643_1, arg1644_1, arg1645_1, arg1646_1, arg1647_1, arg1648_1, arg1649_1, arg1650_1, arg1651_1, arg1652_1, arg1653_1, arg1654_1, arg1655_1, arg1656_1, arg1657_1, arg1658_1, arg1659_1, arg1660_1, arg1661_1, arg1662_1, arg1663_1, arg1664_1, arg1665_1, arg1666_1, arg1667_1, arg1668_1, arg1669_1, arg1670_1, arg1671_1, arg1672_1, arg1673_1, arg1674_1, arg1675_1, arg1676_1, arg1677_1, arg1678_1, arg1679_1, arg1680_1, arg1681_1, arg1682_1, arg1683_1, arg1684_1, arg1685_1, arg1686_1, arg1687_1, arg1688_1, arg1689_1, arg1690_1, arg1691_1, arg1692_1, arg1693_1, arg1694_1, arg1695_1, arg1696_1, arg1697_1, arg1698_1, arg1699_1, arg1700_1, arg1701_1, arg1702_1, arg1703_1, arg1704_1, arg1705_1, arg1706_1, arg1707_1, arg1708_1, arg1709_1, arg1710_1, arg1711_1, arg1712_1, arg1713_1, arg1714_1, arg1715_1, arg1716_1, arg1717_1, arg1718_1, arg1719_1, arg1720_1, arg1721_1, arg1722_1, arg1723_1, arg1724_1, arg1725_1, arg1726_1, arg1727_1, arg1728_1, arg1729_1, arg1730_1, arg1731_1, arg1732_1, arg1733_1, arg1734_1, arg1735_1, arg1736_1, arg1737_1, arg1738_1, arg1739_1, arg1740_1, arg1741_1, arg1742_1, arg1743_1, arg1744_1, arg1745_1, arg1746_1, arg1747_1, arg1748_1, arg1749_1, arg1750_1, arg1751_1, arg1752_1, arg1753_1, arg1754_1, arg1755_1, arg1756_1, arg1757_1, arg1758_1, arg1759_1, arg1760_1, arg1761_1, arg1762_1, arg1763_1, arg1764_1, arg1765_1, arg1766_1, arg1767_1, arg1768_1, arg1769_1, arg1770_1, arg1771_1, arg1772_1, arg1773_1, arg1774_1, arg1775_1, arg1776_1, arg1777_1, arg1778_1, arg1779_1, arg1780_1, arg1781_1, arg1782_1, arg1783_1, arg1784_1, arg1785_1, arg1786_1, arg1787_1, arg1788_1, arg1789_1, arg1790_1, arg1791_1, arg1792_1, arg1793_1, arg1794_1, arg1795_1, arg1796_1, arg1797_1, arg1798_1, arg1799_1, arg1800_1, arg1801_1, arg1802_1, arg1803_1, arg1804_1, arg1805_1, arg1806_1, arg1807_1, arg1808_1, arg1809_1, arg1810_1, arg1811_1, arg1812_1, arg1813_1, arg1814_1, arg1815_1, arg1816_1, arg1817_1, arg1818_1, arg1819_1, arg1820_1, arg1821_1, arg1822_1, arg1823_1, arg1824_1, arg1825_1, arg1826_1, arg1827_1, arg1828_1, arg1829_1, arg1830_1, arg1831_1, arg1832_1, arg1833_1, arg1834_1, arg1835_1, arg1836_1, arg1837_1, arg1838_1, arg1839_1, arg1840_1, arg1841_1, arg1842_1, arg1843_1, arg1844_1, arg1845_1, arg1846_1, arg1847_1, arg1848_1, arg1849_1, arg1850_1, arg1851_1, arg1852_1, arg1853_1, arg1854_1, arg1855_1, arg1856_1, arg1857_1, arg1858_1, arg1859_1, arg1860_1, arg1861_1, arg1862_1, arg1863_1, arg1864_1, arg1865_1, arg1866_1, arg1867_1, arg1868_1, arg1869_1, arg1870_1, arg1871_1, arg1872_1, arg1873_1, arg1874_1, arg1875_1, arg1876_1, arg1877_1, arg1878_1, arg1879_1, arg1880_1, arg1881_1, arg1882_1, arg1883_1, arg1884_1, arg1885_1, arg1886_1, arg1887_1, arg1888_1, arg1889_1, arg1890_1, arg1891_1, arg1892_1, arg1893_1, arg1894_1, arg1895_1, arg1896_1, arg1897_1, arg1898_1, arg1899_1, arg1900_1, arg1901_1, arg1902_1, arg1903_1, arg1904_1, arg1905_1, arg1906_1, arg1907_1, arg1908_1, arg1909_1, arg1910_1, arg1911_1, arg1912_1, arg1913_1, arg1914_1, arg1915_1, arg1916_1, arg1917_1, arg1918_1, arg1919_1, arg1920_1, arg1921_1, arg1922_1, arg1923_1, arg1924_1, arg1925_1, arg1926_1, arg1927_1, arg1928_1, arg1929_1, arg1930_1, arg1931_1, arg1932_1, arg1933_1, arg1934_1, arg1935_1, arg1936_1, arg1937_1, arg1938_1, arg1939_1, arg1940_1, arg1941_1, arg1942_1, arg1943_1, arg1944_1, arg1945_1, arg1946_1, arg1947_1, arg1948_1, arg1949_1, arg1950_1, arg1951_1, arg1952_1, arg1953_1, arg1954_1, arg1955_1, arg1956_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hrnet_w18', benchmark_compiled_module)
