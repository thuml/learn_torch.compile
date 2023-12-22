
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


# kernel path: /tmp/torchinductor_youkaichao/bc/cbc26ryhnv6k6kyzwcglaq4l7ibabopk2ewic4umotsrlafm2yx6.py
# Source Nodes: [shortcut, x_1, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# shortcut => mul_3, sigmoid
# x_1 => add_1, mul_1, mul_2, sub
# x_6 => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 16
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/7o/c7ou7hcqlmox46c5qjnkvi7362xwq6ax6jyqsgd3nbu42ri3tkrb.py
# Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_11 => mul_7, sigmoid_1
# x_12 => convolution_2
# x_7 => add_3, mul_5, mul_6, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 64
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjenspzyjqtdlcqe2l3qdlofkhzrydqjqlbdo4y7ufihijftgnk.py
# Source Nodes: [x_21, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
# x_21 => add_7, mul_13, mul_14, sub_3
# x_28 => convolution_4
triton_poi_fused__native_batch_norm_legit_no_training_convolution_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
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
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckisb5k7j4akm7m4xbd7gvctchiu4ydlted7soiyytqx3gmi7qog.py
# Source Nodes: [x_29, x_33, x_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_29 => add_9, mul_16, mul_17, sub_4
# x_33 => mul_18, sigmoid_3
# x_34 => convolution_5
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 128
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/caiool56dplowv43epvmvxud7bypbukhazrls4hviheruzqq6qln.py
# Source Nodes: [x_35, x_39, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_35 => add_11, mul_20, mul_21, sub_5
# x_39 => mul_22, sigmoid_4
# x_42 => convolution_6
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qo/cqou7jrf4wfzekkth7bvzjlrxjph7mfxjzq5tq63shw3ce326jhe.py
# Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_43 => add_13, mul_24, mul_25, sub_6
triton_poi_fused__native_batch_norm_legit_no_training_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
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
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5w/c5w3xgl3xdzbr5kt3736jmsdlks2cxageod3xmwbgfhsy2b3x53a.py
# Source Nodes: [x_51, x_55, x_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_51 => add_15, mul_27, mul_28, sub_7
# x_55 => mul_29, sigmoid_5
# x_56 => convolution_8
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lj/clj2jyxn6stqbdwlchsgv6gl53ainpfkb45ocjhp6er346el3dzp.py
# Source Nodes: [x_65, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# x_65 => add_19, mul_35, mul_36, sub_9
# x_72 => add_20
triton_poi_fused__native_batch_norm_legit_no_training_add_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
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
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ci/cci6ci7ognrryylhdivfp7zi7qa5ncrgscetsyye3phbwoeyq4ig.py
# Source Nodes: [x_103, x_107, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_103 => add_31, mul_53, mul_54, sub_14
# x_107 => mul_55, sigmoid_10
# x_110 => convolution_15
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 256
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cpr24spq4h5pt6getn7yitxa4ze64qr435mgt74rbzpzcdgd7d4k.py
# Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_111 => add_33, mul_57, mul_58, sub_15
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 96
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
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbh4su2yoqz6dhiscpwk2an6zlifywfhew7llm4s7qjai4piohh.py
# Source Nodes: [x_119, x_123, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_119 => add_35, mul_60, mul_61, sub_16
# x_123 => mul_62, sigmoid_11
# x_124 => convolution_17
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 96
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpacb6jzxsuqc6yzemmqkhzamv7wg5wpqqchlpgcmn24pzutki2.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1 => add_36, add_37, mul_63, mul_64, rsqrt, sub_17, var_mean
triton_red_fused_native_layer_norm_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 16)) + (32*((x0 % 4) // 2)) + (64*((((4*x1) + (1024*r2) + (147456*(x0 // 4)) + (x0 % 4)) // 64) % 18432)) + ((x0 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 16)) + (32*((x0 % 4) // 2)) + (64*((((4*x1) + (1024*r2) + (147456*(x0 // 4)) + (x0 % 4)) // 64) % 18432)) + ((x0 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 144.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (r2 + (144*x1) + (36864*x0)), tmp16, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sx/csxod6km6hhjiu5yujkbs3hb3us6a7i5bfnolncni6cxo6fhtta7.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2, x_131], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2 => var_mean_1
# x_131 => add_38
triton_red_fused_add_native_layer_norm_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*((((4*x0) + (1024*r2) + (147456*(x1 // 4)) + (x1 % 4)) // 64) % 18432)) + ((x1 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/66/c66lut5o3jzzexvzukzj2tiohu6wozxz4ecmjkxtuzavnjwqd3vz.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2, x_131], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2 => add_39, add_40, mul_65, mul_66, rsqrt_1, sub_18, var_mean_1
# x_131 => add_38
triton_poi_fused_add_native_layer_norm_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 36864
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 144
    x2 = (xindex // 144)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 16)) + (32*((y0 % 4) // 2)) + (64*((((4*x2) + (1024*x1) + (147456*(y0 // 4)) + (y0 % 4)) // 64) % 18432)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (36864*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (256*y0)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (256*y0)), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 144.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3 + (36864*y0)), tmp17, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3fgl6ixujdrdiqhopalwufsnamj35vkta6sguz6nal4xi3cgsx.py
# Source Nodes: [x_133], Original ATen: [aten.silu]
# x_133 => mul_67, sigmoid_12
triton_poi_fused_silu_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 288
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sr/csrkezpqxbabovuc3gt3dekuljl27n6xzv4e52d4a4h3djlkhnae.py
# Source Nodes: [x_131, x_138], Original ATen: [aten.add]
# x_131 => add_38
# x_138 => add_41
triton_poi_fused_add_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 36864
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 144
    x2 = (xindex // 144)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 16)) + (32*((y0 % 4) // 2)) + (64*((((4*x2) + (1024*x1) + (147456*(y0 // 4)) + (y0 % 4)) // 64) % 18432)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (36864*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x3 + (36864*y0)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (36864*y0)), tmp8, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5a/c5afzc7ws762dz7n2r5fryxm5jkq2egikwg3jkvtbxlnadgfeb7l.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1 => add_42, add_43, mul_68, mul_69, rsqrt_2, sub_19, var_mean_2
triton_per_fused_native_layer_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 144.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp27, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gy/cgybyzvxc3u5j6wrm2w5lhdfdquhhsurk3a6ksxy6zujxawdlgzl.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2, x_143], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2 => add_45, add_46, mul_70, mul_71, rsqrt_3, sub_20, var_mean_3
# x_143 => add_44
triton_per_fused_add_native_layer_norm_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 144.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp31, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pz/cpzkuqss5abzto642c5iy4smpiqs7xkgzabh4ogauuuybmgwwzyh.py
# Source Nodes: [x_143, x_151, x_152], Original ATen: [aten.add, aten.native_layer_norm]
# x_143 => add_44
# x_151 => add_47
# x_152 => var_mean_4
triton_per_fused_add_native_layer_norm_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp8, rmask)
    tl.store(out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cx6j7scsx76zagvlugtozxlphnobywmhsljjz7ij3fuy6honum.py
# Source Nodes: [x_156], Original ATen: [aten.convolution]
# x_156 => convolution_18
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 32
    x3 = (xindex // 32)
    y0 = yindex % 144
    y1 = (yindex // 144)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((144*((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (x2 % 2)) // 4) % 256)) + (36864*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (147456*((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (147456*y1) + (x2 % 2)) // 147456) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((256*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (1024*((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (147456*y1) + (x2 % 2)) // 147456) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (x2 % 2)) // 4) % 256)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((256*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (1024*((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (147456*y1) + (x2 % 2)) // 147456) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (x2 % 2)) // 4) % 256)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 144.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctq7irtth5uwimrxcvtizsltavv4ybw3qqtyglfsly335jmrvbz5.py
# Source Nodes: [cat_5, x_162], Original ATen: [aten.cat, aten.convolution]
# cat_5 => cat
# x_162 => convolution_19
triton_poi_fused_cat_convolution_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024) % 192
    x2 = (xindex // 196608)
    x3 = xindex % 196608
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (98304*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 192, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-98304) + x3 + (98304*x2)), tmp8, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mk4tm4wgdhp7jbqa2lri6er3pfahdnurancb3lqewob3fehllm.py
# Source Nodes: [x_169, x_173, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_169 => add_55, mul_84, mul_85, sub_24
# x_173 => mul_86, sigmoid_16
# x_174 => convolution_21
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 384
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rg/crg2remrhlbn7ndxdlefntitvzzgv2bqrv6sgst55m7lroqq6t3h.py
# Source Nodes: [x_175, x_179, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_175 => add_57, mul_88, mul_89, sub_25
# x_179 => mul_90, sigmoid_17
# x_182 => convolution_22
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 384
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbehdh7c4wjtrtfp6h33h7jexzys42i4r772tn22b3f36ludhigz.py
# Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_183 => add_59, mul_92, mul_93, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 128
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
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/th/cthdrtrtqsbmemcg3wzzhh7idxkisqgohbh4xcmjisxurgcqr6mc.py
# Source Nodes: [x_191, x_195, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_191 => add_61, mul_95, mul_96, sub_27
# x_195 => mul_97, sigmoid_18
# x_196 => convolution_24
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 128
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6qbsalno6upkyvm2nvgigkmqt32bmnb2g4gxajixsuff2xjjr5.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 => var_mean_5
triton_red_fused_native_layer_norm_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x2 = (xindex // 2048)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r3) + (24576*x2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 12288)) + ((x0 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, None)
    tl.store(out_ptr1 + (x4), tmp3, None)
    tl.store(out_ptr2 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c456mdfwxqb42topa7h3z3iqaywt3oej7wolhblvbltjofqdlyxu.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 => var_mean_5
triton_per_fused_native_layer_norm_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x3 + (2048*r2)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (2048*r2)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x3 + (2048*r2)), rmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp3, 0)
    tmp8 = tl.where(rmask, tmp4, 0)
    tmp9 = tl.where(rmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozsw7upe4wzhlstfntijy5vr6lnkmqd3amntmouyqp2ffpnwmas.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 => add_62, add_63, mul_98, mul_99, rsqrt_5, sub_28, var_mean_5
triton_poi_fused_native_layer_norm_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 12288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 192
    x2 = (xindex // 192)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 8)) + (16*((y0 % 4) // 2)) + (32*((((4*x2) + (256*x1) + (49152*(y0 // 4)) + (y0 % 4)) // 32) % 12288)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (32*x2)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (12288*y0)), tmp13, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crn5d4kyyeze6d5bknwqhtiephvt2b62kt3rl7dnqeftdydir7h7.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 => var_mean_6
# x_203 => add_64
triton_red_fused_add_native_layer_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 64
    x2 = (xindex // 128)
    x4 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x2 % 4)) // 4) % 8)) + (16*((x2 % 4) // 2)) + (32*((((4*x1) + (256*r3) + (24576*x0) + (49152*(x2 // 4)) + (x2 % 4)) // 32) % 12288)) + ((x2 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + (96*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
    tl.store(out_ptr1 + (x4), tmp7, None)
    tl.store(out_ptr2 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5dceaxrjlbhm7rim7vr5h6rq7uxptk25a4oznkipefkya3k3ev.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 => var_mean_6
# x_203 => add_64
triton_per_fused_add_native_layer_norm_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), rmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp3, 0)
    tmp8 = tl.where(rmask, tmp4, 0)
    tmp9 = tl.where(rmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr1 + (x0), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2ogutc6e6fbaqkacx3f5oxscje3e7anxitm455knohcvxg7x7k.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 => add_65, add_66, mul_100, mul_101, rsqrt_6, sub_29, var_mean_6
# x_203 => add_64
triton_poi_fused_add_native_layer_norm_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 12288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 192
    x2 = (xindex // 192)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 8)) + (16*((y0 % 4) // 2)) + (32*((((4*x2) + (256*x1) + (49152*(y0 // 4)) + (y0 % 4)) // 32) % 12288)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (12288*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 192.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3 + (12288*y0)), tmp17, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c27ut2zkwwx4mlheugs3b6fvrsmgpgdibyumhtxojmc7hgycrrv4.py
# Source Nodes: [x_205], Original ATen: [aten.silu]
# x_205 => mul_102, sigmoid_19
triton_poi_fused_silu_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2hn5dhx7f622vr4wotkri6dsvltugzf5sdvpz4uei7b35rd7xc.py
# Source Nodes: [x_203, x_210], Original ATen: [aten.add]
# x_203 => add_64
# x_210 => add_67
triton_poi_fused_add_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 12288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 192
    x2 = (xindex // 192)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 8)) + (16*((y0 % 4) // 2)) + (32*((((4*x2) + (256*x1) + (49152*(y0 // 4)) + (y0 % 4)) // 32) % 12288)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + (12288*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x3 + (12288*y0)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (12288*y0)), tmp8, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgx6oke6jprej26bntvjgw4ffdxwk6d74keboomyzqvclktdecd.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1 => add_68, add_69, mul_103, mul_104, rsqrt_7, sub_30, var_mean_7
triton_per_fused_native_layer_norm_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 192.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp27, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/cko42z4euq3ylq7uuw2q4wnsyx4i6cs7p2qjndwnrarjtbdj3a5o.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2, x_215], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2 => add_71, add_72, mul_105, mul_106, rsqrt_8, sub_31, var_mean_8
# x_215 => add_70
triton_per_fused_add_native_layer_norm_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp31, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clf7yin3v65mzuac2xyyalkpntuni2whoeylyzzdqxiuephdadc7.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1, x_215, x_222], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1 => add_74, add_75, mul_108, mul_109, rsqrt_9, sub_32, var_mean_9
# x_215 => add_70
# x_222 => add_73
triton_per_fused_add_native_layer_norm_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_35', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp35, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c626vjkh3gwzqnhuppct5qaqax7bnennopxuzwwth63rew7smryr.py
# Source Nodes: [x_239, x_247, x_248], Original ATen: [aten.add, aten.native_layer_norm]
# x_239 => add_82
# x_247 => add_85
# x_248 => var_mean_13
triton_per_fused_add_native_layer_norm_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask)
    tl.store(out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/czttpbn3mfa6ydigl3db5i3fovhfuwlzxw7sdk2puyaseiuqi6rq.py
# Source Nodes: [x_252], Original ATen: [aten.convolution]
# x_252 => convolution_25
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 16
    x3 = (xindex // 16)
    y0 = yindex % 192
    y1 = (yindex // 192)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((192*((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (x2 % 2)) // 4) % 64)) + (12288*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (49152*((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (49152*y1) + (x2 % 2)) // 49152) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (x2 % 2)) // 256) % 192)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((64*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (256*((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (49152*y1) + (x2 % 2)) // 49152) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (x2 % 2)) // 4) % 64)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((64*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (256*((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (49152*y1) + (x2 % 2)) // 49152) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (x2 % 2)) // 4) % 64)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (x2 % 2)) // 256) % 192), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (x2 % 2)) // 256) % 192), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x5 + (256*y4)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g6/cg6ou57gephk6isshhoypmlcxh2nolpdoxbfhnbfleg7bjbf7oyc.py
# Source Nodes: [cat_4, x_258], Original ATen: [aten.cat, aten.convolution]
# cat_4 => cat_1
# x_258 => convolution_26
triton_poi_fused_cat_convolution_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 256
    x2 = (xindex // 65536)
    x3 = xindex % 65536
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (32768*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 256, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-32768) + x3 + (32768*x2)), tmp8, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcuw7pyv3hwkmg4cmqcpqd4v6dhrifnromugimoeun25altmpk3.py
# Source Nodes: [x_265, x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_265 => add_93, mul_129, mul_130, sub_39
# x_269 => mul_131, sigmoid_25
# x_270 => convolution_28
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 512
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndufax2uqjzczuj3qotn7f2heswvmm6a5fkym3sgshgn5nmbaqv.py
# Source Nodes: [x_271, x_275, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_271 => add_95, mul_133, mul_134, sub_40
# x_275 => mul_135, sigmoid_26
# x_278 => convolution_29
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wk/cwki4546uhk7jonandz763o3oif3z5mw5al7gjhl2osgibyfrix5.py
# Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_279 => add_97, mul_137, mul_138, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 160
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
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/24/c24pr6qbr5n5wnjkwocgpjjkrtycvczjz7uyy4ployv3fkeckzba.py
# Source Nodes: [x_287, x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_287 => add_99, mul_140, mul_141, sub_42
# x_291 => mul_142, sigmoid_27
# x_292 => convolution_31
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 160
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3d4pzjdjx3btzy2lvcmdjsh4dw2665fxt2ct27wruplth3pr3lp.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 => var_mean_14
triton_red_fused_native_layer_norm_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r3) + (7680*x2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 7680)) + ((x0 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
    tl.store(out_ptr2 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dc/cdctkregmzriu5l4rlm3jvtkt56zd6frgxchyfyre6kndwyerb4r.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 => var_mean_14
triton_per_fused_native_layer_norm_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x3 + (512*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (512*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x3 + (512*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x1 + (16*x0)), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oc/cocple4swspzdjnzpikesxdztwx5xn2boyc3ujgineksauynlsx4.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 => add_100, add_101, mul_143, mul_144, rsqrt_14, sub_43, var_mean_14
triton_poi_fused_native_layer_norm_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 3840
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 240
    x2 = (xindex // 240)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 4)) + (8*((y0 % 4) // 2)) + (16*((((4*x2) + (64*x1) + (15360*(y0 // 4)) + (y0 % 4)) // 16) % 7680)) + ((y0 % 4) % 2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (32*x2)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 240.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (3840*y0)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bg/cbggf6o5quuxbqijt2qxihlbsne3v32bhbc52ai5v7tifddou6rh.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2 => var_mean_15
# x_299 => add_102
triton_red_fused_add_native_layer_norm_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 16
    x2 = (xindex // 32)
    x4 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x2 % 4)) // 4) % 4)) + (8*((x2 % 4) // 2)) + (16*((((4*x1) + (64*r3) + (7680*x0) + (15360*(x2 // 4)) + (x2 % 4)) // 16) % 7680)) + ((x2 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (120*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + (120*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4sm5izadrirdvn5d64mkrinldeons7n67uatzoye25rn54yzza.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2 => var_mean_15
# x_299 => add_102
triton_per_fused_add_native_layer_norm_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/czn2a2uzlv2b3goymogyxm4to77tvs2bhxj4357m2xa3xqgp7p2h.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2 => add_103, add_104, mul_145, mul_146, rsqrt_15, sub_44, var_mean_15
# x_299 => add_102
triton_poi_fused_add_native_layer_norm_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 3840
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 240
    x2 = (xindex // 240)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 4)) + (8*((y0 % 4) // 2)) + (16*((((4*x2) + (64*x1) + (15360*(y0 // 4)) + (y0 % 4)) // 16) % 7680)) + ((y0 % 4) % 2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (3840*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 240.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3 + (3840*y0)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6xakslpo44idvvkigly4pzrm6lvwnuhwkrdxxb5mo4ev3vvmai.py
# Source Nodes: [x_301], Original ATen: [aten.silu]
# x_301 => mul_147, sigmoid_28
triton_poi_fused_silu_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7sccgtwkpqbl5qe4nwdtdqhaeeircawh7r7td4qoo2nk2ingawx.py
# Source Nodes: [x_299, x_306], Original ATen: [aten.add]
# x_299 => add_102
# x_306 => add_105
triton_poi_fused_add_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 3840
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 240
    x2 = (xindex // 240)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 4)) + (8*((y0 % 4) // 2)) + (16*((((4*x2) + (64*x1) + (15360*(y0 // 4)) + (y0 % 4)) // 16) % 7680)) + ((y0 % 4) % 2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + (3840*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x3 + (3840*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (3840*y0)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosus4uxiewdbxqwd3efsjo5nds6cnvwwoj5u6a3ciugd7ibng65.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1 => add_106, add_107, mul_148, mul_149, rsqrt_16, sub_45, var_mean_16
triton_per_fused_native_layer_norm_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 240.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewe2cifjeacexpxlmymjbfck5u3e2ki3jdkiz5sjav3laakkoss.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2, x_311], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2 => add_109, add_110, mul_150, mul_151, rsqrt_17, sub_46, var_mean_17
# x_311 => add_108
triton_per_fused_add_native_layer_norm_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 240.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyp6s6c4uhsrvusodmjssjylkrwv2gty25opqvsyfyurkracdng.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1, x_311, x_318], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1 => add_112, add_113, mul_153, mul_154, rsqrt_18, sub_47, var_mean_18
# x_311 => add_108
# x_318 => add_111
triton_per_fused_add_native_layer_norm_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_53', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 240.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (240*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2wxgafsmjeea6j26rji5z26kkj52esxldpos4kuaez2bexpibl.py
# Source Nodes: [x_323, x_331, x_332], Original ATen: [aten.add, aten.native_layer_norm]
# x_323 => add_114
# x_331 => add_117
# x_332 => var_mean_20
triton_per_fused_add_native_layer_norm_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(in_out_ptr0 + (r1 + (240*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h6/ch67lefw5w4vks2hdx6sb6jt4u6mnyrhusxomyp7xakkw2dlecyr.py
# Source Nodes: [x_336], Original ATen: [aten.convolution]
# x_336 => convolution_32
triton_poi_fused_convolution_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 240
    y1 = (yindex // 240)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((240*((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (x2 % 2)) // 4) % 16)) + (3840*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (15360*((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (15360*y1) + (x2 % 2)) // 15360) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((16*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (64*((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (15360*y1) + (x2 % 2)) // 15360) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (x2 % 2)) // 4) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (64*((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (15360*y1) + (x2 % 2)) // 15360) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (x2 % 2)) // 4) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 240.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x5 + (64*y4)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqei4iqllb47iuhv5wvhxwxzksrccqv7sin2cfiijjr4patz3ov.py
# Source Nodes: [cat_3, x_342], Original ATen: [aten.cat, aten.convolution]
# cat_3 => cat_2
# x_342 => convolution_33
triton_poi_fused_cat_convolution_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 320
    x2 = (xindex // 20480)
    x3 = xindex % 20480
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 160, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (10240*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 320, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-10240) + x3 + (10240*x2)), tmp8, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbku4chpqjbxobhwmfldm3cc5ek4hayoskc23bnxoih3lqvnddc.py
# Source Nodes: [x_350, x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_350 => add_125, mul_169, mul_170, sub_52
# x_355 => mul_171, sigmoid_33
# x_356 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_57', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 640
    tmp0 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 64.0
    tmp22 = tmp20 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, ), (1, ))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (32, ), (1, ))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (96, ), (1, ))
    assert_size_stride(arg31_1, (96, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (96, ), (1, ))
    assert_size_stride(arg36_1, (96, ), (1, ))
    assert_size_stride(arg37_1, (96, ), (1, ))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (384, ), (1, ))
    assert_size_stride(arg41_1, (384, ), (1, ))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (160, ), (1, ))
    assert_size_stride(arg55_1, (160, ), (1, ))
    assert_size_stride(arg56_1, (160, ), (1, ))
    assert_size_stride(arg57_1, (160, ), (1, ))
    assert_size_stride(arg58_1, (160, ), (1, ))
    assert_size_stride(arg59_1, (160, ), (1, ))
    assert_size_stride(arg60_1, (160, ), (1, ))
    assert_size_stride(arg61_1, (160, ), (1, ))
    assert_size_stride(arg62_1, (640, ), (1, ))
    assert_size_stride(arg63_1, (640, ), (1, ))
    assert_size_stride(arg64_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg65_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg66_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg67_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg68_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg69_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg70_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg71_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg72_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg73_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg74_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg75_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg76_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg77_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg78_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg79_1, (96, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg80_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg81_1, (144, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg82_1, (144, ), (1, ))
    assert_size_stride(arg83_1, (144, ), (1, ))
    assert_size_stride(arg84_1, (432, 144), (144, 1))
    assert_size_stride(arg85_1, (432, ), (1, ))
    assert_size_stride(arg86_1, (144, 144), (144, 1))
    assert_size_stride(arg87_1, (144, ), (1, ))
    assert_size_stride(arg88_1, (144, ), (1, ))
    assert_size_stride(arg89_1, (144, ), (1, ))
    assert_size_stride(arg90_1, (288, 144), (144, 1))
    assert_size_stride(arg91_1, (288, ), (1, ))
    assert_size_stride(arg92_1, (144, 288), (288, 1))
    assert_size_stride(arg93_1, (144, ), (1, ))
    assert_size_stride(arg94_1, (144, ), (1, ))
    assert_size_stride(arg95_1, (144, ), (1, ))
    assert_size_stride(arg96_1, (432, 144), (144, 1))
    assert_size_stride(arg97_1, (432, ), (1, ))
    assert_size_stride(arg98_1, (144, 144), (144, 1))
    assert_size_stride(arg99_1, (144, ), (1, ))
    assert_size_stride(arg100_1, (144, ), (1, ))
    assert_size_stride(arg101_1, (144, ), (1, ))
    assert_size_stride(arg102_1, (288, 144), (144, 1))
    assert_size_stride(arg103_1, (288, ), (1, ))
    assert_size_stride(arg104_1, (144, 288), (288, 1))
    assert_size_stride(arg105_1, (144, ), (1, ))
    assert_size_stride(arg106_1, (144, ), (1, ))
    assert_size_stride(arg107_1, (144, ), (1, ))
    assert_size_stride(arg108_1, (96, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg109_1, (96, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg110_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg111_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg112_1, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg113_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg114_1, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg115_1, (192, ), (1, ))
    assert_size_stride(arg116_1, (192, ), (1, ))
    assert_size_stride(arg117_1, (576, 192), (192, 1))
    assert_size_stride(arg118_1, (576, ), (1, ))
    assert_size_stride(arg119_1, (192, 192), (192, 1))
    assert_size_stride(arg120_1, (192, ), (1, ))
    assert_size_stride(arg121_1, (192, ), (1, ))
    assert_size_stride(arg122_1, (192, ), (1, ))
    assert_size_stride(arg123_1, (384, 192), (192, 1))
    assert_size_stride(arg124_1, (384, ), (1, ))
    assert_size_stride(arg125_1, (192, 384), (384, 1))
    assert_size_stride(arg126_1, (192, ), (1, ))
    assert_size_stride(arg127_1, (192, ), (1, ))
    assert_size_stride(arg128_1, (192, ), (1, ))
    assert_size_stride(arg129_1, (576, 192), (192, 1))
    assert_size_stride(arg130_1, (576, ), (1, ))
    assert_size_stride(arg131_1, (192, 192), (192, 1))
    assert_size_stride(arg132_1, (192, ), (1, ))
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, ), (1, ))
    assert_size_stride(arg135_1, (384, 192), (192, 1))
    assert_size_stride(arg136_1, (384, ), (1, ))
    assert_size_stride(arg137_1, (192, 384), (384, 1))
    assert_size_stride(arg138_1, (192, ), (1, ))
    assert_size_stride(arg139_1, (192, ), (1, ))
    assert_size_stride(arg140_1, (192, ), (1, ))
    assert_size_stride(arg141_1, (576, 192), (192, 1))
    assert_size_stride(arg142_1, (576, ), (1, ))
    assert_size_stride(arg143_1, (192, 192), (192, 1))
    assert_size_stride(arg144_1, (192, ), (1, ))
    assert_size_stride(arg145_1, (192, ), (1, ))
    assert_size_stride(arg146_1, (192, ), (1, ))
    assert_size_stride(arg147_1, (384, 192), (192, 1))
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (192, 384), (384, 1))
    assert_size_stride(arg150_1, (192, ), (1, ))
    assert_size_stride(arg151_1, (192, ), (1, ))
    assert_size_stride(arg152_1, (192, ), (1, ))
    assert_size_stride(arg153_1, (576, 192), (192, 1))
    assert_size_stride(arg154_1, (576, ), (1, ))
    assert_size_stride(arg155_1, (192, 192), (192, 1))
    assert_size_stride(arg156_1, (192, ), (1, ))
    assert_size_stride(arg157_1, (192, ), (1, ))
    assert_size_stride(arg158_1, (192, ), (1, ))
    assert_size_stride(arg159_1, (384, 192), (192, 1))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (192, 384), (384, 1))
    assert_size_stride(arg162_1, (192, ), (1, ))
    assert_size_stride(arg163_1, (192, ), (1, ))
    assert_size_stride(arg164_1, (192, ), (1, ))
    assert_size_stride(arg165_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg166_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg167_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg168_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg169_1, (160, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg170_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg171_1, (240, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg172_1, (240, ), (1, ))
    assert_size_stride(arg173_1, (240, ), (1, ))
    assert_size_stride(arg174_1, (720, 240), (240, 1))
    assert_size_stride(arg175_1, (720, ), (1, ))
    assert_size_stride(arg176_1, (240, 240), (240, 1))
    assert_size_stride(arg177_1, (240, ), (1, ))
    assert_size_stride(arg178_1, (240, ), (1, ))
    assert_size_stride(arg179_1, (240, ), (1, ))
    assert_size_stride(arg180_1, (480, 240), (240, 1))
    assert_size_stride(arg181_1, (480, ), (1, ))
    assert_size_stride(arg182_1, (240, 480), (480, 1))
    assert_size_stride(arg183_1, (240, ), (1, ))
    assert_size_stride(arg184_1, (240, ), (1, ))
    assert_size_stride(arg185_1, (240, ), (1, ))
    assert_size_stride(arg186_1, (720, 240), (240, 1))
    assert_size_stride(arg187_1, (720, ), (1, ))
    assert_size_stride(arg188_1, (240, 240), (240, 1))
    assert_size_stride(arg189_1, (240, ), (1, ))
    assert_size_stride(arg190_1, (240, ), (1, ))
    assert_size_stride(arg191_1, (240, ), (1, ))
    assert_size_stride(arg192_1, (480, 240), (240, 1))
    assert_size_stride(arg193_1, (480, ), (1, ))
    assert_size_stride(arg194_1, (240, 480), (480, 1))
    assert_size_stride(arg195_1, (240, ), (1, ))
    assert_size_stride(arg196_1, (240, ), (1, ))
    assert_size_stride(arg197_1, (240, ), (1, ))
    assert_size_stride(arg198_1, (720, 240), (240, 1))
    assert_size_stride(arg199_1, (720, ), (1, ))
    assert_size_stride(arg200_1, (240, 240), (240, 1))
    assert_size_stride(arg201_1, (240, ), (1, ))
    assert_size_stride(arg202_1, (240, ), (1, ))
    assert_size_stride(arg203_1, (240, ), (1, ))
    assert_size_stride(arg204_1, (480, 240), (240, 1))
    assert_size_stride(arg205_1, (480, ), (1, ))
    assert_size_stride(arg206_1, (240, 480), (480, 1))
    assert_size_stride(arg207_1, (240, ), (1, ))
    assert_size_stride(arg208_1, (240, ), (1, ))
    assert_size_stride(arg209_1, (240, ), (1, ))
    assert_size_stride(arg210_1, (160, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg211_1, (160, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg212_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg213_1, (1000, 640), (640, 1))
    assert_size_stride(arg214_1, (1000, ), (1, ))
    assert_size_stride(arg215_1, (16, ), (1, ))
    assert_size_stride(arg216_1, (16, ), (1, ))
    assert_size_stride(arg217_1, (64, ), (1, ))
    assert_size_stride(arg218_1, (64, ), (1, ))
    assert_size_stride(arg219_1, (64, ), (1, ))
    assert_size_stride(arg220_1, (64, ), (1, ))
    assert_size_stride(arg221_1, (32, ), (1, ))
    assert_size_stride(arg222_1, (32, ), (1, ))
    assert_size_stride(arg223_1, (128, ), (1, ))
    assert_size_stride(arg224_1, (128, ), (1, ))
    assert_size_stride(arg225_1, (128, ), (1, ))
    assert_size_stride(arg226_1, (128, ), (1, ))
    assert_size_stride(arg227_1, (64, ), (1, ))
    assert_size_stride(arg228_1, (64, ), (1, ))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (256, ), (1, ))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (64, ), (1, ))
    assert_size_stride(arg234_1, (64, ), (1, ))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, ), (1, ))
    assert_size_stride(arg237_1, (256, ), (1, ))
    assert_size_stride(arg238_1, (256, ), (1, ))
    assert_size_stride(arg239_1, (64, ), (1, ))
    assert_size_stride(arg240_1, (64, ), (1, ))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (96, ), (1, ))
    assert_size_stride(arg246_1, (96, ), (1, ))
    assert_size_stride(arg247_1, (96, ), (1, ))
    assert_size_stride(arg248_1, (96, ), (1, ))
    assert_size_stride(arg249_1, (96, ), (1, ))
    assert_size_stride(arg250_1, (96, ), (1, ))
    assert_size_stride(arg251_1, (96, ), (1, ))
    assert_size_stride(arg252_1, (96, ), (1, ))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (384, ), (1, ))
    assert_size_stride(arg255_1, (384, ), (1, ))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (128, ), (1, ))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (128, ), (1, ))
    assert_size_stride(arg264_1, (128, ), (1, ))
    assert_size_stride(arg265_1, (512, ), (1, ))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (160, ), (1, ))
    assert_size_stride(arg270_1, (160, ), (1, ))
    assert_size_stride(arg271_1, (160, ), (1, ))
    assert_size_stride(arg272_1, (160, ), (1, ))
    assert_size_stride(arg273_1, (160, ), (1, ))
    assert_size_stride(arg274_1, (160, ), (1, ))
    assert_size_stride(arg275_1, (160, ), (1, ))
    assert_size_stride(arg276_1, (160, ), (1, ))
    assert_size_stride(arg277_1, (640, ), (1, ))
    assert_size_stride(arg278_1, (640, ), (1, ))
    assert_size_stride(arg279_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg279_1, arg64_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 16, 128, 128), (262144, 16384, 128, 1))
        del arg279_1
        del arg64_1
        buf1 = buf0; del buf0  # reuse
        buf2 = buf1; del buf1  # reuse
        # Source Nodes: [shortcut, x_1, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_0.run(buf2, arg215_1, arg216_1, arg0_1, arg1_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg0_1
        del arg1_1
        del arg215_1
        del arg216_1
        # Source Nodes: [shortcut, x_6], Original ATen: [aten.convolution, aten.silu]
        buf3 = extern_kernels.convolution(buf2, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg65_1
        del buf2
        buf4 = buf3; del buf3  # reuse
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_1.run(buf5, arg217_1, arg218_1, arg2_1, arg3_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg217_1
        del arg218_1
        del arg2_1
        del arg3_1
        # Source Nodes: [x_11, x_12], Original ATen: [aten.convolution, aten.silu]
        buf6 = extern_kernels.convolution(buf5, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf6, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg66_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf8 = buf7; del buf7  # reuse
        # Source Nodes: [x_13, x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_1.run(buf8, arg219_1, arg220_1, arg4_1, arg5_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg219_1
        del arg220_1
        del arg4_1
        del arg5_1
        # Source Nodes: [x_17, x_20], Original ATen: [aten.convolution, aten.silu]
        buf9 = extern_kernels.convolution(buf8, arg67_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 32, 128, 128), (524288, 16384, 128, 1))
        del arg67_1
        del buf8
        buf10 = buf9; del buf9  # reuse
        # Source Nodes: [x_21, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_2.run(buf10, arg221_1, arg222_1, arg6_1, arg7_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg221_1
        del arg222_1
        del arg6_1
        del arg7_1
        # Source Nodes: [x_21, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 128, 128, 128), (2097152, 16384, 128, 1))
        del arg68_1
        del buf10
        buf12 = buf11; del buf11  # reuse
        buf13 = buf12; del buf12  # reuse
        # Source Nodes: [x_29, x_33, x_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_3.run(buf13, arg223_1, arg224_1, arg8_1, arg9_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg223_1
        del arg224_1
        del arg8_1
        del arg9_1
        # Source Nodes: [x_33, x_34], Original ATen: [aten.convolution, aten.silu]
        buf14 = extern_kernels.convolution(buf13, arg69_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf14, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg69_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        buf16 = buf15; del buf15  # reuse
        # Source Nodes: [x_35, x_39, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4.run(buf16, arg225_1, arg226_1, arg10_1, arg11_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg10_1
        del arg11_1
        del arg225_1
        del arg226_1
        # Source Nodes: [x_39, x_42], Original ATen: [aten.convolution, aten.silu]
        buf17 = extern_kernels.convolution(buf16, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg70_1
        del buf16
        buf18 = buf17; del buf17  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_5.run(buf18, arg227_1, arg228_1, arg12_1, arg13_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg12_1
        del arg13_1
        del arg227_1
        del arg228_1
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg71_1
        buf20 = buf19; del buf19  # reuse
        buf21 = buf20; del buf20  # reuse
        # Source Nodes: [x_51, x_55, x_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_6.run(buf21, arg229_1, arg230_1, arg14_1, arg15_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg14_1
        del arg15_1
        del arg229_1
        del arg230_1
        # Source Nodes: [x_55, x_56], Original ATen: [aten.convolution, aten.silu]
        buf22 = extern_kernels.convolution(buf21, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf22, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg72_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [x_57, x_61, x_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_6.run(buf24, arg231_1, arg232_1, arg16_1, arg17_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg16_1
        del arg17_1
        del arg231_1
        del arg232_1
        # Source Nodes: [x_61, x_64], Original ATen: [aten.convolution, aten.silu]
        buf25 = extern_kernels.convolution(buf24, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg73_1
        del buf24
        buf26 = buf18; del buf18  # reuse
        # Source Nodes: [x_65, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_7.run(buf26, buf25, arg233_1, arg234_1, arg18_1, arg19_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg18_1
        del arg19_1
        del arg233_1
        del arg234_1
        del buf25
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg74_1
        buf28 = buf27; del buf27  # reuse
        buf29 = buf28; del buf28  # reuse
        # Source Nodes: [x_74, x_78, x_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_6.run(buf29, arg235_1, arg236_1, arg20_1, arg21_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg20_1
        del arg21_1
        del arg235_1
        del arg236_1
        # Source Nodes: [x_78, x_79], Original ATen: [aten.convolution, aten.silu]
        buf30 = extern_kernels.convolution(buf29, arg75_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf30, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg75_1
        del buf29
        buf31 = buf30; del buf30  # reuse
        buf32 = buf31; del buf31  # reuse
        # Source Nodes: [x_80, x_84, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_6.run(buf32, arg237_1, arg238_1, arg22_1, arg23_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg22_1
        del arg237_1
        del arg238_1
        del arg23_1
        # Source Nodes: [x_84, x_87], Original ATen: [aten.convolution, aten.silu]
        buf33 = extern_kernels.convolution(buf32, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg76_1
        del buf32
        buf34 = buf26; del buf26  # reuse
        # Source Nodes: [x_88, x_95, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_7.run(buf34, buf33, arg239_1, arg240_1, arg24_1, arg25_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg239_1
        del arg240_1
        del arg24_1
        del arg25_1
        del buf33
        # Source Nodes: [x_88, x_95, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg77_1
        del buf34
        buf36 = buf35; del buf35  # reuse
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [x_101, x_102, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_6.run(buf37, arg241_1, arg242_1, arg26_1, arg27_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg241_1
        del arg242_1
        del arg26_1
        del arg27_1
        # Source Nodes: [x_101, x_102], Original ATen: [aten.convolution, aten.silu]
        buf38 = extern_kernels.convolution(buf37, arg78_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf38, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg78_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        buf40 = buf39; del buf39  # reuse
        # Source Nodes: [x_103, x_107, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_8.run(buf40, arg243_1, arg244_1, arg28_1, arg29_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg243_1
        del arg244_1
        del arg28_1
        del arg29_1
        # Source Nodes: [x_107, x_110], Original ATen: [aten.convolution, aten.silu]
        buf41 = extern_kernels.convolution(buf40, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 96, 32, 32), (98304, 1024, 32, 1))
        del arg79_1
        del buf40
        buf42 = buf41; del buf41  # reuse
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf42, arg245_1, arg246_1, arg30_1, arg31_1, 786432, grid=grid(786432), stream=stream0)
        del arg245_1
        del arg246_1
        del arg30_1
        del arg31_1
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg80_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 96, 32, 32), (98304, 1024, 32, 1))
        del arg80_1
        buf44 = buf43; del buf43  # reuse
        buf45 = buf44; del buf44  # reuse
        # Source Nodes: [x_119, x_123, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_10.run(buf45, arg247_1, arg248_1, arg32_1, arg33_1, 786432, grid=grid(786432), stream=stream0)
        del arg247_1
        del arg248_1
        del arg32_1
        del arg33_1
        # Source Nodes: [x_123, x_124], Original ATen: [aten.convolution, aten.silu]
        buf46 = extern_kernels.convolution(buf45, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 144, 32, 32), (147456, 1024, 32, 1))
        del arg81_1
        del buf45
        buf50 = empty((32, 256, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_11.run(buf46, arg82_1, arg83_1, buf50, 8192, 144, grid=grid(8192), stream=stream0)
        del arg82_1
        del arg83_1
        buf51 = empty((8192, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg85_1, reinterpret_tensor(buf50, (8192, 144), (144, 1), 0), reinterpret_tensor(arg84_1, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf51)
        del arg84_1
        del arg85_1
        # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf52 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf51, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf51, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf51, (32, 4, 256, 36), (110592, 36, 432, 1), 288), None, False)
        buf53 = buf52[0]
        del buf52
        buf57 = reinterpret_tensor(buf50, (8192, 144), (144, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (8192, 144), (144, 1), 0), reinterpret_tensor(arg86_1, (144, 144), (1, 144), 0), out=buf57)
        del arg86_1
        buf58 = empty_strided((32, 256, 1), (256, 1, 8192), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((32, 256, 1), (256, 1, 8192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2, x_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf46, buf57, arg87_1, buf58, buf59, 8192, 144, grid=grid(8192), stream=stream0)
        buf61 = reinterpret_tensor(buf53, (32, 256, 144), (36864, 144, 1), 0); del buf53  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2, x_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_13.run(buf46, buf57, arg87_1, buf58, buf59, arg88_1, arg89_1, buf61, 32, 36864, grid=grid(32, 36864), stream=stream0)
        del arg88_1
        del arg89_1
        buf62 = empty((8192, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (8192, 144), (144, 1), 0), reinterpret_tensor(arg90_1, (144, 288), (1, 144), 0), out=buf62)
        del arg90_1
        buf63 = reinterpret_tensor(buf62, (32, 256, 288), (73728, 288, 1), 0); del buf62  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.silu]
        triton_poi_fused_silu_14.run(buf63, arg91_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg91_1
        buf64 = reinterpret_tensor(buf61, (8192, 144), (144, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (8192, 288), (288, 1), 0), reinterpret_tensor(arg92_1, (288, 144), (1, 288), 0), out=buf64)
        del arg92_1
        buf65 = reinterpret_tensor(buf64, (32, 256, 144), (36864, 144, 1), 0); del buf64  # reuse
        # Source Nodes: [x_131, x_138], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf65, buf46, buf57, arg87_1, arg93_1, 32, 36864, grid=grid(32, 36864), stream=stream0)
        del arg87_1
        del arg93_1
        del buf46
        buf69 = reinterpret_tensor(buf57, (32, 256, 144), (36864, 144, 1), 0); del buf57  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf65, arg94_1, arg95_1, buf69, 8192, 144, grid=grid(8192), stream=stream0)
        del arg94_1
        del arg95_1
        buf70 = buf51; del buf51  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg97_1, reinterpret_tensor(buf69, (8192, 144), (144, 1), 0), reinterpret_tensor(arg96_1, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf70)
        del arg96_1
        del arg97_1
        # Source Nodes: [x_139], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf71 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf70, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf70, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf70, (32, 4, 256, 36), (110592, 36, 432, 1), 288), None, False)
        del buf70
        buf72 = buf71[0]
        del buf71
        buf76 = reinterpret_tensor(buf69, (8192, 144), (144, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (8192, 144), (144, 1), 0), reinterpret_tensor(arg98_1, (144, 144), (1, 144), 0), out=buf76)
        del arg98_1
        buf80 = reinterpret_tensor(buf72, (32, 256, 144), (36864, 144, 1), 0); del buf72  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2, x_143], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf65, buf76, arg99_1, arg100_1, arg101_1, buf80, 8192, 144, grid=grid(8192), stream=stream0)
        del arg100_1
        del arg101_1
        buf81 = reinterpret_tensor(buf63, (8192, 288), (288, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (8192, 144), (144, 1), 0), reinterpret_tensor(arg102_1, (144, 288), (1, 144), 0), out=buf81)
        del arg102_1
        buf82 = reinterpret_tensor(buf81, (32, 256, 288), (73728, 288, 1), 0); del buf81  # reuse
        # Source Nodes: [x_145], Original ATen: [aten.silu]
        triton_poi_fused_silu_14.run(buf82, arg103_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg103_1
        buf83 = reinterpret_tensor(buf80, (8192, 144), (144, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (8192, 288), (288, 1), 0), reinterpret_tensor(arg104_1, (288, 144), (1, 288), 0), out=buf83)
        del arg104_1
        del buf82
        buf84 = reinterpret_tensor(buf83, (32, 256, 144), (36864, 144, 1), 0); del buf83  # reuse
        buf85 = buf59; del buf59  # reuse
        buf86 = buf58; del buf58  # reuse
        # Source Nodes: [x_143, x_151, x_152], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_18.run(buf84, buf65, buf76, arg99_1, arg105_1, buf85, buf86, 8192, 144, grid=grid(8192), stream=stream0)
        del arg105_1
        del arg99_1
        del buf65
        buf88 = reinterpret_tensor(buf76, (8, 144, 32, 32), (147456, 1024, 32, 1), 0); del buf76  # reuse
        # Source Nodes: [x_156], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf84, buf85, buf86, arg106_1, arg107_1, buf88, 1152, 1024, grid=grid(1152, 1024), stream=stream0)
        del arg106_1
        del arg107_1
        del buf84
        del buf85
        del buf86
        # Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 96, 32, 32), (98304, 1024, 32, 1))
        del arg108_1
        buf90 = buf89; del buf89  # reuse
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf90, arg249_1, arg250_1, arg34_1, arg35_1, 786432, grid=grid(786432), stream=stream0)
        del arg249_1
        del arg250_1
        del arg34_1
        del arg35_1
        buf91 = empty((8, 192, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5, x_162], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_20.run(buf42, buf90, buf91, 1572864, grid=grid(1572864), stream=stream0)
        del buf42
        del buf90
        # Source Nodes: [cat_5, x_162], Original ATen: [aten.cat, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg109_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 96, 32, 32), (98304, 1024, 32, 1))
        del arg109_1
        del buf91
        buf93 = buf92; del buf92  # reuse
        buf94 = buf93; del buf93  # reuse
        # Source Nodes: [shortcut_6, x_163, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_10.run(buf94, arg251_1, arg252_1, arg36_1, arg37_1, 786432, grid=grid(786432), stream=stream0)
        del arg251_1
        del arg252_1
        del arg36_1
        del arg37_1
        # Source Nodes: [shortcut_6, x_168], Original ATen: [aten.convolution, aten.silu]
        buf95 = extern_kernels.convolution(buf94, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 384, 32, 32), (393216, 1024, 32, 1))
        del arg110_1
        del buf94
        buf96 = buf95; del buf95  # reuse
        buf97 = buf96; del buf96  # reuse
        # Source Nodes: [x_169, x_173, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_21.run(buf97, arg253_1, arg254_1, arg38_1, arg39_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg253_1
        del arg254_1
        del arg38_1
        del arg39_1
        # Source Nodes: [x_173, x_174], Original ATen: [aten.convolution, aten.silu]
        buf98 = extern_kernels.convolution(buf97, arg111_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf98, (8, 384, 16, 16), (98304, 256, 16, 1))
        del arg111_1
        del buf97
        buf99 = buf98; del buf98  # reuse
        buf100 = buf99; del buf99  # reuse
        # Source Nodes: [x_175, x_179, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_22.run(buf100, arg255_1, arg256_1, arg40_1, arg41_1, 786432, grid=grid(786432), stream=stream0)
        del arg255_1
        del arg256_1
        del arg40_1
        del arg41_1
        # Source Nodes: [x_179, x_182], Original ATen: [aten.convolution, aten.silu]
        buf101 = extern_kernels.convolution(buf100, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 128, 16, 16), (32768, 256, 16, 1))
        del arg112_1
        buf102 = buf101; del buf101  # reuse
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf102, arg257_1, arg258_1, arg42_1, arg43_1, 262144, grid=grid(262144), stream=stream0)
        del arg257_1
        del arg258_1
        del arg42_1
        del arg43_1
        # Source Nodes: [x_190], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg113_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 128, 16, 16), (32768, 256, 16, 1))
        del arg113_1
        buf104 = buf103; del buf103  # reuse
        buf105 = buf104; del buf104  # reuse
        # Source Nodes: [x_191, x_195, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_24.run(buf105, arg259_1, arg260_1, arg44_1, arg45_1, 262144, grid=grid(262144), stream=stream0)
        del arg259_1
        del arg260_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_195, x_196], Original ATen: [aten.convolution, aten.silu]
        buf106 = extern_kernels.convolution(buf105, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 192, 16, 16), (49152, 256, 16, 1))
        del arg114_1
        del buf105
        buf107 = empty_strided((32, 64, 1, 2), (1, 32, 4096, 2048), device='cuda', dtype=torch.float32)
        buf108 = empty_strided((32, 64, 1, 2), (1, 32, 4096, 2048), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((32, 64, 1, 2), (1, 32, 4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_25.run(buf106, buf107, buf108, buf109, 4096, 96, grid=grid(4096), stream=stream0)
        buf110 = empty_strided((32, 64, 1), (64, 1, 2048), device='cuda', dtype=torch.float32)
        buf111 = empty_strided((32, 64, 1), (1, 32, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf107, buf108, buf109, buf110, buf111, 2048, 2, grid=grid(2048), stream=stream0)
        buf113 = empty((32, 64, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_27.run(buf106, buf110, buf111, arg115_1, arg116_1, buf113, 32, 12288, grid=grid(32, 12288), stream=stream0)
        del arg115_1
        del arg116_1
        buf114 = reinterpret_tensor(buf88, (2048, 576), (576, 1), 0); del buf88  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg118_1, reinterpret_tensor(buf113, (2048, 192), (192, 1), 0), reinterpret_tensor(arg117_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf114)
        del arg117_1
        del arg118_1
        # Source Nodes: [x_199], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf115 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf114, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf114, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf114, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, False)
        buf116 = buf115[0]
        del buf115
        buf120 = reinterpret_tensor(buf113, (2048, 192), (192, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (2048, 192), (192, 1), 0), reinterpret_tensor(arg119_1, (192, 192), (1, 192), 0), out=buf120)
        del arg119_1
        buf121 = reinterpret_tensor(buf109, (32, 64, 1, 2), (128, 2, 4096, 1), 0); del buf109  # reuse
        buf122 = reinterpret_tensor(buf108, (32, 64, 1, 2), (128, 2, 4096, 1), 0); del buf108  # reuse
        buf123 = reinterpret_tensor(buf107, (32, 64, 1, 2), (128, 2, 4096, 1), 0); del buf107  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_28.run(buf106, buf120, arg120_1, buf121, buf122, buf123, 4096, 96, grid=grid(4096), stream=stream0)
        buf124 = reinterpret_tensor(buf111, (32, 64, 1), (64, 1, 2048), 0); del buf111  # reuse
        buf125 = buf110; del buf110  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_29.run(buf121, buf122, buf123, buf124, buf125, 2048, 2, grid=grid(2048), stream=stream0)
        del buf121
        del buf122
        del buf123
        buf127 = reinterpret_tensor(buf116, (32, 64, 192), (12288, 192, 1), 0); del buf116  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_30.run(buf106, buf120, arg120_1, buf124, buf125, arg121_1, arg122_1, buf127, 32, 12288, grid=grid(32, 12288), stream=stream0)
        del arg121_1
        del arg122_1
        buf128 = reinterpret_tensor(buf100, (2048, 384), (384, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (2048, 192), (192, 1), 0), reinterpret_tensor(arg123_1, (192, 384), (1, 192), 0), out=buf128)
        del arg123_1
        buf129 = reinterpret_tensor(buf128, (32, 64, 384), (24576, 384, 1), 0); del buf128  # reuse
        # Source Nodes: [x_205], Original ATen: [aten.silu]
        triton_poi_fused_silu_31.run(buf129, arg124_1, 786432, grid=grid(786432), stream=stream0)
        del arg124_1
        buf130 = reinterpret_tensor(buf127, (2048, 192), (192, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (2048, 384), (384, 1), 0), reinterpret_tensor(arg125_1, (384, 192), (1, 384), 0), out=buf130)
        del arg125_1
        buf131 = reinterpret_tensor(buf120, (32, 64, 192), (12288, 192, 1), 0); del buf120  # reuse
        # Source Nodes: [x_203, x_210], Original ATen: [aten.add]
        triton_poi_fused_add_32.run(buf131, buf106, arg120_1, buf130, arg126_1, 32, 12288, grid=grid(32, 12288), stream=stream0)
        del arg120_1
        del arg126_1
        buf135 = reinterpret_tensor(buf130, (32, 64, 192), (12288, 192, 1), 0); del buf130  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_33.run(buf131, arg127_1, arg128_1, buf135, 2048, 192, grid=grid(2048), stream=stream0)
        del arg127_1
        del arg128_1
        buf136 = buf114; del buf114  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg130_1, reinterpret_tensor(buf135, (2048, 192), (192, 1), 0), reinterpret_tensor(arg129_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf136)
        del arg129_1
        del arg130_1
        # Source Nodes: [x_211], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf137 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf136, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf136, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf136, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, False)
        buf138 = buf137[0]
        del buf137
        buf142 = reinterpret_tensor(buf135, (2048, 192), (192, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (2048, 192), (192, 1), 0), reinterpret_tensor(arg131_1, (192, 192), (1, 192), 0), out=buf142)
        del arg131_1
        buf146 = reinterpret_tensor(buf138, (32, 64, 192), (12288, 192, 1), 0); del buf138  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2, x_215], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_34.run(buf131, buf142, arg132_1, arg133_1, arg134_1, buf146, 2048, 192, grid=grid(2048), stream=stream0)
        del arg133_1
        del arg134_1
        buf147 = reinterpret_tensor(buf129, (2048, 384), (384, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (2048, 192), (192, 1), 0), reinterpret_tensor(arg135_1, (192, 384), (1, 192), 0), out=buf147)
        del arg135_1
        buf148 = reinterpret_tensor(buf147, (32, 64, 384), (24576, 384, 1), 0); del buf147  # reuse
        # Source Nodes: [x_217], Original ATen: [aten.silu]
        triton_poi_fused_silu_31.run(buf148, arg136_1, 786432, grid=grid(786432), stream=stream0)
        del arg136_1
        buf149 = reinterpret_tensor(buf146, (2048, 192), (192, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (2048, 384), (384, 1), 0), reinterpret_tensor(arg137_1, (384, 192), (1, 384), 0), out=buf149)
        del arg137_1
        buf150 = reinterpret_tensor(buf149, (32, 64, 192), (12288, 192, 1), 0); del buf149  # reuse
        buf154 = reinterpret_tensor(buf106, (32, 64, 192), (12288, 192, 1), 0); del buf106  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1, x_215, x_222], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_35.run(buf150, buf131, buf142, arg132_1, arg138_1, arg139_1, arg140_1, buf154, 2048, 192, grid=grid(2048), stream=stream0)
        del arg132_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf131
        buf155 = buf136; del buf136  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg142_1, reinterpret_tensor(buf154, (2048, 192), (192, 1), 0), reinterpret_tensor(arg141_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf155)
        del arg141_1
        del arg142_1
        # Source Nodes: [x_223], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf156 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf155, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf155, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf155, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, False)
        buf157 = buf156[0]
        del buf156
        buf161 = reinterpret_tensor(buf154, (2048, 192), (192, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (2048, 192), (192, 1), 0), reinterpret_tensor(arg143_1, (192, 192), (1, 192), 0), out=buf161)
        del arg143_1
        buf165 = reinterpret_tensor(buf157, (32, 64, 192), (12288, 192, 1), 0); del buf157  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm2, x_227], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_34.run(buf150, buf161, arg144_1, arg145_1, arg146_1, buf165, 2048, 192, grid=grid(2048), stream=stream0)
        del arg145_1
        del arg146_1
        buf166 = reinterpret_tensor(buf148, (2048, 384), (384, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (2048, 192), (192, 1), 0), reinterpret_tensor(arg147_1, (192, 384), (1, 192), 0), out=buf166)
        del arg147_1
        buf167 = reinterpret_tensor(buf166, (32, 64, 384), (24576, 384, 1), 0); del buf166  # reuse
        # Source Nodes: [x_229], Original ATen: [aten.silu]
        triton_poi_fused_silu_31.run(buf167, arg148_1, 786432, grid=grid(786432), stream=stream0)
        del arg148_1
        buf168 = reinterpret_tensor(buf165, (2048, 192), (192, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2048, 384), (384, 1), 0), reinterpret_tensor(arg149_1, (384, 192), (1, 384), 0), out=buf168)
        del arg149_1
        buf169 = reinterpret_tensor(buf168, (32, 64, 192), (12288, 192, 1), 0); del buf168  # reuse
        buf173 = reinterpret_tensor(buf142, (32, 64, 192), (12288, 192, 1), 0); del buf142  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm1, x_227, x_234], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_35.run(buf169, buf150, buf161, arg144_1, arg150_1, arg151_1, arg152_1, buf173, 2048, 192, grid=grid(2048), stream=stream0)
        del arg144_1
        del arg150_1
        del arg151_1
        del arg152_1
        del buf150
        del buf161
        buf174 = buf155; del buf155  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg154_1, reinterpret_tensor(buf173, (2048, 192), (192, 1), 0), reinterpret_tensor(arg153_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf174)
        del arg153_1
        del arg154_1
        # Source Nodes: [x_235], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf175 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf174, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf174, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf174, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, False)
        del buf174
        buf176 = buf175[0]
        del buf175
        buf180 = reinterpret_tensor(buf173, (2048, 192), (192, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (2048, 192), (192, 1), 0), reinterpret_tensor(arg155_1, (192, 192), (1, 192), 0), out=buf180)
        del arg155_1
        buf184 = reinterpret_tensor(buf176, (32, 64, 192), (12288, 192, 1), 0); del buf176  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm2, x_239], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_34.run(buf169, buf180, arg156_1, arg157_1, arg158_1, buf184, 2048, 192, grid=grid(2048), stream=stream0)
        del arg157_1
        del arg158_1
        buf185 = reinterpret_tensor(buf167, (2048, 384), (384, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (2048, 192), (192, 1), 0), reinterpret_tensor(arg159_1, (192, 384), (1, 192), 0), out=buf185)
        del arg159_1
        buf186 = reinterpret_tensor(buf185, (32, 64, 384), (24576, 384, 1), 0); del buf185  # reuse
        # Source Nodes: [x_241], Original ATen: [aten.silu]
        triton_poi_fused_silu_31.run(buf186, arg160_1, 786432, grid=grid(786432), stream=stream0)
        del arg160_1
        buf187 = reinterpret_tensor(buf184, (2048, 192), (192, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (2048, 384), (384, 1), 0), reinterpret_tensor(arg161_1, (384, 192), (1, 384), 0), out=buf187)
        del arg161_1
        del buf186
        buf188 = reinterpret_tensor(buf187, (32, 64, 192), (12288, 192, 1), 0); del buf187  # reuse
        buf189 = buf125; del buf125  # reuse
        buf190 = buf124; del buf124  # reuse
        # Source Nodes: [x_239, x_247, x_248], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_36.run(buf188, buf169, buf180, arg156_1, arg162_1, buf189, buf190, 2048, 192, grid=grid(2048), stream=stream0)
        del arg156_1
        del arg162_1
        del buf169
        buf192 = reinterpret_tensor(buf180, (8, 192, 16, 16), (49152, 256, 16, 1), 0); del buf180  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf188, buf189, buf190, arg163_1, arg164_1, buf192, 1536, 256, grid=grid(1536, 256), stream=stream0)
        del arg163_1
        del arg164_1
        del buf188
        del buf189
        del buf190
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 128, 16, 16), (32768, 256, 16, 1))
        del arg165_1
        del buf192
        buf194 = buf193; del buf193  # reuse
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf194, arg261_1, arg262_1, arg46_1, arg47_1, 262144, grid=grid(262144), stream=stream0)
        del arg261_1
        del arg262_1
        del arg46_1
        del arg47_1
        buf195 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_4, x_258], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_38.run(buf102, buf194, buf195, 524288, grid=grid(524288), stream=stream0)
        del buf102
        del buf194
        # Source Nodes: [cat_4, x_258], Original ATen: [aten.cat, aten.convolution]
        buf196 = extern_kernels.convolution(buf195, arg166_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 128, 16, 16), (32768, 256, 16, 1))
        del arg166_1
        del buf195
        buf197 = buf196; del buf196  # reuse
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [shortcut_8, x_259, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_24.run(buf198, arg263_1, arg264_1, arg48_1, arg49_1, 262144, grid=grid(262144), stream=stream0)
        del arg263_1
        del arg264_1
        del arg48_1
        del arg49_1
        # Source Nodes: [shortcut_8, x_264], Original ATen: [aten.convolution, aten.silu]
        buf199 = extern_kernels.convolution(buf198, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg167_1
        del buf198
        buf200 = buf199; del buf199  # reuse
        buf201 = buf200; del buf200  # reuse
        # Source Nodes: [x_265, x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_39.run(buf201, arg265_1, arg266_1, arg50_1, arg51_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg265_1
        del arg266_1
        del arg50_1
        del arg51_1
        # Source Nodes: [x_269, x_270], Original ATen: [aten.convolution, aten.silu]
        buf202 = extern_kernels.convolution(buf201, arg168_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf202, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg168_1
        del buf201
        buf203 = buf202; del buf202  # reuse
        buf204 = buf203; del buf203  # reuse
        # Source Nodes: [x_271, x_275, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_40.run(buf204, arg267_1, arg268_1, arg52_1, arg53_1, 262144, grid=grid(262144), stream=stream0)
        del arg267_1
        del arg268_1
        del arg52_1
        del arg53_1
        # Source Nodes: [x_275, x_278], Original ATen: [aten.convolution, aten.silu]
        buf205 = extern_kernels.convolution(buf204, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 160, 8, 8), (10240, 64, 8, 1))
        del arg169_1
        del buf204
        buf206 = buf205; del buf205  # reuse
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf206, arg269_1, arg270_1, arg54_1, arg55_1, 81920, grid=grid(81920), stream=stream0)
        del arg269_1
        del arg270_1
        del arg54_1
        del arg55_1
        # Source Nodes: [x_286], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg170_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 160, 8, 8), (10240, 64, 8, 1))
        del arg170_1
        buf208 = buf207; del buf207  # reuse
        buf209 = buf208; del buf208  # reuse
        # Source Nodes: [x_287, x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_42.run(buf209, arg271_1, arg272_1, arg56_1, arg57_1, 81920, grid=grid(81920), stream=stream0)
        del arg271_1
        del arg272_1
        del arg56_1
        del arg57_1
        # Source Nodes: [x_291, x_292], Original ATen: [aten.convolution, aten.silu]
        buf210 = extern_kernels.convolution(buf209, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 240, 8, 8), (15360, 64, 8, 1))
        del arg171_1
        del buf209
        buf211 = empty_strided((32, 16, 1, 2), (1, 32, 1024, 512), device='cuda', dtype=torch.float32)
        buf212 = empty_strided((32, 16, 1, 2), (1, 32, 1024, 512), device='cuda', dtype=torch.float32)
        buf213 = empty_strided((32, 16, 1, 2), (1, 32, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_43.run(buf210, buf211, buf212, buf213, 1024, 120, grid=grid(1024), stream=stream0)
        buf214 = empty_strided((32, 16, 1), (16, 1, 512), device='cuda', dtype=torch.float32)
        buf215 = empty_strided((32, 16, 1), (1, 32, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_44.run(buf211, buf212, buf213, buf214, buf215, 512, 2, grid=grid(512), stream=stream0)
        buf217 = empty((32, 16, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_45.run(buf210, buf214, buf215, arg172_1, arg173_1, buf217, 32, 3840, grid=grid(32, 3840), stream=stream0)
        del arg172_1
        del arg173_1
        buf218 = empty((512, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg175_1, reinterpret_tensor(buf217, (512, 240), (240, 1), 0), reinterpret_tensor(arg174_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf218)
        del arg174_1
        del arg175_1
        # Source Nodes: [x_295], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf219 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf218, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf218, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf218, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, False)
        buf220 = buf219[0]
        del buf219
        buf224 = reinterpret_tensor(buf217, (512, 240), (240, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 240), (240, 1), 0), reinterpret_tensor(arg176_1, (240, 240), (1, 240), 0), out=buf224)
        del arg176_1
        buf225 = reinterpret_tensor(buf213, (32, 16, 1, 2), (32, 2, 1024, 1), 0); del buf213  # reuse
        buf226 = reinterpret_tensor(buf212, (32, 16, 1, 2), (32, 2, 1024, 1), 0); del buf212  # reuse
        buf227 = reinterpret_tensor(buf211, (32, 16, 1, 2), (32, 2, 1024, 1), 0); del buf211  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_46.run(buf210, buf224, arg177_1, buf225, buf226, buf227, 1024, 120, grid=grid(1024), stream=stream0)
        buf228 = reinterpret_tensor(buf215, (32, 16, 1), (16, 1, 512), 0); del buf215  # reuse
        buf229 = buf214; del buf214  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf225, buf226, buf227, buf228, buf229, 512, 2, grid=grid(512), stream=stream0)
        del buf225
        del buf226
        del buf227
        buf231 = reinterpret_tensor(buf220, (32, 16, 240), (3840, 240, 1), 0); del buf220  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_48.run(buf210, buf224, arg177_1, buf228, buf229, arg178_1, arg179_1, buf231, 32, 3840, grid=grid(32, 3840), stream=stream0)
        del arg178_1
        del arg179_1
        buf232 = empty((512, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (512, 240), (240, 1), 0), reinterpret_tensor(arg180_1, (240, 480), (1, 240), 0), out=buf232)
        del arg180_1
        buf233 = reinterpret_tensor(buf232, (32, 16, 480), (7680, 480, 1), 0); del buf232  # reuse
        # Source Nodes: [x_301], Original ATen: [aten.silu]
        triton_poi_fused_silu_49.run(buf233, arg181_1, 245760, grid=grid(245760), stream=stream0)
        del arg181_1
        buf234 = reinterpret_tensor(buf231, (512, 240), (240, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (512, 480), (480, 1), 0), reinterpret_tensor(arg182_1, (480, 240), (1, 480), 0), out=buf234)
        del arg182_1
        buf235 = reinterpret_tensor(buf224, (32, 16, 240), (3840, 240, 1), 0); del buf224  # reuse
        # Source Nodes: [x_299, x_306], Original ATen: [aten.add]
        triton_poi_fused_add_50.run(buf235, buf210, arg177_1, buf234, arg183_1, 32, 3840, grid=grid(32, 3840), stream=stream0)
        del arg177_1
        del arg183_1
        buf239 = reinterpret_tensor(buf234, (32, 16, 240), (3840, 240, 1), 0); del buf234  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_51.run(buf235, arg184_1, arg185_1, buf239, 512, 240, grid=grid(512), stream=stream0)
        del arg184_1
        del arg185_1
        buf240 = buf218; del buf218  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg187_1, reinterpret_tensor(buf239, (512, 240), (240, 1), 0), reinterpret_tensor(arg186_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf240)
        del arg186_1
        del arg187_1
        # Source Nodes: [x_307], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf241 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf240, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf240, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf240, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, False)
        buf242 = buf241[0]
        del buf241
        buf246 = reinterpret_tensor(buf239, (512, 240), (240, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (512, 240), (240, 1), 0), reinterpret_tensor(arg188_1, (240, 240), (1, 240), 0), out=buf246)
        del arg188_1
        buf250 = reinterpret_tensor(buf242, (32, 16, 240), (3840, 240, 1), 0); del buf242  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2, x_311], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf235, buf246, arg189_1, arg190_1, arg191_1, buf250, 512, 240, grid=grid(512), stream=stream0)
        del arg190_1
        del arg191_1
        buf251 = reinterpret_tensor(buf233, (512, 480), (480, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (512, 240), (240, 1), 0), reinterpret_tensor(arg192_1, (240, 480), (1, 240), 0), out=buf251)
        del arg192_1
        buf252 = reinterpret_tensor(buf251, (32, 16, 480), (7680, 480, 1), 0); del buf251  # reuse
        # Source Nodes: [x_313], Original ATen: [aten.silu]
        triton_poi_fused_silu_49.run(buf252, arg193_1, 245760, grid=grid(245760), stream=stream0)
        del arg193_1
        buf253 = reinterpret_tensor(buf250, (512, 240), (240, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (512, 480), (480, 1), 0), reinterpret_tensor(arg194_1, (480, 240), (1, 480), 0), out=buf253)
        del arg194_1
        buf254 = reinterpret_tensor(buf253, (32, 16, 240), (3840, 240, 1), 0); del buf253  # reuse
        buf258 = reinterpret_tensor(buf210, (32, 16, 240), (3840, 240, 1), 0); del buf210  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1, x_311, x_318], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_53.run(buf254, buf235, buf246, arg189_1, arg195_1, arg196_1, arg197_1, buf258, 512, 240, grid=grid(512), stream=stream0)
        del arg189_1
        del arg195_1
        del arg196_1
        del arg197_1
        del buf235
        del buf246
        buf259 = buf240; del buf240  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg199_1, reinterpret_tensor(buf258, (512, 240), (240, 1), 0), reinterpret_tensor(arg198_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf259)
        del arg198_1
        del arg199_1
        # Source Nodes: [x_319], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf260 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf259, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf259, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf259, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, False)
        del buf259
        buf261 = buf260[0]
        del buf260
        buf265 = reinterpret_tensor(buf258, (512, 240), (240, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf261, (512, 240), (240, 1), 0), reinterpret_tensor(arg200_1, (240, 240), (1, 240), 0), out=buf265)
        del arg200_1
        buf269 = reinterpret_tensor(buf261, (32, 16, 240), (3840, 240, 1), 0); del buf261  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm2, x_323], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_52.run(buf254, buf265, arg201_1, arg202_1, arg203_1, buf269, 512, 240, grid=grid(512), stream=stream0)
        del arg202_1
        del arg203_1
        buf270 = reinterpret_tensor(buf252, (512, 480), (480, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (512, 240), (240, 1), 0), reinterpret_tensor(arg204_1, (240, 480), (1, 240), 0), out=buf270)
        del arg204_1
        buf271 = reinterpret_tensor(buf270, (32, 16, 480), (7680, 480, 1), 0); del buf270  # reuse
        # Source Nodes: [x_325], Original ATen: [aten.silu]
        triton_poi_fused_silu_49.run(buf271, arg205_1, 245760, grid=grid(245760), stream=stream0)
        del arg205_1
        buf272 = reinterpret_tensor(buf269, (512, 240), (240, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (512, 480), (480, 1), 0), reinterpret_tensor(arg206_1, (480, 240), (1, 480), 0), out=buf272)
        del arg206_1
        del buf271
        buf273 = reinterpret_tensor(buf272, (32, 16, 240), (3840, 240, 1), 0); del buf272  # reuse
        buf274 = buf229; del buf229  # reuse
        buf275 = buf228; del buf228  # reuse
        # Source Nodes: [x_323, x_331, x_332], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_54.run(buf273, buf254, buf265, arg201_1, arg207_1, buf274, buf275, 512, 240, grid=grid(512), stream=stream0)
        del arg201_1
        del arg207_1
        del buf254
        buf277 = reinterpret_tensor(buf265, (8, 240, 8, 8), (15360, 64, 8, 1), 0); del buf265  # reuse
        # Source Nodes: [x_336], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_55.run(buf273, buf274, buf275, arg208_1, arg209_1, buf277, 1920, 64, grid=grid(1920, 64), stream=stream0)
        del arg208_1
        del arg209_1
        del buf273
        del buf274
        del buf275
        # Source Nodes: [x_336], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 160, 8, 8), (10240, 64, 8, 1))
        del arg210_1
        del buf277
        buf279 = buf278; del buf278  # reuse
        # Source Nodes: [x_337], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf279, arg273_1, arg274_1, arg58_1, arg59_1, 81920, grid=grid(81920), stream=stream0)
        del arg273_1
        del arg274_1
        del arg58_1
        del arg59_1
        buf280 = empty((8, 320, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_3, x_342], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_56.run(buf206, buf279, buf280, 163840, grid=grid(163840), stream=stream0)
        del buf206
        del buf279
        # Source Nodes: [cat_3, x_342], Original ATen: [aten.cat, aten.convolution]
        buf281 = extern_kernels.convolution(buf280, arg211_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 160, 8, 8), (10240, 64, 8, 1))
        del arg211_1
        del buf280
        buf282 = buf281; del buf281  # reuse
        buf283 = buf282; del buf282  # reuse
        # Source Nodes: [x_343, x_348, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_42.run(buf283, arg275_1, arg276_1, arg60_1, arg61_1, 81920, grid=grid(81920), stream=stream0)
        del arg275_1
        del arg276_1
        del arg60_1
        del arg61_1
        # Source Nodes: [x_348, x_349], Original ATen: [aten.convolution, aten.silu]
        buf284 = extern_kernels.convolution(buf283, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg212_1
        del buf283
        buf285 = buf284; del buf284  # reuse
        buf286 = empty_strided((8, 640, 1, 1), (640, 1, 5120, 5120), device='cuda', dtype=torch.float32)
        buf287 = reinterpret_tensor(buf286, (8, 640, 1, 1), (640, 1, 1, 1), 0); del buf286  # reuse
        # Source Nodes: [x_350, x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_57.run(buf285, buf287, arg277_1, arg278_1, arg62_1, arg63_1, 5120, 64, grid=grid(5120), stream=stream0)
        del arg277_1
        del arg278_1
        del arg62_1
        del arg63_1
        del buf285
        buf288 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_360], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg214_1, reinterpret_tensor(buf287, (8, 640), (640, 1), 0), reinterpret_tensor(arg213_1, (640, 1000), (1, 640), 0), alpha=1, beta=1, out=buf288)
        del arg213_1
        del arg214_1
        return (buf288, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((144, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((432, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((144, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((288, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((144, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((432, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((144, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((288, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((144, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((96, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((96, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((160, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((160, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((160, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1000, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilevit_s', benchmark_compiled_module)
