
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


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4iyzq5k3afr2yb5ks64ma7rjy2vkfsxgkjzzddecv7ytpvgda5.py
# Source Nodes: [l__mod___stem_1, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___stem_1 => add_1, mul_1, mul_2, sub
# x => relu
# x_1 => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
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


# kernel path: /tmp/torchinductor_youkaichao/wy/cwye3ieoxtzh6t4wvgofu4r2mrnrififbof6a7akfufq7tckpui5.py
# Source Nodes: [add, getattr_l__mod___stage1___0___norm2, l__mod___stem_1, x, x_1, x_3, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# add => add_4
# getattr_l__mod___stage1___0___norm2 => add_6, mul_7, mul_8, sub_2
# l__mod___stem_1 => add_1, mul_1, mul_2, sub
# x => relu
# x_1 => convolution_1
# x_3 => add_3, mul_4, mul_5, sub_1
# x_5 => convolution_2
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    x4 = xindex % 150528
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp6
    tmp23 = tl.sqrt(tmp22)
    tmp24 = 1 / tmp23
    tmp25 = tmp24 * tmp10
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3l/c3lfug7sptadqizniwbafqj677xkdnezym4runsve6buhvl3v7ng.py
# Source Nodes: [x_6, x_8], Original ATen: [aten.convolution, aten.gelu]
# x_6 => add_7, erf, mul_10, mul_11, mul_9
# x_8 => convolution_3
triton_poi_fused_convolution_gelu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5v3zuqhenvzzmdtx4hmehphfbzafv3vkee4f733or7a5o7yevpt.py
# Source Nodes: [getattr_l__mod___stage1___1___norm2, x_12, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage1___1___norm2 => add_11, mul_16, mul_17, sub_3
# x_12 => add_9
# x_13 => convolution_5
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceulch5ex2cvy2hvdctyiticixegucxvm6lddc743f2cdtaj3k7i.py
# Source Nodes: [getattr_l__mod___stage1___2___norm2, x_12, x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage1___2___norm2 => add_16, mul_25, mul_26, sub_4
# x_12 => add_9
# x_20 => add_14
# x_21 => convolution_8
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sqrt(tmp9)
    tmp11 = 1 / tmp10
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp6 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tl.store(out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdqc3s7fajz3skbyacaauxur7jzxvi6ifjdhz6kjfh2p6ofllhc.py
# Source Nodes: [getattr_l__mod___stage1___3___norm2, x_12, x_20, x_28, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage1___3___norm2 => add_21, mul_34, mul_35, sub_5
# x_12 => add_9
# x_20 => add_14
# x_28 => add_19
# x_29 => convolution_11
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.sqrt(tmp11)
    tmp13 = 1 / tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp8 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxaspbra637bvfeds4teihznn2z5vwuckce2m53icmdaeextfb4.py
# Source Nodes: [getattr_l__mod___stage1___4___norm2, x_12, x_20, x_28, x_36, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage1___4___norm2 => add_26, mul_43, mul_44, sub_6
# x_12 => add_9
# x_20 => add_14
# x_28 => add_19
# x_36 => add_24
# x_37 => convolution_14
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jx/cjx4jvlk4pc6q2sckzwvwlmnfcvn5v7blokhrqw5ueyjjqph64qr.py
# Source Nodes: [x_44, x_52, x_61, x_62], Original ATen: [aten.add, aten.convolution]
# x_44 => add_29
# x_52 => add_34
# x_61 => add_39
# x_62 => convolution_23
triton_poi_fused_add_convolution_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chprbroumre73ka2ony3zyzymcddkdoie4xzufcyafiqx5u72rz3.py
# Source Nodes: [add_8, getattr_l__mod___stage2___0___attn_qkv, getattr_l__mod___stage2___0___norm1, x_44, x_52, x_61, x_62, x_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# add_8 => add_42
# getattr_l__mod___stage2___0___attn_qkv => convolution_24
# getattr_l__mod___stage2___0___norm1 => add_44, mul_73, mul_74, sub_10
# x_44 => add_29
# x_52 => add_34
# x_61 => add_39
# x_62 => convolution_23
# x_64 => add_41, mul_70, mul_71, sub_9
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    x4 = xindex % 75264
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp6
    tmp23 = tl.sqrt(tmp22)
    tmp24 = 1 / tmp23
    tmp25 = tmp24 * tmp10
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpxn77b4vtosolz2vhkptwrhwb6qt27og3nw3cmwqqhkstdfwp7.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_16
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9408
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 6
    y2 = (yindex // 1176)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x3) + (12544*y1) + (225792*y2)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (64*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clv2i33li5yvdqdoikqtnqk2mowmgvg3dyvg2uqijbuiemvwo5wm.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_17
triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 75264
    x1 = (xindex // 75264)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (75264 + x0 + (225792*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxg62vwha2qhe2zx4vfyo6k6nn2w2vywundy3kb2ld74zjhjyw4p.py
# Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
# attn => mul_75
# attn_1 => amax, div, exp, sub_11, sum_1
triton_per_fused__softmax_mul_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ri/crif2ugh2ejce6vxjjeqhovflotgnzs22euilqvfgfs4spd2zmnr.py
# Source Nodes: [x_67], Original ATen: [aten.clone]
# x_67 => clone_19
triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9408
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 6
    y2 = (yindex // 1176)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (150528 + y0 + (196*x3) + (12544*y1) + (225792*y2)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (64*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4smauqyw36a6jr5vxpkt53yo6ji2o3snlhtoowwqsdg76oieno.py
# Source Nodes: [x_69], Original ATen: [aten.convolution]
# x_69 => convolution_25
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + ((64*x2) + (12544*(y0 // 64)) + (75264*y1) + (y0 % 64)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2rv6izwrsrumauquig6sklgbk4iyotp7b4kis2cwgaifaovbsm.py
# Source Nodes: [getattr_l__mod___stage2___0___norm2, x_71, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage2___0___norm2 => add_47, mul_77, mul_78, sub_12
# x_71 => add_45
# x_72 => convolution_26
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6t3xspjg3jwgrkyrxkk7irzrbkwtyjcds24fwsylhe7tn5sk26.py
# Source Nodes: [getattr_l__mod___stage2___1___attn_qkv, getattr_l__mod___stage2___1___norm1, x_71, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage2___1___attn_qkv => convolution_28
# getattr_l__mod___stage2___1___norm1 => add_51, mul_83, mul_84, sub_13
# x_71 => add_45
# x_77 => add_49
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sqrt(tmp9)
    tmp11 = 1 / tmp10
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp6 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tl.store(out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7nvt7okyjepszu5qthyf6hpf4q3wkqv5ryocodspmmdfopt3ll.py
# Source Nodes: [getattr_l__mod___stage2___1___norm2, x_71, x_77, x_83, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage2___1___norm2 => add_54, mul_87, mul_88, sub_15
# x_71 => add_45
# x_77 => add_49
# x_83 => add_52
# x_84 => convolution_30
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.sqrt(tmp11)
    tmp13 = 1 / tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp8 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4ndnmnhkmdypovcahxj3jyaw6oxi5uhh4spj3v4vj42jn5dtzou.py
# Source Nodes: [getattr_l__mod___stage2___2___attn_qkv, getattr_l__mod___stage2___2___norm1, x_71, x_77, x_83, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage2___2___attn_qkv => convolution_32
# getattr_l__mod___stage2___2___norm1 => add_58, mul_93, mul_94, sub_16
# x_71 => add_45
# x_77 => add_49
# x_83 => add_52
# x_89 => add_56
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7vt4uznmwnay2y3u5njaurznz5z2p7kihlyyjg3iwbwk7agjsn.py
# Source Nodes: [x_101, x_107, x_114, x_115, x_95], Original ATen: [aten.add, aten.convolution]
# x_101 => add_63
# x_107 => add_66
# x_114 => add_70
# x_115 => convolution_40
# x_95 => add_59
triton_poi_fused_add_convolution_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdo6guuolysdjz5ypfrrfmubj73xi6feni7qnhlldq5zfekiyduc.py
# Source Nodes: [add_17, getattr_l__mod___stage3___0___attn_qkv, getattr_l__mod___stage3___0___norm1, x_101, x_107, x_114, x_115, x_117, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# add_17 => add_73
# getattr_l__mod___stage3___0___attn_qkv => convolution_41
# getattr_l__mod___stage3___0___norm1 => add_75, mul_116, mul_117, sub_23
# x_101 => add_63
# x_107 => add_66
# x_114 => add_70
# x_115 => convolution_40
# x_117 => add_72, mul_113, mul_114, sub_22
# x_95 => add_59
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    x4 = xindex % 37632
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp6
    tmp23 = tl.sqrt(tmp22)
    tmp24 = 1 / tmp23
    tmp25 = tmp24 * tmp10
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhv7jvqzutneobigtw3nt7eze6ht4s52bl5vklpift7uiadzm7c.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_49
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2352
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49) % 6
    y2 = (yindex // 294)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x3) + (6272*y1) + (112896*y2)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (128*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5trwt7e2jafns2f6klqgavtnvk33tgpuqo5jyc224fs7c6lo27l.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_50
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 37632
    x1 = (xindex // 37632)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (37632 + x0 + (112896*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7a/c7aqedcdlgfajqg6wyy32jftfepi6xn4wgzoshezbjdw4gje6pfk.py
# Source Nodes: [attn_12, attn_13], Original ATen: [aten._softmax, aten.mul]
# attn_12 => mul_118
# attn_13 => amax_4, div_4, exp_4, sub_24, sum_5
triton_per_fused__softmax_mul_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (49*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rayaale56lrudwtcj6vri7dmoc3o6nup6k2wsoznz7z6h3k6ry.py
# Source Nodes: [x_120], Original ATen: [aten.clone]
# x_120 => clone_52
triton_poi_fused_clone_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2352
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49) % 6
    y2 = (yindex // 294)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (75264 + y0 + (49*x3) + (6272*y1) + (112896*y2)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (128*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43sl2wggdo4ttr5m7tgccu443wrox76gblekkqpc2jr3canf7af.py
# Source Nodes: [x_122], Original ATen: [aten.convolution]
# x_122 => convolution_42
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + ((128*x2) + (6272*(y0 // 128)) + (37632*y1) + (y0 % 128)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g3/cg3boneiifkvr2pkqf4lrlgj5ycdsvwze2who667zlfhatcup2co.py
# Source Nodes: [getattr_l__mod___stage3___0___norm2, x_124, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage3___0___norm2 => add_78, mul_120, mul_121, sub_25
# x_124 => add_76
# x_125 => convolution_43
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdgb3cc26ozguqlucligutnj5uysxxbcvmcgk7fiha2bk45yv6do.py
# Source Nodes: [x_126, x_128], Original ATen: [aten.convolution, aten.gelu]
# x_126 => add_79, erf_18, mul_122, mul_123, mul_124
# x_128 => convolution_44
triton_poi_fused_convolution_gelu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpok3yjh2lhx4fwrzidsjphzycpcuo7ycr2r6fnmumohhgkrfo7.py
# Source Nodes: [getattr_l__mod___stage3___1___attn_qkv, getattr_l__mod___stage3___1___norm1, x_124, x_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage3___1___attn_qkv => convolution_45
# getattr_l__mod___stage3___1___norm1 => add_82, mul_126, mul_127, sub_26
# x_124 => add_76
# x_130 => add_80
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sqrt(tmp9)
    tmp11 = 1 / tmp10
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp6 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tl.store(out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z6/cz6zkfsdfw6aiutu2n5jbt3gclt5v633t6nljt6mawkfs3ge4rt3.py
# Source Nodes: [getattr_l__mod___stage3___1___norm2, x_124, x_130, x_136, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage3___1___norm2 => add_85, mul_130, mul_131, sub_28
# x_124 => add_76
# x_130 => add_80
# x_136 => add_83
# x_137 => convolution_47
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.sqrt(tmp11)
    tmp13 = 1 / tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp8 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/csphfbbiguomtghdqzbzlglvabloxf5nwfc72henrvhqsxaxwt6l.py
# Source Nodes: [getattr_l__mod___stage3___2___attn_qkv, getattr_l__mod___stage3___2___norm1, x_124, x_130, x_136, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
# getattr_l__mod___stage3___2___attn_qkv => convolution_49
# getattr_l__mod___stage3___2___norm1 => add_89, mul_136, mul_137, sub_29
# x_124 => add_76
# x_130 => add_80
# x_136 => add_83
# x_142 => add_87
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2n/c2n5jcixst66rwx2cguldo2u3fwlmro4wit4vhphorsqqa3o66b5.py
# Source Nodes: [x_148, x_154, x_160, x_167, x_169, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
# x_148 => add_90
# x_154 => add_94
# x_160 => add_97
# x_167 => add_101
# x_169 => add_103, mul_156, mul_157, sub_35
# x_170 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (49*x3)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r2 + (49*x3)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (49*x3)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 49.0
    tmp28 = tmp26 / tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(arg1_1, (1, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(arg2_1, (1, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(arg3_1, (32, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (192, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(arg7_1, (192, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (192, ), (1, ))
    assert_size_stride(arg10_1, (192, ), (1, ))
    assert_size_stride(arg11_1, (192, ), (1, ))
    assert_size_stride(arg12_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg13_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg14_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg15_1, (192, ), (1, ))
    assert_size_stride(arg16_1, (192, ), (1, ))
    assert_size_stride(arg17_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg18_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg19_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg20_1, (192, ), (1, ))
    assert_size_stride(arg21_1, (192, ), (1, ))
    assert_size_stride(arg22_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg23_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg24_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg28_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg29_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg33_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg34_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg35_1, (192, ), (1, ))
    assert_size_stride(arg36_1, (192, ), (1, ))
    assert_size_stride(arg37_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg38_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg39_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg40_1, (192, ), (1, ))
    assert_size_stride(arg41_1, (192, ), (1, ))
    assert_size_stride(arg42_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg43_1, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg44_1, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg45_1, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(arg46_1, (384, ), (1, ))
    assert_size_stride(arg47_1, (384, ), (1, ))
    assert_size_stride(arg48_1, (384, ), (1, ))
    assert_size_stride(arg49_1, (384, ), (1, ))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg52_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg56_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (384, ), (1, ))
    assert_size_stride(arg59_1, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg60_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg61_1, (384, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg64_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg68_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (384, ), (1, ))
    assert_size_stride(arg71_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg72_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg76_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg80_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg81_1, (768, 384, 2, 2), (1536, 4, 2, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg88_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg92_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg96_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg100_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg104_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg108_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg112_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg116_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (1000, 768), (768, 1))
    assert_size_stride(arg120_1, (1000, ), (1, ))
    assert_size_stride(arg121_1, (32, ), (1, ))
    assert_size_stride(arg122_1, (32, ), (1, ))
    assert_size_stride(arg123_1, (), ())
    assert_size_stride(arg124_1, (192, ), (1, ))
    assert_size_stride(arg125_1, (192, ), (1, ))
    assert_size_stride(arg126_1, (), ())
    assert_size_stride(arg127_1, (192, ), (1, ))
    assert_size_stride(arg128_1, (192, ), (1, ))
    assert_size_stride(arg129_1, (), ())
    assert_size_stride(arg130_1, (192, ), (1, ))
    assert_size_stride(arg131_1, (192, ), (1, ))
    assert_size_stride(arg132_1, (), ())
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, ), (1, ))
    assert_size_stride(arg135_1, (), ())
    assert_size_stride(arg136_1, (192, ), (1, ))
    assert_size_stride(arg137_1, (192, ), (1, ))
    assert_size_stride(arg138_1, (), ())
    assert_size_stride(arg139_1, (192, ), (1, ))
    assert_size_stride(arg140_1, (192, ), (1, ))
    assert_size_stride(arg141_1, (), ())
    assert_size_stride(arg142_1, (192, ), (1, ))
    assert_size_stride(arg143_1, (192, ), (1, ))
    assert_size_stride(arg144_1, (), ())
    assert_size_stride(arg145_1, (192, ), (1, ))
    assert_size_stride(arg146_1, (192, ), (1, ))
    assert_size_stride(arg147_1, (), ())
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (), ())
    assert_size_stride(arg151_1, (384, ), (1, ))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (), ())
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (), ())
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (), ())
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (), ())
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (384, ), (1, ))
    assert_size_stride(arg165_1, (), ())
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (), ())
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (384, ), (1, ))
    assert_size_stride(arg171_1, (), ())
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (), ())
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (), ())
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (), ())
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (), ())
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (), ())
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (), ())
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (), ())
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (), ())
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (), ())
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (), ())
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (), ())
    assert_size_stride(arg205_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg205_1, arg3_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg205_1
        del arg3_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__mod___stem_1, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg121_1, arg122_1, arg4_1, arg5_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg121_1
        del arg122_1
        del arg4_1
        del arg5_1
        # Source Nodes: [l__mod___stem_1, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf2 = extern_kernels.convolution(buf1, arg6_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg6_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, getattr_l__mod___stage1___0___norm2, l__mod___stem_1, x, x_1, x_3, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_1.run(buf3, arg7_1, arg124_1, arg125_1, arg8_1, arg9_1, arg0_1, arg127_1, arg128_1, arg10_1, arg11_1, buf4, 1204224, grid=grid(1204224), stream=stream0)
        del arg0_1
        del arg10_1
        del arg11_1
        del arg124_1
        del arg125_1
        del arg127_1
        del arg128_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Source Nodes: [getattr_l__mod___stage1___0___norm2, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg12_1
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [x_6, x_8], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf6, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_6, x_8], Original ATen: [aten.convolution, aten.gelu]
        buf7 = extern_kernels.convolution(buf6, arg13_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf7, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg13_1
        del buf6
        buf8 = buf7; del buf7  # reuse
        # Source Nodes: [x_10, x_9], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf8, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_10, x_9], Original ATen: [aten.convolution, aten.gelu]
        buf9 = extern_kernels.convolution(buf8, arg14_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg14_1
        del buf8
        buf10 = buf4; del buf4  # reuse
        # Source Nodes: [getattr_l__mod___stage1___1___norm2, x_12, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3.run(buf3, buf9, arg130_1, arg131_1, arg15_1, arg16_1, buf10, 1204224, grid=grid(1204224), stream=stream0)
        del arg130_1
        del arg131_1
        del arg15_1
        del arg16_1
        # Source Nodes: [getattr_l__mod___stage1___1___norm2, x_12, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg17_1
        buf12 = buf11; del buf11  # reuse
        # Source Nodes: [x_14, x_16], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf12, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_14, x_16], Original ATen: [aten.convolution, aten.gelu]
        buf13 = extern_kernels.convolution(buf12, arg18_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf13, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg18_1
        del buf12
        buf14 = buf13; del buf13  # reuse
        # Source Nodes: [x_17, x_18], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf14, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_17, x_18], Original ATen: [aten.convolution, aten.gelu]
        buf15 = extern_kernels.convolution(buf14, arg19_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg19_1
        del buf14
        buf16 = buf10; del buf10  # reuse
        # Source Nodes: [getattr_l__mod___stage1___2___norm2, x_12, x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_4.run(buf3, buf9, buf15, arg133_1, arg134_1, arg20_1, arg21_1, buf16, 1204224, grid=grid(1204224), stream=stream0)
        del arg133_1
        del arg134_1
        del arg20_1
        del arg21_1
        # Source Nodes: [getattr_l__mod___stage1___2___norm2, x_12, x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg22_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg22_1
        buf18 = buf17; del buf17  # reuse
        # Source Nodes: [x_22, x_24], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf18, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_22, x_24], Original ATen: [aten.convolution, aten.gelu]
        buf19 = extern_kernels.convolution(buf18, arg23_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf19, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg23_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Source Nodes: [x_25, x_26], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf20, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_25, x_26], Original ATen: [aten.convolution, aten.gelu]
        buf21 = extern_kernels.convolution(buf20, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg24_1
        del buf20
        buf22 = buf16; del buf16  # reuse
        # Source Nodes: [getattr_l__mod___stage1___3___norm2, x_12, x_20, x_28, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_5.run(buf3, buf9, buf15, buf21, arg136_1, arg137_1, arg25_1, arg26_1, buf22, 1204224, grid=grid(1204224), stream=stream0)
        del arg136_1
        del arg137_1
        del arg25_1
        del arg26_1
        # Source Nodes: [getattr_l__mod___stage1___3___norm2, x_12, x_20, x_28, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg27_1
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [x_30, x_32], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf24, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_30, x_32], Original ATen: [aten.convolution, aten.gelu]
        buf25 = extern_kernels.convolution(buf24, arg28_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf25, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg28_1
        del buf24
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [x_33, x_34], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf26, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_33, x_34], Original ATen: [aten.convolution, aten.gelu]
        buf27 = extern_kernels.convolution(buf26, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg29_1
        del buf26
        buf28 = buf15; del buf15  # reuse
        buf29 = buf22; del buf22  # reuse
        # Source Nodes: [getattr_l__mod___stage1___4___norm2, x_12, x_20, x_28, x_36, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_6.run(buf28, buf3, buf9, buf21, buf27, arg139_1, arg140_1, arg30_1, arg31_1, buf29, 1204224, grid=grid(1204224), stream=stream0)
        del arg139_1
        del arg140_1
        del arg30_1
        del arg31_1
        del buf21
        del buf27
        del buf3
        del buf9
        # Source Nodes: [getattr_l__mod___stage1___4___norm2, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg32_1
        buf31 = buf30; del buf30  # reuse
        # Source Nodes: [x_38, x_40], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf31, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_38, x_40], Original ATen: [aten.convolution, aten.gelu]
        buf32 = extern_kernels.convolution(buf31, arg33_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf32, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg33_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [x_41, x_42], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf33, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_41, x_42], Original ATen: [aten.convolution, aten.gelu]
        buf34 = extern_kernels.convolution(buf33, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg34_1
        del buf33
        buf35 = buf29; del buf29  # reuse
        # Source Nodes: [getattr_l__mod___stage1___5___norm2, x_44, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_3.run(buf28, buf34, arg142_1, arg143_1, arg35_1, arg36_1, buf35, 1204224, grid=grid(1204224), stream=stream0)
        del arg142_1
        del arg143_1
        del arg35_1
        del arg36_1
        # Source Nodes: [getattr_l__mod___stage1___5___norm2, x_44, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg37_1
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [x_46, x_48], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf37, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_46, x_48], Original ATen: [aten.convolution, aten.gelu]
        buf38 = extern_kernels.convolution(buf37, arg38_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf38, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg38_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf39, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten.gelu]
        buf40 = extern_kernels.convolution(buf39, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg39_1
        del buf39
        buf41 = buf35; del buf35  # reuse
        # Source Nodes: [getattr_l__mod___stage1___6___norm2, x_44, x_52, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_4.run(buf28, buf34, buf40, arg145_1, arg146_1, arg40_1, arg41_1, buf41, 1204224, grid=grid(1204224), stream=stream0)
        del arg145_1
        del arg146_1
        del arg40_1
        del arg41_1
        # Source Nodes: [getattr_l__mod___stage1___6___norm2, x_44, x_52, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg42_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [x_54, x_56], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf43, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_54, x_56], Original ATen: [aten.convolution, aten.gelu]
        buf44 = extern_kernels.convolution(buf43, arg43_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf44, (8, 384, 28, 28), (301056, 784, 28, 1))
        del arg43_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        # Source Nodes: [x_57, x_58], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf45, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_57, x_58], Original ATen: [aten.convolution, aten.gelu]
        buf46 = extern_kernels.convolution(buf45, arg44_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg44_1
        del buf45
        buf47 = buf28; del buf28  # reuse
        # Source Nodes: [x_44, x_52, x_61, x_62], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_add_convolution_7.run(buf47, buf34, buf40, buf46, 1204224, grid=grid(1204224), stream=stream0)
        del buf34
        del buf40
        del buf46
        # Source Nodes: [x_44, x_52, x_61, x_62], Original ATen: [aten.add, aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg45_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg45_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        buf50 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, getattr_l__mod___stage2___0___attn_qkv, getattr_l__mod___stage2___0___norm1, x_44, x_52, x_61, x_62, x_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_8.run(buf49, arg46_1, arg148_1, arg149_1, arg47_1, arg48_1, arg1_1, arg151_1, arg152_1, arg49_1, arg50_1, buf50, 602112, grid=grid(602112), stream=stream0)
        del arg148_1
        del arg149_1
        del arg151_1
        del arg152_1
        del arg1_1
        del arg46_1
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        # Source Nodes: [getattr_l__mod___stage2___0___attn_qkv, getattr_l__mod___stage2___0___norm1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 1152, 14, 14), (225792, 196, 14, 1))
        del arg51_1
        buf52 = reinterpret_tensor(buf50, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf50  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf51, buf52, 9408, 64, grid=grid(9408, 64), stream=stream0)
        buf53 = empty((8, 6, 64, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf51, buf53, 602112, grid=grid(602112), stream=stream0)
        buf54 = empty((48, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf52, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf53, (48, 64, 196), (12544, 196, 1), 0), out=buf54)
        buf57 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_11.run(buf54, buf57, 9408, 196, grid=grid(9408), stream=stream0)
        buf58 = reinterpret_tensor(buf53, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [x_67], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf51, buf58, 9408, 64, grid=grid(9408, 64), stream=stream0)
        del buf51
        buf59 = reinterpret_tensor(buf52, (48, 196, 64), (12544, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [x_67], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf58, (48, 196, 64), (12544, 64, 1), 0), out=buf59)
        buf60 = reinterpret_tensor(buf58, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf58  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf59, buf60, 3072, 196, grid=grid(3072, 196), stream=stream0)
        # Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg52_1
        buf62 = buf60; del buf60  # reuse
        # Source Nodes: [getattr_l__mod___stage2___0___norm2, x_71, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14.run(buf49, buf61, arg154_1, arg155_1, arg53_1, arg54_1, buf62, 602112, grid=grid(602112), stream=stream0)
        del arg154_1
        del arg155_1
        del arg53_1
        del arg54_1
        # Source Nodes: [getattr_l__mod___stage2___0___norm2, x_71, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg55_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg55_1
        buf64 = buf63; del buf63  # reuse
        # Source Nodes: [x_73, x_75], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf64, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_73, x_75], Original ATen: [aten.convolution, aten.gelu]
        buf65 = extern_kernels.convolution(buf64, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg56_1
        del buf64
        buf66 = buf62; del buf62  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___attn_qkv, getattr_l__mod___stage2___1___norm1, x_71, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15.run(buf49, buf61, buf65, arg157_1, arg158_1, arg57_1, arg58_1, buf66, 602112, grid=grid(602112), stream=stream0)
        del arg157_1
        del arg158_1
        del arg57_1
        del arg58_1
        # Source Nodes: [getattr_l__mod___stage2___1___attn_qkv, getattr_l__mod___stage2___1___norm1, x_71, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg59_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 1152, 14, 14), (225792, 196, 14, 1))
        del arg59_1
        buf68 = reinterpret_tensor(buf66, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf66  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf67, buf68, 9408, 64, grid=grid(9408, 64), stream=stream0)
        buf69 = reinterpret_tensor(buf59, (8, 6, 64, 196), (75264, 12544, 196, 1), 0); del buf59  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf67, buf69, 602112, grid=grid(602112), stream=stream0)
        buf70 = reinterpret_tensor(buf57, (48, 196, 196), (38416, 196, 1), 0); del buf57  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf69, (48, 64, 196), (12544, 196, 1), 0), out=buf70)
        buf73 = reinterpret_tensor(buf54, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf54  # reuse
        # Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_11.run(buf70, buf73, 9408, 196, grid=grid(9408), stream=stream0)
        buf74 = reinterpret_tensor(buf69, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf69  # reuse
        # Source Nodes: [x_79], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf67, buf74, 9408, 64, grid=grid(9408, 64), stream=stream0)
        del buf67
        buf75 = reinterpret_tensor(buf68, (48, 196, 64), (12544, 64, 1), 0); del buf68  # reuse
        # Source Nodes: [x_79], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf74, (48, 196, 64), (12544, 64, 1), 0), out=buf75)
        buf76 = reinterpret_tensor(buf74, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf74  # reuse
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf75, buf76, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del buf75
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg60_1
        buf78 = buf76; del buf76  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___norm2, x_71, x_77, x_83, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_16.run(buf49, buf61, buf65, buf77, arg160_1, arg161_1, arg61_1, arg62_1, buf78, 602112, grid=grid(602112), stream=stream0)
        del arg160_1
        del arg161_1
        del arg61_1
        del arg62_1
        # Source Nodes: [getattr_l__mod___stage2___1___norm2, x_71, x_77, x_83, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg63_1
        buf80 = buf79; del buf79  # reuse
        # Source Nodes: [x_85, x_87], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf80, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_85, x_87], Original ATen: [aten.convolution, aten.gelu]
        buf81 = extern_kernels.convolution(buf80, arg64_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg64_1
        del buf80
        buf82 = buf49; del buf49  # reuse
        buf83 = buf78; del buf78  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___attn_qkv, getattr_l__mod___stage2___2___norm1, x_71, x_77, x_83, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_17.run(buf82, buf61, buf65, buf77, buf81, arg163_1, arg164_1, arg65_1, arg66_1, buf83, 602112, grid=grid(602112), stream=stream0)
        del arg163_1
        del arg164_1
        del arg65_1
        del arg66_1
        del buf61
        del buf65
        del buf77
        # Source Nodes: [getattr_l__mod___stage2___2___attn_qkv, getattr_l__mod___stage2___2___norm1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg67_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 1152, 14, 14), (225792, 196, 14, 1))
        del arg67_1
        buf85 = reinterpret_tensor(buf83, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf84, buf85, 9408, 64, grid=grid(9408, 64), stream=stream0)
        buf86 = reinterpret_tensor(buf81, (8, 6, 64, 196), (75264, 12544, 196, 1), 0); del buf81  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf84, buf86, 602112, grid=grid(602112), stream=stream0)
        buf87 = reinterpret_tensor(buf73, (48, 196, 196), (38416, 196, 1), 0); del buf73  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf86, (48, 64, 196), (12544, 196, 1), 0), out=buf87)
        buf90 = reinterpret_tensor(buf70, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf70  # reuse
        # Source Nodes: [attn_6, attn_7], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_11.run(buf87, buf90, 9408, 196, grid=grid(9408), stream=stream0)
        buf91 = reinterpret_tensor(buf86, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf86  # reuse
        # Source Nodes: [x_91], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf84, buf91, 9408, 64, grid=grid(9408, 64), stream=stream0)
        del buf84
        buf92 = reinterpret_tensor(buf85, (48, 196, 64), (12544, 64, 1), 0); del buf85  # reuse
        # Source Nodes: [x_91], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf91, (48, 196, 64), (12544, 64, 1), 0), out=buf92)
        buf93 = reinterpret_tensor(buf91, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf91  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf92, buf93, 3072, 196, grid=grid(3072, 196), stream=stream0)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg68_1
        buf95 = buf93; del buf93  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___norm2, x_95, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_14.run(buf82, buf94, arg166_1, arg167_1, arg69_1, arg70_1, buf95, 602112, grid=grid(602112), stream=stream0)
        del arg166_1
        del arg167_1
        del arg69_1
        del arg70_1
        # Source Nodes: [getattr_l__mod___stage2___2___norm2, x_95, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg71_1
        buf97 = buf96; del buf96  # reuse
        # Source Nodes: [x_97, x_99], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf97, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_97, x_99], Original ATen: [aten.convolution, aten.gelu]
        buf98 = extern_kernels.convolution(buf97, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg72_1
        del buf97
        buf99 = buf95; del buf95  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___attn_qkv, getattr_l__mod___stage2___3___norm1, x_101, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_15.run(buf82, buf94, buf98, arg169_1, arg170_1, arg73_1, arg74_1, buf99, 602112, grid=grid(602112), stream=stream0)
        del arg169_1
        del arg170_1
        del arg73_1
        del arg74_1
        # Source Nodes: [getattr_l__mod___stage2___3___attn_qkv, getattr_l__mod___stage2___3___norm1, x_101, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 1152, 14, 14), (225792, 196, 14, 1))
        del arg75_1
        buf101 = reinterpret_tensor(buf99, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf99  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf100, buf101, 9408, 64, grid=grid(9408, 64), stream=stream0)
        buf102 = reinterpret_tensor(buf92, (8, 6, 64, 196), (75264, 12544, 196, 1), 0); del buf92  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf100, buf102, 602112, grid=grid(602112), stream=stream0)
        buf103 = reinterpret_tensor(buf90, (48, 196, 196), (38416, 196, 1), 0); del buf90  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf102, (48, 64, 196), (12544, 196, 1), 0), out=buf103)
        buf106 = reinterpret_tensor(buf87, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf87  # reuse
        # Source Nodes: [attn_10, attn_9], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_11.run(buf103, buf106, 9408, 196, grid=grid(9408), stream=stream0)
        del buf103
        buf107 = reinterpret_tensor(buf102, (8, 6, 196, 64), (75264, 12544, 64, 1), 0); del buf102  # reuse
        # Source Nodes: [x_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf100, buf107, 9408, 64, grid=grid(9408, 64), stream=stream0)
        del buf100
        buf108 = reinterpret_tensor(buf101, (48, 196, 64), (12544, 64, 1), 0); del buf101  # reuse
        # Source Nodes: [x_103], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf107, (48, 196, 64), (12544, 64, 1), 0), out=buf108)
        del buf106
        buf109 = reinterpret_tensor(buf107, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf107  # reuse
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf108, buf109, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del buf108
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg76_1
        buf111 = buf109; del buf109  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___norm2, x_101, x_107, x_108, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_16.run(buf82, buf94, buf98, buf110, arg172_1, arg173_1, arg77_1, arg78_1, buf111, 602112, grid=grid(602112), stream=stream0)
        del arg172_1
        del arg173_1
        del arg77_1
        del arg78_1
        # Source Nodes: [getattr_l__mod___stage2___3___norm2, x_101, x_107, x_108, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg79_1
        del buf111
        buf113 = buf112; del buf112  # reuse
        # Source Nodes: [x_109, x_111], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_2.run(buf113, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_109, x_111], Original ATen: [aten.convolution, aten.gelu]
        buf114 = extern_kernels.convolution(buf113, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg80_1
        del buf113
        buf115 = buf110; del buf110  # reuse
        # Source Nodes: [x_101, x_107, x_114, x_115, x_95], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_add_convolution_18.run(buf115, buf82, buf94, buf98, buf114, 602112, grid=grid(602112), stream=stream0)
        del buf114
        del buf82
        del buf94
        del buf98
        # Source Nodes: [x_101, x_107, x_114, x_115, x_95], Original ATen: [aten.add, aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg81_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg81_1
        del buf115
        buf117 = buf116; del buf116  # reuse
        buf118 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, getattr_l__mod___stage3___0___attn_qkv, getattr_l__mod___stage3___0___norm1, x_101, x_107, x_114, x_115, x_117, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19.run(buf117, arg82_1, arg175_1, arg176_1, arg83_1, arg84_1, arg2_1, arg178_1, arg179_1, arg85_1, arg86_1, buf118, 301056, grid=grid(301056), stream=stream0)
        del arg175_1
        del arg176_1
        del arg178_1
        del arg179_1
        del arg2_1
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        # Source Nodes: [getattr_l__mod___stage3___0___attn_qkv, getattr_l__mod___stage3___0___norm1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf119 = extern_kernels.convolution(buf118, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 2304, 7, 7), (112896, 49, 7, 1))
        del arg87_1
        buf120 = reinterpret_tensor(buf118, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf118  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf119, buf120, 2352, 128, grid=grid(2352, 128), stream=stream0)
        buf121 = empty((8, 6, 128, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf119, buf121, 301056, grid=grid(301056), stream=stream0)
        buf122 = empty((48, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf121, (48, 128, 49), (6272, 49, 1), 0), out=buf122)
        buf125 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_22.run(buf122, buf125, 2352, 49, grid=grid(2352), stream=stream0)
        buf126 = reinterpret_tensor(buf121, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf121  # reuse
        # Source Nodes: [x_120], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf119, buf126, 2352, 128, grid=grid(2352, 128), stream=stream0)
        del buf119
        buf127 = reinterpret_tensor(buf120, (48, 49, 128), (6272, 128, 1), 0); del buf120  # reuse
        # Source Nodes: [x_120], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf126, (48, 49, 128), (6272, 128, 1), 0), out=buf127)
        buf128 = reinterpret_tensor(buf126, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf126  # reuse
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf127, buf128, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg88_1
        buf130 = buf128; del buf128  # reuse
        # Source Nodes: [getattr_l__mod___stage3___0___norm2, x_124, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_25.run(buf117, buf129, arg181_1, arg182_1, arg89_1, arg90_1, buf130, 301056, grid=grid(301056), stream=stream0)
        del arg181_1
        del arg182_1
        del arg89_1
        del arg90_1
        # Source Nodes: [getattr_l__mod___stage3___0___norm2, x_124, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg91_1
        buf132 = buf131; del buf131  # reuse
        # Source Nodes: [x_126, x_128], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_26.run(buf132, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_126, x_128], Original ATen: [aten.convolution, aten.gelu]
        buf133 = extern_kernels.convolution(buf132, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg92_1
        del buf132
        buf134 = buf130; del buf130  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___attn_qkv, getattr_l__mod___stage3___1___norm1, x_124, x_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27.run(buf117, buf129, buf133, arg184_1, arg185_1, arg93_1, arg94_1, buf134, 301056, grid=grid(301056), stream=stream0)
        del arg184_1
        del arg185_1
        del arg93_1
        del arg94_1
        # Source Nodes: [getattr_l__mod___stage3___1___attn_qkv, getattr_l__mod___stage3___1___norm1, x_124, x_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf135 = extern_kernels.convolution(buf134, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 2304, 7, 7), (112896, 49, 7, 1))
        del arg95_1
        buf136 = reinterpret_tensor(buf134, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf134  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf135, buf136, 2352, 128, grid=grid(2352, 128), stream=stream0)
        buf137 = reinterpret_tensor(buf127, (8, 6, 128, 49), (37632, 6272, 49, 1), 0); del buf127  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf135, buf137, 301056, grid=grid(301056), stream=stream0)
        buf138 = reinterpret_tensor(buf125, (48, 49, 49), (2401, 49, 1), 0); del buf125  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf137, (48, 128, 49), (6272, 49, 1), 0), out=buf138)
        buf141 = reinterpret_tensor(buf122, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf122  # reuse
        # Source Nodes: [attn_15, attn_16], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_22.run(buf138, buf141, 2352, 49, grid=grid(2352), stream=stream0)
        buf142 = reinterpret_tensor(buf137, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf137  # reuse
        # Source Nodes: [x_132], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf135, buf142, 2352, 128, grid=grid(2352, 128), stream=stream0)
        del buf135
        buf143 = reinterpret_tensor(buf136, (48, 49, 128), (6272, 128, 1), 0); del buf136  # reuse
        # Source Nodes: [x_132], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf142, (48, 49, 128), (6272, 128, 1), 0), out=buf143)
        buf144 = reinterpret_tensor(buf142, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf142  # reuse
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf143, buf144, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del buf143
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg96_1
        buf146 = buf144; del buf144  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___norm2, x_124, x_130, x_136, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_28.run(buf117, buf129, buf133, buf145, arg187_1, arg188_1, arg97_1, arg98_1, buf146, 301056, grid=grid(301056), stream=stream0)
        del arg187_1
        del arg188_1
        del arg97_1
        del arg98_1
        # Source Nodes: [getattr_l__mod___stage3___1___norm2, x_124, x_130, x_136, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf147 = extern_kernels.convolution(buf146, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg99_1
        buf148 = buf147; del buf147  # reuse
        # Source Nodes: [x_138, x_140], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_26.run(buf148, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_138, x_140], Original ATen: [aten.convolution, aten.gelu]
        buf149 = extern_kernels.convolution(buf148, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg100_1
        del buf148
        buf150 = buf117; del buf117  # reuse
        buf151 = buf146; del buf146  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___attn_qkv, getattr_l__mod___stage3___2___norm1, x_124, x_130, x_136, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_29.run(buf150, buf129, buf133, buf145, buf149, arg190_1, arg191_1, arg101_1, arg102_1, buf151, 301056, grid=grid(301056), stream=stream0)
        del arg101_1
        del arg102_1
        del arg190_1
        del arg191_1
        del buf129
        del buf133
        del buf145
        # Source Nodes: [getattr_l__mod___stage3___2___attn_qkv, getattr_l__mod___stage3___2___norm1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 2304, 7, 7), (112896, 49, 7, 1))
        del arg103_1
        buf153 = reinterpret_tensor(buf151, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf151  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf152, buf153, 2352, 128, grid=grid(2352, 128), stream=stream0)
        buf154 = reinterpret_tensor(buf149, (8, 6, 128, 49), (37632, 6272, 49, 1), 0); del buf149  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf152, buf154, 301056, grid=grid(301056), stream=stream0)
        buf155 = reinterpret_tensor(buf141, (48, 49, 49), (2401, 49, 1), 0); del buf141  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf154, (48, 128, 49), (6272, 49, 1), 0), out=buf155)
        buf158 = reinterpret_tensor(buf138, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf138  # reuse
        # Source Nodes: [attn_18, attn_19], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_22.run(buf155, buf158, 2352, 49, grid=grid(2352), stream=stream0)
        buf159 = reinterpret_tensor(buf154, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf154  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf152, buf159, 2352, 128, grid=grid(2352, 128), stream=stream0)
        del buf152
        buf160 = reinterpret_tensor(buf153, (48, 49, 128), (6272, 128, 1), 0); del buf153  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf159, (48, 49, 128), (6272, 128, 1), 0), out=buf160)
        buf161 = reinterpret_tensor(buf159, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf159  # reuse
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf160, buf161, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg104_1
        buf163 = buf161; del buf161  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___norm2, x_148, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_25.run(buf150, buf162, arg193_1, arg194_1, arg105_1, arg106_1, buf163, 301056, grid=grid(301056), stream=stream0)
        del arg105_1
        del arg106_1
        del arg193_1
        del arg194_1
        # Source Nodes: [getattr_l__mod___stage3___2___norm2, x_148, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf164 = extern_kernels.convolution(buf163, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg107_1
        buf165 = buf164; del buf164  # reuse
        # Source Nodes: [x_150, x_152], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_26.run(buf165, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_150, x_152], Original ATen: [aten.convolution, aten.gelu]
        buf166 = extern_kernels.convolution(buf165, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg108_1
        del buf165
        buf167 = buf163; del buf163  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___attn_qkv, getattr_l__mod___stage3___3___norm1, x_148, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_27.run(buf150, buf162, buf166, arg196_1, arg197_1, arg109_1, arg110_1, buf167, 301056, grid=grid(301056), stream=stream0)
        del arg109_1
        del arg110_1
        del arg196_1
        del arg197_1
        # Source Nodes: [getattr_l__mod___stage3___3___attn_qkv, getattr_l__mod___stage3___3___norm1, x_148, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 2304, 7, 7), (112896, 49, 7, 1))
        del arg111_1
        buf169 = reinterpret_tensor(buf167, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf167  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf168, buf169, 2352, 128, grid=grid(2352, 128), stream=stream0)
        buf170 = reinterpret_tensor(buf160, (8, 6, 128, 49), (37632, 6272, 49, 1), 0); del buf160  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf168, buf170, 301056, grid=grid(301056), stream=stream0)
        buf171 = reinterpret_tensor(buf158, (48, 49, 49), (2401, 49, 1), 0); del buf158  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf169, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf170, (48, 128, 49), (6272, 49, 1), 0), out=buf171)
        buf174 = reinterpret_tensor(buf155, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf155  # reuse
        # Source Nodes: [attn_21, attn_22], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_22.run(buf171, buf174, 2352, 49, grid=grid(2352), stream=stream0)
        del buf171
        buf175 = reinterpret_tensor(buf170, (8, 6, 49, 128), (37632, 6272, 128, 1), 0); del buf170  # reuse
        # Source Nodes: [x_156], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf168, buf175, 2352, 128, grid=grid(2352, 128), stream=stream0)
        del buf168
        buf176 = reinterpret_tensor(buf169, (48, 49, 128), (6272, 128, 1), 0); del buf169  # reuse
        # Source Nodes: [x_156], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf175, (48, 49, 128), (6272, 128, 1), 0), out=buf176)
        del buf174
        buf177 = reinterpret_tensor(buf175, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf175  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf176, buf177, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del buf176
        # Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg112_1
        buf179 = buf177; del buf177  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___norm2, x_148, x_154, x_160, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_28.run(buf150, buf162, buf166, buf178, arg199_1, arg200_1, arg113_1, arg114_1, buf179, 301056, grid=grid(301056), stream=stream0)
        del arg113_1
        del arg114_1
        del arg199_1
        del arg200_1
        # Source Nodes: [getattr_l__mod___stage3___3___norm2, x_148, x_154, x_160, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg115_1
        del buf179
        buf181 = buf180; del buf180  # reuse
        # Source Nodes: [x_162, x_164], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_26.run(buf181, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_162, x_164], Original ATen: [aten.convolution, aten.gelu]
        buf182 = extern_kernels.convolution(buf181, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg116_1
        del buf181
        buf183 = buf150; del buf150  # reuse
        buf184 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf185 = reinterpret_tensor(buf184, (8, 768, 1, 1), (768, 1, 1, 1), 0); del buf184  # reuse
        # Source Nodes: [x_148, x_154, x_160, x_167, x_169, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_30.run(buf183, buf185, buf162, buf166, buf178, buf182, arg202_1, arg203_1, arg117_1, arg118_1, 6144, 49, grid=grid(6144), stream=stream0)
        del arg117_1
        del arg118_1
        del arg202_1
        del arg203_1
        del buf162
        del buf166
        del buf178
        del buf182
        del buf183
        buf186 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg120_1, reinterpret_tensor(buf185, (8, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf186)
        del arg119_1
        del arg120_1
        return (buf186, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((192, 32, 4, 4), (512, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 384, 2, 2), (1536, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg124_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg127_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg130_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg133_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg136_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg139_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg142_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg145_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg148_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg151_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg157_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg163_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg169_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg205_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('visformer_small', benchmark_compiled_module)
