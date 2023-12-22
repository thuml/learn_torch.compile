
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cgg463u26szk2koomsmwdhziq76kysbb7oomrurjsyrliqijjwz4.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1000 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2000 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3000 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/lg/clg2tzzqrwf3zrsuh52qipnavesnkobsojuhthq7osdnaty7pqvt.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp20 = tmp18 - tmp19
    tmp21 = tmp5 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = tl.math.rsqrt(tmp28)
    tmp30 = tmp17 * tmp29
    tmp32 = tmp31 + tmp27
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp25 * tmp33
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp30, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp34, None)
    tl.store(out_ptr0 + (x0), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4m/c4msgvkgubcu5fyuht62mckax4pa7bc7jnia3awueiuberthxsxc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13 + tmp7
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp5 * tmp17
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xo/cxogh5hybldmg3e7yxxs4bhpvcwvwpui2nbjvzpp3zlgbo7i24bf.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), None)
    tmp18 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(7, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(7, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdh4vfrihaowaybnvfruu7l4kislwd4gyfutyg7qndidwb4evkfo.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 512
    x2 = (xindex // 1024)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (196*x0) + (100352*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (196*x4)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/cclumvtymnz6d7xvj6lji3om4l3ollvrwmdkcvazxviozpnjz3mv.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_poi_fused__softmax_backward_data_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp2 - tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4crgiby4fglgsawf4euazjna6rzjnccztwm3ryhjoylpmzp4q4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2048 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3072 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujn5x2qykhkj747odzw2usniahvlo4ng7dga5nxbh74dmosue2c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp7 = tl.load(in_ptr1 + (256 + x0), xmask)
    tmp10 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp12 = tl.load(in_ptr1 + (512 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (768 + x0), xmask)
    tmp17 = tl.load(in_ptr1 + (768 + x0), xmask)
    tmp20 = tl.load(in_ptr2 + (x0), xmask)
    tmp21 = tl.load(in_ptr3 + (x0), xmask)
    tmp24 = tl.load(in_ptr2 + (256 + x0), xmask)
    tmp28 = tl.load(in_ptr2 + (512 + x0), xmask)
    tmp32 = tl.load(in_ptr2 + (768 + x0), xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = tmp5 <= tmp1
    tmp8 = tl.where(tmp6, tmp1, tmp7)
    tmp9 = tmp4 + tmp8
    tmp11 = tmp10 <= tmp1
    tmp13 = tl.where(tmp11, tmp1, tmp12)
    tmp14 = tmp9 + tmp13
    tmp16 = tmp15 <= tmp1
    tmp18 = tl.where(tmp16, tmp1, tmp17)
    tmp19 = tmp14 + tmp18
    tmp22 = tmp20 - tmp21
    tmp23 = tmp4 * tmp22
    tmp25 = tmp24 - tmp21
    tmp26 = tmp8 * tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp28 - tmp21
    tmp30 = tmp13 * tmp29
    tmp31 = tmp27 + tmp30
    tmp33 = tmp32 - tmp21
    tmp34 = tmp18 * tmp33
    tmp35 = tmp31 + tmp34
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp35 * tmp39
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tl.store(in_out_ptr0 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfpzoh7xahsfnr34lwfw4mcbhyh74riamd7x4x5o3qyq2fjnf6q.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ek/cekwwvl6vl6d4wwkgmiw3lxqdbzytekgpse4l6eatrvtcevddzdl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (768 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhixeikrh2l4rnurnr5rdi775xgl7ee77vdq3cpm4s4hcknsce5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*(x0 % 512)) + (100352*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + ((512*r2) + (x0 % 512)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 196.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp15 - tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp22 * tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp27, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqh5k5kc5aq7lglfvf2aex6ht3i4vqd3uesuqujfympcowutfnk4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196) % 1024
    x2 = (xindex // 200704)
    x4 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (196*(x1 % 512)) + (100352*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((512*x2) + (x1 % 512)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 196.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfaqgi4fvdwdcf72ca4bki3kmrcygpdnvga2mm2crkdfsuv5njnv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7desf2necqbbc4u55v6i6coqcma6dtzvdyppr3mx6ne3loldrr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ra/crai2uqmova2hneddxnqrg7u5q7cujlemver3vyy2m2cirrah63c.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = (rindex // 196)
    r4 = rindex % 196
    x0 = xindex
    r1 = rindex % 14
    r2 = (rindex // 14) % 14
    tmp0 = tl.load(in_ptr0 + (r4 + (196*x0) + (200704*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2))))) >= 0, 0, 7))) + (49*x0) + (50176*r3) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) >= 0, 0, 7))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + (r4 + (196*x0) + (200704*r3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r4 + (196*x0) + (200704*r3)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (r4 + (196*x0) + (200704*r3)), rmask & xmask, other=0.0)
    tmp29 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (r2 // 2))
    tmp6 = tl.math.min(7, 1 + (r2 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (r1 // 2))
    tmp9 = tl.math.min(7, 1 + (r1 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp22 = tmp20 - tmp21
    tmp23 = tmp15 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp30 = tmp28 - tmp29
    tmp31 = tmp15 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp27 * tmp39
    tmp42 = tmp41 + tmp37
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp35 * tmp43
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp40, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp44, xmask)
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56mzi7mjc7bj6y4ybs34vswgk5n3q7jbzkwiahovmptmnuks3mk.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x5 = (xindex // 196)
    x2 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x5) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x4), None)
    tmp16 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (x1 // 2))
    tmp6 = tl.math.min(7, 1 + (x1 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (x0 // 2))
    tmp9 = tl.math.min(7, 1 + (x0 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 * tmp21
    tmp24 = tmp23 + tmp17
    tmp25 = tl.math.rsqrt(tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp15 * tmp27
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c37afhyi6ofe5hnlq2j7557sjslghyblaairc2dojdv6u76te2su.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), None)
    tmp18 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(14, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(14, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2oblm3hfd5x4gzg3gik2sdvxeojgbhifgj7oz45jfqfqrluw72b.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 256
    x2 = (xindex // 512)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (784*x0) + (200704*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (784*x4)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wj/cwj3twq5tzfttqlg7iv3akxqgc2xovxuyvbo5f4ubrt47h756vg3.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_poi_fused__softmax_backward_data_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (256 + x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (256 + x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp2 - tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nj/cnj4g7jv7wzsg4cnk3iqt3k3vpgmtlle75zloxudf6hs2blnsf7c.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (1536 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czd2p6xfrzk6xywu6f2pvrlmhwcyvdanegkou2z4e3r62y73wgsi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr0 + (128 + x0), xmask)
    tmp7 = tl.load(in_ptr1 + (128 + x0), xmask)
    tmp10 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp12 = tl.load(in_ptr1 + (256 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (384 + x0), xmask)
    tmp17 = tl.load(in_ptr1 + (384 + x0), xmask)
    tmp20 = tl.load(in_ptr2 + (x0), xmask)
    tmp21 = tl.load(in_ptr3 + (x0), xmask)
    tmp24 = tl.load(in_ptr2 + (128 + x0), xmask)
    tmp28 = tl.load(in_ptr2 + (256 + x0), xmask)
    tmp32 = tl.load(in_ptr2 + (384 + x0), xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = tmp5 <= tmp1
    tmp8 = tl.where(tmp6, tmp1, tmp7)
    tmp9 = tmp4 + tmp8
    tmp11 = tmp10 <= tmp1
    tmp13 = tl.where(tmp11, tmp1, tmp12)
    tmp14 = tmp9 + tmp13
    tmp16 = tmp15 <= tmp1
    tmp18 = tl.where(tmp16, tmp1, tmp17)
    tmp19 = tmp14 + tmp18
    tmp22 = tmp20 - tmp21
    tmp23 = tmp4 * tmp22
    tmp25 = tmp24 - tmp21
    tmp26 = tmp8 * tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp28 - tmp21
    tmp30 = tmp13 * tmp29
    tmp31 = tmp27 + tmp30
    tmp33 = tmp32 - tmp21
    tmp34 = tmp18 * tmp33
    tmp35 = tmp31 + tmp34
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp35 * tmp39
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tl.store(in_out_ptr0 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/ceheg52rex3zu27yczn2mfgbruheshhj5nvxh74iigjb72b2bn6h.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chts6oukzbvnbejw35xqeqou5t2qif7ryjj735f4n5tjwdffvywb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (128 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (384 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rq/crqqaomus54g6tectdj3pe75xiydypgiavvktr52wd66rdkuarlk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*(x0 % 256)) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((256*r2) + (x0 % 256)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 784.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvrjurtw2fusop5evo37qpmtxt54oryl5e22pvwow3rvhpdvkwq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784) % 512
    x2 = (xindex // 401408)
    x4 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (784*(x1 % 256)) + (200704*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((256*x2) + (x1 % 256)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 784.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5o5gdgias5milnlxxwt6zznjlary3pvvvmjztypuvnanpfad6j.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp19 = tmp13 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jba74r7enhiiiebbyp73y2on7tmw6e2bac7lyu5zwouipnxy2s.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjd7mt2r22fpyegb7odpntxwwzmbgscblfmnmijl3yn5zoogxu3.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp27 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 784)
        r4 = rindex % 784
        r1 = rindex % 28
        r2 = (rindex // 28) % 28
        tmp0 = tl.load(in_ptr0 + (r4 + (784*x0) + (401408*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2))))) >= 0, 0, 14))) + (196*x0) + (100352*r3) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) >= 0, 0, 14))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (r4 + (784*x0) + (401408*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r4 + (784*x0) + (401408*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr5 + (r4 + (784*x0) + (401408*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 / 4
        tmp5 = tl.math.max(0, (r2 // 2))
        tmp6 = tl.math.min(14, 1 + (r2 // 2))
        tmp7 = tmp5 < tmp6
        tmp8 = tl.math.max(0, (r1 // 2))
        tmp9 = tl.math.min(14, 1 + (r1 // 2))
        tmp10 = tmp8 < tmp9
        tmp11 = tmp7 & tmp10
        tmp12 = tl.where(tmp11, tmp4, tmp1)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(tmp2, tmp1, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp15 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp24 * tmp36
    tmp39 = tmp38 + tmp34
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp31 * tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp37, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5bdsaefysywuuridpilejsb643ce2f4l7d3msd36vixlmzqazm.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x5 = (xindex // 784)
    x2 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x5) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x4), None)
    tmp16 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (x1 // 2))
    tmp6 = tl.math.min(14, 1 + (x1 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (x0 // 2))
    tmp9 = tl.math.min(14, 1 + (x0 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 * tmp21
    tmp24 = tmp23 + tmp17
    tmp25 = tl.math.rsqrt(tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp15 * tmp27
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccr7eho4sazkpc3t6nkkzkyoxyf6wivhjo5nh6u67ainxcoqjh5z.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None)
    tmp18 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(28, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(28, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxjkrwnoj5iz4shmsetdoclcd2m6oa4jbxzzxfgpw2tf7exoaba.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x2 = (xindex // 256)
    x4 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (3136*x0) + (401408*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (3136*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/cletznnck4aq26ekidwput7miwztfhakobq75gxpvg6vwh4kzeav.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_poi_fused__softmax_backward_data_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (256*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (256*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (128 + x0 + (256*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (128 + x0 + (256*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp2 - tmp10
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvlfrwizvww5oboscxwy66oenwedrrawqrezifs52qolu2rckwr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp7 = tl.load(in_ptr1 + (64 + x0), xmask)
    tmp10 = tl.load(in_ptr0 + (128 + x0), xmask)
    tmp12 = tl.load(in_ptr1 + (128 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (192 + x0), xmask)
    tmp17 = tl.load(in_ptr1 + (192 + x0), xmask)
    tmp20 = tl.load(in_ptr2 + (x0), xmask)
    tmp21 = tl.load(in_ptr3 + (x0), xmask)
    tmp24 = tl.load(in_ptr2 + (64 + x0), xmask)
    tmp28 = tl.load(in_ptr2 + (128 + x0), xmask)
    tmp32 = tl.load(in_ptr2 + (192 + x0), xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = tmp5 <= tmp1
    tmp8 = tl.where(tmp6, tmp1, tmp7)
    tmp9 = tmp4 + tmp8
    tmp11 = tmp10 <= tmp1
    tmp13 = tl.where(tmp11, tmp1, tmp12)
    tmp14 = tmp9 + tmp13
    tmp16 = tmp15 <= tmp1
    tmp18 = tl.where(tmp16, tmp1, tmp17)
    tmp19 = tmp14 + tmp18
    tmp22 = tmp20 - tmp21
    tmp23 = tmp4 * tmp22
    tmp25 = tmp24 - tmp21
    tmp26 = tmp8 * tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp28 - tmp21
    tmp30 = tmp13 * tmp29
    tmp31 = tmp27 + tmp30
    tmp33 = tmp32 - tmp21
    tmp34 = tmp18 * tmp33
    tmp35 = tmp31 + tmp34
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp35 * tmp39
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tl.store(in_out_ptr0 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/53/c53srcbr2iq53vkxmiez3pnob5cru5t5pyc35se2bvfiveui4mcl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjkpvhtpuot3sdsuk3mulnnmhclbovucywviq5bq4nxodbwjzgm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (192 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdq5kxwilrczk33ywnttkpjthmpw373fr4ygh3uemlxjvaegl2p.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*(x0 % 128)) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((128*r2) + (x0 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 3136.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mg/cmgblno7pbxlagbr4dn3qhr4nmc5nhwmr22ansm6mnx76ce4uiur.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 3136
    x1 = (xindex // 3136) % 256
    x2 = (xindex // 802816)
    x4 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (3136*(x1 % 128)) + (401408*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((128*x2) + (x1 % 128)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 3136.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmftadrmjoiqnmpg6u246agx4g4bqtl5kha63imuwz33o6g67yec.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwnovcp3kovztajpw2vjfwllxrdfpain7lxseirbeiwiqudevbsa.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2c/c2cayavkxpvhn4rhsiein4ugvnq5kzikceu26rbzqii4kssvuthw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/ckeq3qcjegt7spkiulxwvioz5z6w2xr4odria3gy2pdhiffc4vtg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2lnb2xxt7qexa24q5cs4h6p7gfvsskrkpi4fvmpv5vqqlwiu7jl.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp27 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 3136)
        r4 = rindex % 3136
        r1 = rindex % 56
        r2 = (rindex // 56) % 56
        tmp0 = tl.load(in_ptr0 + (r4 + (3136*x0) + (802816*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((28*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(28, 1 + (r2 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(28, 1 + (r2 // 2))))) >= 0, 0, 28))) + (784*x0) + (200704*r3) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(28, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(28, 1 + (r1 // 2))))) >= 0, 0, 28))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (r4 + (3136*x0) + (802816*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r4 + (3136*x0) + (802816*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr5 + (r4 + (3136*x0) + (802816*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 / 4
        tmp5 = tl.math.max(0, (r2 // 2))
        tmp6 = tl.math.min(28, 1 + (r2 // 2))
        tmp7 = tmp5 < tmp6
        tmp8 = tl.math.max(0, (r1 // 2))
        tmp9 = tl.math.min(28, 1 + (r1 // 2))
        tmp10 = tmp8 < tmp9
        tmp11 = tmp7 & tmp10
        tmp12 = tl.where(tmp11, tmp4, tmp1)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(tmp2, tmp1, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp15 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp24 * tmp36
    tmp39 = tmp38 + tmp34
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp31 * tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp37, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fy/cfyg47pqinvo6msd2ry5eq726dz2hupfzlrxtueu36uarlfkri4z.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x5 = (xindex // 3136)
    x2 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + ((28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x5) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x4), None)
    tmp16 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (x1 // 2))
    tmp6 = tl.math.min(28, 1 + (x1 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (x0 // 2))
    tmp9 = tl.math.min(28, 1 + (x0 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 * tmp21
    tmp24 = tmp23 + tmp17
    tmp25 = tl.math.rsqrt(tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp15 * tmp27
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotkyb7ettplsj34jxywoxah6a4lhpehtywhul4ioqivsrpdry3v.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x2 = (xindex // 128)
    x4 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (3136*x0) + (200704*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (3136*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6e6ecc5x66vztsg6rbjnlui55xqntnd5omtcx2u3fhwjfahcru.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_poi_fused__softmax_backward_data_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (64 + x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (64 + x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp2 - tmp10
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdoy7xdougnvbz25eooxx3avpw3vaqyk6bmx4y72nux7gywc6anp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr0 + (32 + x0), xmask)
    tmp7 = tl.load(in_ptr1 + (32 + x0), xmask)
    tmp10 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp12 = tl.load(in_ptr1 + (64 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (96 + x0), xmask)
    tmp17 = tl.load(in_ptr1 + (96 + x0), xmask)
    tmp20 = tl.load(in_ptr2 + (x0), xmask)
    tmp21 = tl.load(in_ptr3 + (x0), xmask)
    tmp24 = tl.load(in_ptr2 + (32 + x0), xmask)
    tmp28 = tl.load(in_ptr2 + (64 + x0), xmask)
    tmp32 = tl.load(in_ptr2 + (96 + x0), xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = tmp5 <= tmp1
    tmp8 = tl.where(tmp6, tmp1, tmp7)
    tmp9 = tmp4 + tmp8
    tmp11 = tmp10 <= tmp1
    tmp13 = tl.where(tmp11, tmp1, tmp12)
    tmp14 = tmp9 + tmp13
    tmp16 = tmp15 <= tmp1
    tmp18 = tl.where(tmp16, tmp1, tmp17)
    tmp19 = tmp14 + tmp18
    tmp22 = tmp20 - tmp21
    tmp23 = tmp4 * tmp22
    tmp25 = tmp24 - tmp21
    tmp26 = tmp8 * tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp28 - tmp21
    tmp30 = tmp13 * tmp29
    tmp31 = tmp27 + tmp30
    tmp33 = tmp32 - tmp21
    tmp34 = tmp18 * tmp33
    tmp35 = tmp31 + tmp34
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp35 * tmp39
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tl.store(in_out_ptr0 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6typkindylic37ktg73i5pzlhjco4svl3oc5252dbdmgam6icns.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tu/ctumdkn2msef2tjytzov5huxrrszapvzr2hpcbez3wqomhsfh4ny.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (32 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (96 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbb4hi4zxdtz475y5n25dez6xhpehshmyl3m5wbrpn3ztit6byz4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*(x0 % 64)) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*(r2 // 3136)) + (256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((64*(r2 // 3136)) + (128*x1) + (x0 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 3136.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nl/cnla5kvkkt7usigxqhglrbhuyhgbnrefcz5mqskbczxcp3qaxkzb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 3136
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (3136*(x1 % 64)) + (200704*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((64*x2) + (x1 % 64)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 3136.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmipekwhtfmlzulgnghihx4e3fxm326xkihk72cqv53b3jbswh3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dc/cdcrgytbpmzsz22njqyj6wl62xwcfp33hzhl4ftovzqagyqxkjkr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/na/cnaujtlney67dwuxepmgb4r6x6mn6vbohpc6rga7kbvinb7cgs72.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6i/c6imgf6r3ujlwm7d2vqt43q3clukbjdr4t5zyf23lerdokbdp32n.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czsnkivzluvi26pdmwrlrrcs2jdspihclglet56zniok43nrxjih.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccne5xmvpbxp4dgyfqzpiyk26jtybencdbmxo2ybyjscemktii55.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = (xindex // 112) % 112
    x2 = (xindex // 12544)
    x3 = xindex % 12544
    x5 = xindex
    x6 = (xindex // 12544) % 64
    tmp0 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp7 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp19 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp31 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp37 = tl.load(in_ptr2 + (x5), None)
    tmp40 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (x1 // 2))
    tmp10 = tl.math.min(56, 1 + ((1 + x1) // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp13 = tl.math.min(56, 1 + ((1 + x0) // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.math.max(0, (x0 // 2))
    tmp25 = tmp24 < tmp13
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp21
    tmp28 = tmp18 + tmp20
    tmp29 = tl.where(tmp27, tmp28, tmp18)
    tmp32 = tmp30 == tmp2
    tmp33 = tmp23 & tmp14
    tmp34 = tmp33 & tmp32
    tmp35 = tmp29 + tmp31
    tmp36 = tl.where(tmp34, tmp35, tmp29)
    tmp38 = tmp37 <= tmp4
    tmp39 = tl.where(tmp38, tmp4, tmp36)
    tmp41 = 1e-05
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp45 = tmp43 * tmp44
    tmp46 = tmp39 * tmp45
    tl.store(out_ptr0 + (x5), tmp36, None)
    tl.store(out_ptr1 + (x5), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zp/czp5sgrgautqwntltfty6waw7gmlj7l7v2ijqz7lrmtnc7w67yqe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (802816*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (802816*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (802816*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sn/csnvklmr4wayurdjziuxlbap6ppeoakclhiu33yv25nugtkqrzqh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/cznrxqci4tjq4ohjvphnexf2w44i3oatg2tjlo5yfxqu6tccm3nh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pa/cpak7bdr536lmqavoyvbjzpkhlwqebuelienrkgiaffgzqjtwosr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (401408*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (401408*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (401408*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/citkd3z4pvidxukjqcq3f3vvkhdoofyasn3nkhmcvbzl6rviqivp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2kiyndhf4e7lvzomksdnj5urn3zzyvtkw2zdzvxl3sewdg5up5c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqb5hyf5nb7gct5tjnovtskldmff3c2gr527ohuckj6p5fr2gq2l.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_54, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_72, primals_74, primals_76, primals_77, primals_79, primals_80, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, getitem, getitem_1, convolution_3, relu_3, convolution_4, relu_4, mean, convolution_5, relu_5, div, sum_3, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, mean_1, convolution_11, relu_9, div_1, sum_6, avg_pool2d, convolution_13, avg_pool2d_1, convolution_14, relu_10, convolution_15, relu_11, convolution_16, relu_12, mean_2, convolution_17, relu_13, div_2, sum_9, avg_pool2d_2, convolution_19, avg_pool2d_3, convolution_20, relu_14, convolution_21, relu_15, convolution_22, relu_16, mean_3, convolution_23, relu_17, div_3, sum_12, avg_pool2d_4, convolution_25, avg_pool2d_5, convolution_26, view_24, permute_5, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_16, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_20, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_28, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_34, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_38, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_46, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_52, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_56, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_58, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_61, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_62, (1024, ), (1, ))
    assert_size_stride(primals_64, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_67, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_70, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_74, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_76, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_77, (2048, ), (1, ))
    assert_size_stride(primals_79, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_80, (2048, ), (1, ))
    assert_size_stride(primals_84, (32, ), (1, ))
    assert_size_stride(primals_85, (32, ), (1, ))
    assert_size_stride(primals_87, (32, ), (1, ))
    assert_size_stride(primals_88, (32, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, ), (1, ))
    assert_size_stride(primals_138, (512, ), (1, ))
    assert_size_stride(primals_139, (512, ), (1, ))
    assert_size_stride(primals_141, (1024, ), (1, ))
    assert_size_stride(primals_142, (1024, ), (1, ))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_147, (2048, ), (1, ))
    assert_size_stride(primals_148, (2048, ), (1, ))
    assert_size_stride(primals_150, (2048, ), (1, ))
    assert_size_stride(primals_151, (2048, ), (1, ))
    assert_size_stride(primals_153, (4, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (4, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(relu, (4, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(convolution_1, (4, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(relu_1, (4, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(convolution_2, (4, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(relu_2, (4, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(getitem, (4, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(getitem_1, (4, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_3, (4, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(relu_3, (4, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_4, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_4, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(mean, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(convolution_5, (4, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(relu_5, (4, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(div, (4, 2, 1, 64), (128, 64, 64, 1))
    assert_size_stride(sum_3, (4, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_7, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_8, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(relu_6, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_9, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_7, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_10, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(relu_8, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(mean_1, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_11, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(relu_9, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(div_1, (4, 2, 1, 128), (256, 128, 128, 1))
    assert_size_stride(sum_6, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(avg_pool2d, (4, 128, 28, 28), (100352, 784, 28, 1))
    assert_size_stride(convolution_13, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(avg_pool2d_1, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_14, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(relu_10, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_15, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(relu_11, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_16, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(relu_12, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(mean_2, (4, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_17, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_13, (4, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_2, (4, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_9, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(avg_pool2d_2, (4, 256, 14, 14), (50176, 196, 14, 1))
    assert_size_stride(convolution_19, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(avg_pool2d_3, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_20, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_14, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_21, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_15, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_22, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_16, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(mean_3, (4, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_23, (4, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(relu_17, (4, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(div_3, (4, 2, 1, 512), (1024, 512, 512, 1))
    assert_size_stride(sum_12, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(avg_pool2d_4, (4, 512, 7, 7), (25088, 49, 7, 1))
    assert_size_stride(convolution_25, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(avg_pool2d_5, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_26, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(view_24, (4, 2048), (2048, 1))
    assert_size_stride(permute_5, (1000, 2048), (2048, 1))
    assert_size_stride(le, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_5, out=buf0)
        del permute_5
        buf1 = empty((1000, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), view_24, out=buf1)
        del view_24
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf4 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf10 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf5 = buf4; del buf4  # reuse
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(buf5, buf11, le, buf0, convolution_26, primals_150, convolution_25, primals_147, primals_151, primals_148, buf3, 2048, 196, grid=grid(2048), stream=stream0)
        del convolution_25
        del convolution_26
        del primals_147
        del primals_150
        buf6 = empty((4, 2048, 7, 7), device='cuda', dtype=torch.float32)
        buf12 = empty((4, 2048, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2.run(le, buf0, primals_151, primals_80, primals_148, primals_77, buf6, buf12, 401408, grid=grid(401408), stream=stream0)
        del buf0
        del le
        del primals_148
        del primals_151
        del primals_77
        del primals_80
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf7 = aten.convolution_backward(buf6, avg_pool2d_5, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_5
        del buf6
        del primals_79
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf13 = aten.convolution_backward(buf12, avg_pool2d_4, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_4
        del primals_76
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        buf16 = reinterpret_tensor(buf12, (4, 512, 14, 14), (100352, 196, 14, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_3.run(buf14, buf16, 401408, grid=grid(401408), stream=stream0)
        del buf14
        buf17 = empty((4, 2, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_4.run(buf16, relu_16, buf17, 4096, 196, grid=grid(4096), stream=stream0)
        buf18 = empty((4, 2, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_5.run(buf17, div_3, buf18, 4096, grid=grid(4096), stream=stream0)
        del buf17
        buf19 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf18, buf19, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf20 = aten.convolution_backward(reinterpret_tensor(buf18, (4, 1024, 1, 1), (1024, 1, 0, 0), 0), relu_17, primals_74, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf18
        del primals_74
        buf21 = buf20[0]
        buf22 = buf20[1]
        del buf20
        buf23 = empty((256, ), device='cuda', dtype=torch.float32)
        buf24 = empty((256, ), device='cuda', dtype=torch.float32)
        buf26 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_7.run(buf26, relu_17, buf21, convolution_23, primals_144, primals_145, buf23, 256, grid=grid(256), stream=stream0)
        del convolution_23
        del primals_144
        buf25 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_8.run(buf25, relu_17, primals_145, primals_72, 1024, grid=grid(1024), stream=stream0)
        del primals_145
        del primals_72
        del relu_17
        buf27 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_9.run(buf25, buf27, 256, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf28 = aten.convolution_backward(buf25, mean_3, primals_70, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_3
        del primals_70
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = reinterpret_tensor(buf25, (1024, ), (1, ), 0); del buf25  # reuse
        buf32 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf33, relu_16, buf16, div_3, buf29, convolution_22, primals_141, primals_142, buf31, 1024, 784, grid=grid(1024), stream=stream0)
        del convolution_22
        del primals_141
        buf34 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11.run(relu_16, buf16, div_3, buf29, primals_142, primals_68, buf34, 802816, grid=grid(802816), stream=stream0)
        del buf16
        del div_3
        del primals_142
        del primals_68
        del relu_16
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf35 = aten.convolution_backward(buf34, relu_15, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_67
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = empty((512, ), device='cuda', dtype=torch.float32)
        buf39 = empty((512, ), device='cuda', dtype=torch.float32)
        buf40 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_12.run(buf40, relu_15, buf36, convolution_21, primals_138, primals_139, buf38, 512, 784, grid=grid(512), stream=stream0)
        del convolution_21
        del primals_138
        buf41 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13.run(buf41, relu_15, primals_139, primals_65, 401408, grid=grid(401408), stream=stream0)
        del primals_139
        del primals_65
        del relu_15
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf42 = aten.convolution_backward(buf41, relu_14, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf41
        del primals_64
        buf43 = buf42[0]
        buf44 = buf42[1]
        del buf42
        buf45 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf46 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf52 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf47 = buf46; del buf46  # reuse
        buf53 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_14.run(buf47, buf53, relu_14, buf8, buf43, convolution_20, primals_135, convolution_19, primals_132, primals_136, primals_133, buf45, 1024, 784, grid=grid(1024), stream=stream0)
        del convolution_19
        del convolution_20
        del primals_132
        del primals_135
        buf48 = buf34; del buf34  # reuse
        buf54 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(relu_14, buf8, buf43, primals_136, primals_62, primals_133, primals_59, buf48, buf54, 802816, grid=grid(802816), stream=stream0)
        del buf43
        del buf8
        del primals_133
        del primals_136
        del primals_59
        del primals_62
        del relu_14
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf49 = aten.convolution_backward(buf48, avg_pool2d_3, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_3
        del buf48
        del primals_61
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf55 = aten.convolution_backward(buf54, avg_pool2d_2, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_2
        del primals_58
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf58 = reinterpret_tensor(buf54, (4, 256, 28, 28), (200704, 784, 28, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_16.run(buf56, buf58, 802816, grid=grid(802816), stream=stream0)
        del buf56
        buf59 = reinterpret_tensor(buf29, (4, 2, 256, 1, 1), (512, 256, 1, 1, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_17.run(buf58, relu_12, buf59, 2048, 784, grid=grid(2048), stream=stream0)
        buf60 = empty((4, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_18.run(buf59, div_2, buf60, 2048, grid=grid(2048), stream=stream0)
        del buf59
        buf61 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_19.run(buf60, buf61, 512, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf62 = aten.convolution_backward(reinterpret_tensor(buf60, (4, 512, 1, 1), (512, 1, 0, 0), 0), relu_13, primals_56, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf60
        del primals_56
        buf63 = buf62[0]
        buf64 = buf62[1]
        del buf62
        buf65 = empty((128, ), device='cuda', dtype=torch.float32)
        buf66 = empty((128, ), device='cuda', dtype=torch.float32)
        buf68 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_20.run(buf68, relu_13, buf63, convolution_17, primals_129, primals_130, buf65, 128, grid=grid(128), stream=stream0)
        del convolution_17
        del primals_129
        buf67 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_21.run(buf67, relu_13, primals_130, primals_54, 512, grid=grid(512), stream=stream0)
        del primals_130
        del primals_54
        del relu_13
        buf69 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_22.run(buf67, buf69, 128, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf70 = aten.convolution_backward(buf67, mean_2, primals_52, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_2
        del primals_52
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = reinterpret_tensor(buf67, (512, ), (1, ), 0); del buf67  # reuse
        buf74 = empty((512, ), device='cuda', dtype=torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(buf75, relu_12, buf58, div_2, buf71, convolution_16, primals_126, primals_127, buf73, 512, 3136, grid=grid(512), stream=stream0)
        del convolution_16
        del primals_126
        buf76 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(relu_12, buf58, div_2, buf71, primals_127, primals_50, buf76, 1605632, grid=grid(1605632), stream=stream0)
        del buf58
        del div_2
        del primals_127
        del primals_50
        del relu_12
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf77 = aten.convolution_backward(buf76, relu_11, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_49
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf80 = empty((256, ), device='cuda', dtype=torch.float32)
        buf81 = empty((256, ), device='cuda', dtype=torch.float32)
        buf82 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(buf82, relu_11, buf78, convolution_15, primals_123, primals_124, buf80, 256, 3136, grid=grid(256), stream=stream0)
        del convolution_15
        del primals_123
        buf83 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(buf83, relu_11, primals_124, primals_47, 802816, grid=grid(802816), stream=stream0)
        del primals_124
        del primals_47
        del relu_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf84 = aten.convolution_backward(buf83, relu_10, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf83
        del primals_46
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf87 = empty((512, ), device='cuda', dtype=torch.float32)
        buf88 = empty((512, ), device='cuda', dtype=torch.float32)
        buf94 = empty((512, ), device='cuda', dtype=torch.float32)
        buf89 = buf88; del buf88  # reuse
        buf95 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_27.run(buf89, buf95, relu_10, buf50, buf85, convolution_14, primals_120, convolution_13, primals_117, primals_121, primals_118, buf87, 512, 3136, grid=grid(512), stream=stream0)
        del convolution_13
        del convolution_14
        del primals_117
        del primals_120
        buf90 = buf76; del buf76  # reuse
        buf96 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(relu_10, buf50, buf85, primals_121, primals_44, primals_118, primals_41, buf90, buf96, 1605632, grid=grid(1605632), stream=stream0)
        del buf50
        del buf85
        del primals_118
        del primals_121
        del primals_41
        del primals_44
        del relu_10
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf91 = aten.convolution_backward(buf90, avg_pool2d_1, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_1
        del buf90
        del primals_43
        buf92 = buf91[0]
        buf93 = buf91[1]
        del buf91
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf97 = aten.convolution_backward(buf96, avg_pool2d, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d
        del primals_40
        buf98 = buf97[0]
        buf99 = buf97[1]
        del buf97
        buf100 = reinterpret_tensor(buf96, (4, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_29.run(buf98, buf100, 1605632, grid=grid(1605632), stream=stream0)
        del buf98
        buf101 = reinterpret_tensor(buf71, (4, 2, 128, 1, 1), (256, 128, 1, 1, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf100, relu_8, buf101, 1024, 3136, grid=grid(1024), stream=stream0)
        buf102 = empty((4, 2, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_31.run(buf101, div_1, buf102, 1024, grid=grid(1024), stream=stream0)
        del buf101
        buf103 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_9.run(buf102, buf103, 256, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf104 = aten.convolution_backward(reinterpret_tensor(buf102, (4, 256, 1, 1), (256, 1, 0, 0), 0), relu_9, primals_38, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf102
        del primals_38
        buf105 = buf104[0]
        buf106 = buf104[1]
        del buf104
        buf107 = empty((64, ), device='cuda', dtype=torch.float32)
        buf108 = empty((64, ), device='cuda', dtype=torch.float32)
        buf110 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_32.run(buf110, relu_9, buf105, convolution_11, primals_114, primals_115, buf107, 64, grid=grid(64), stream=stream0)
        del convolution_11
        del primals_114
        buf109 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_33.run(buf109, relu_9, primals_115, primals_36, 256, grid=grid(256), stream=stream0)
        del primals_115
        del primals_36
        del relu_9
        buf111 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_34.run(buf109, buf111, 64, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf112 = aten.convolution_backward(buf109, mean_1, primals_34, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_34
        buf113 = buf112[0]
        buf114 = buf112[1]
        del buf112
        buf115 = reinterpret_tensor(buf109, (256, ), (1, ), 0); del buf109  # reuse
        buf116 = empty((256, ), device='cuda', dtype=torch.float32)
        buf117 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(buf117, relu_8, buf100, div_1, buf113, convolution_10, primals_111, primals_112, buf115, 256, 12544, grid=grid(256), stream=stream0)
        del convolution_10
        del primals_111
        buf118 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36.run(relu_8, buf100, div_1, buf113, primals_112, primals_32, buf118, 3211264, grid=grid(3211264), stream=stream0)
        del buf100
        del div_1
        del primals_112
        del primals_32
        del relu_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf119 = aten.convolution_backward(buf118, relu_7, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_31
        buf120 = buf119[0]
        buf121 = buf119[1]
        del buf119
        buf122 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        buf124 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_7, buf120, convolution_9, primals_108, buf122, buf124, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_9
        del primals_108
        buf123 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf122, buf123, 128, 2, grid=grid(128), stream=stream0)
        buf125 = empty((128, ), device='cuda', dtype=torch.float32)
        buf126 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf126, buf124, primals_109, 128, 2, grid=grid(128), stream=stream0)
        buf127 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(buf127, relu_7, primals_109, primals_29, 1605632, grid=grid(1605632), stream=stream0)
        del primals_109
        del primals_29
        del relu_7
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf128 = aten.convolution_backward(buf127, relu_6, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_28
        buf129 = buf128[0]
        buf130 = buf128[1]
        del buf128
        buf131 = reinterpret_tensor(buf124, (256, ), (1, ), 0); del buf124  # reuse
        buf132 = reinterpret_tensor(buf122, (256, ), (1, ), 0); del buf122  # reuse
        buf138 = empty((256, ), device='cuda', dtype=torch.float32)
        buf133 = buf132; del buf132  # reuse
        buf139 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_41.run(buf133, buf139, relu_6, buf92, buf129, convolution_8, primals_105, convolution_7, primals_102, primals_106, primals_103, buf131, 256, 12544, grid=grid(256), stream=stream0)
        del convolution_7
        del convolution_8
        del primals_102
        del primals_105
        buf134 = buf118; del buf118  # reuse
        buf140 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_42.run(relu_6, buf92, buf129, primals_106, primals_26, primals_103, primals_23, buf134, buf140, 3211264, grid=grid(3211264), stream=stream0)
        del buf129
        del buf92
        del primals_103
        del primals_106
        del primals_23
        del primals_26
        del relu_6
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf135 = aten.convolution_backward(buf134, getitem, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_25
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf141 = aten.convolution_backward(buf140, sum_3, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        del sum_3
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = reinterpret_tensor(buf113, (4, 2, 64, 1, 1), (128, 64, 1, 1, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_43.run(buf142, relu_4, buf144, 512, 3136, grid=grid(512), stream=stream0)
        buf145 = empty((4, 2, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_44.run(buf144, div, buf145, 512, grid=grid(512), stream=stream0)
        del buf144
        buf146 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_22.run(buf145, buf146, 128, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf147 = aten.convolution_backward(reinterpret_tensor(buf145, (4, 128, 1, 1), (128, 1, 0, 0), 0), relu_5, primals_20, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf145
        del primals_20
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf150 = empty((32, ), device='cuda', dtype=torch.float32)
        buf151 = empty((32, ), device='cuda', dtype=torch.float32)
        buf153 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_45.run(buf153, relu_5, buf148, convolution_5, primals_99, primals_100, buf150, 32, grid=grid(32), stream=stream0)
        del convolution_5
        del primals_99
        buf152 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_46.run(buf152, relu_5, primals_100, primals_18, 128, grid=grid(128), stream=stream0)
        del primals_100
        del primals_18
        del relu_5
        buf154 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_47.run(buf152, buf154, 32, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf155 = aten.convolution_backward(buf152, mean, primals_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_16
        buf156 = buf155[0]
        buf157 = buf155[1]
        del buf155
        buf158 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        buf160 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_4, buf142, div, buf156, convolution_4, primals_96, buf158, buf160, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_4
        del primals_96
        buf159 = reinterpret_tensor(buf152, (128, ), (1, ), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf158, buf159, 128, 2, grid=grid(128), stream=stream0)
        del buf158
        buf161 = empty((128, ), device='cuda', dtype=torch.float32)
        buf162 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf162, buf160, primals_97, 128, 2, grid=grid(128), stream=stream0)
        del buf160
        buf163 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_4, buf142, div, buf156, primals_97, primals_14, buf163, 1605632, grid=grid(1605632), stream=stream0)
        del buf142
        del buf156
        del div
        del primals_14
        del primals_97
        del relu_4
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf164 = aten.convolution_backward(buf163, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf163
        del primals_13
        buf165 = buf164[0]
        buf166 = buf164[1]
        del buf164
        buf167 = empty_strided((64, 2), (1, 64), device='cuda', dtype=torch.float32)
        buf169 = empty_strided((64, 2), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_50.run(relu_3, buf165, convolution_3, primals_93, buf167, buf169, 128, 6272, grid=grid(128), stream=stream0)
        del convolution_3
        del primals_93
        buf168 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_51.run(buf167, buf168, 64, 2, grid=grid(64), stream=stream0)
        del buf167
        buf170 = empty((64, ), device='cuda', dtype=torch.float32)
        buf171 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_52.run(buf171, buf169, primals_94, 64, 2, grid=grid(64), stream=stream0)
        del buf169
        buf172 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53.run(buf172, relu_3, primals_94, primals_11, 802816, grid=grid(802816), stream=stream0)
        del primals_11
        del primals_94
        del relu_3
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf173 = aten.convolution_backward(buf172, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf172
        del getitem
        del primals_10
        buf174 = buf173[0]
        buf175 = buf173[1]
        del buf173
        buf176 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(buf176, buf174, 802816, grid=grid(802816), stream=stream0)
        del buf174
        buf177 = reinterpret_tensor(buf140, (4, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf140  # reuse
        buf183 = reinterpret_tensor(buf134, (4, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_55.run(getitem_1, buf176, relu_2, primals_91, primals_8, buf177, buf183, 3211264, grid=grid(3211264), stream=stream0)
        del buf176
        del getitem_1
        del primals_8
        buf178 = empty((64, 7), device='cuda', dtype=torch.float32)
        buf180 = empty((64, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_56.run(relu_2, buf177, convolution_2, primals_90, buf178, buf180, 448, 7168, grid=grid(448), stream=stream0)
        del buf177
        del convolution_2
        del primals_90
        del relu_2
        buf179 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_57.run(buf178, buf179, 64, 7, grid=grid(64), stream=stream0)
        del buf178
        buf181 = empty((64, ), device='cuda', dtype=torch.float32)
        buf182 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_58.run(buf182, buf180, primals_91, 64, 7, grid=grid(64), stream=stream0)
        del buf180
        del primals_91
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf184 = aten.convolution_backward(buf183, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf183
        del primals_7
        buf185 = buf184[0]
        buf186 = buf184[1]
        del buf184
        buf187 = empty((32, 7), device='cuda', dtype=torch.float32)
        buf189 = empty((32, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(relu_1, buf185, convolution_1, primals_87, buf187, buf189, 224, 7168, grid=grid(224), stream=stream0)
        del convolution_1
        del primals_87
        buf188 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_60.run(buf187, buf188, 32, 7, grid=grid(32), stream=stream0)
        buf190 = empty((32, ), device='cuda', dtype=torch.float32)
        buf191 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_61.run(buf191, buf189, primals_88, 32, 7, grid=grid(32), stream=stream0)
        buf192 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62.run(buf192, relu_1, primals_88, primals_5, 1605632, grid=grid(1605632), stream=stream0)
        del primals_5
        del primals_88
        del relu_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf193 = aten.convolution_backward(buf192, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf192
        del primals_4
        buf194 = buf193[0]
        buf195 = buf193[1]
        del buf193
        buf196 = buf189; del buf189  # reuse
        buf198 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(relu, buf194, convolution, primals_84, buf196, buf198, 224, 7168, grid=grid(224), stream=stream0)
        del convolution
        del primals_84
        buf197 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_60.run(buf196, buf197, 32, 7, grid=grid(32), stream=stream0)
        del buf196
        buf199 = empty((32, ), device='cuda', dtype=torch.float32)
        buf200 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_61.run(buf200, buf198, primals_85, 32, 7, grid=grid(32), stream=stream0)
        del buf198
        buf201 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62.run(buf201, relu, primals_85, primals_2, 1605632, grid=grid(1605632), stream=stream0)
        del primals_2
        del primals_85
        del relu
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf202 = aten.convolution_backward(buf201, primals_153, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf201
        del primals_1
        del primals_153
        buf203 = buf202[1]
        return (buf203, buf200, buf197, buf195, buf191, buf188, buf186, buf182, buf179, buf175, buf171, buf168, buf166, buf162, buf159, buf157, buf154, buf153, buf150, buf149, buf146, buf143, buf139, buf131, buf137, buf133, buf131, buf130, buf126, buf123, buf121, buf117, buf115, buf114, buf111, buf110, buf107, buf106, buf103, buf99, buf95, buf87, buf93, buf89, buf87, buf86, buf82, buf80, buf79, buf75, buf73, buf72, buf69, buf68, buf65, buf64, buf61, buf57, buf53, buf45, buf51, buf47, buf45, buf44, buf40, buf38, buf37, buf33, buf31, buf30, buf27, buf26, buf23, buf22, buf19, buf15, buf11, buf3, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((4, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((4, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((4, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((4, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.int64)
    convolution_3 = rand_strided((4, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((4, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((4, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((4, 2, 1, 64), (128, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    sum_3 = rand_strided((4, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((4, 2, 1, 128), (256, 128, 128, 1), device='cuda:0', dtype=torch.float32)
    sum_6 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((4, 128, 28, 28), (100352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((4, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((4, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((4, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_9 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((4, 256, 14, 14), (50176, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_3 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((4, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((4, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((4, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((4, 2, 1, 512), (1024, 512, 512, 1), device='cuda:0', dtype=torch.float32)
    sum_12 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_4 = rand_strided((4, 512, 7, 7), (25088, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_5 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_5 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_54, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_72, primals_74, primals_76, primals_77, primals_79, primals_80, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, getitem, getitem_1, convolution_3, relu_3, convolution_4, relu_4, mean, convolution_5, relu_5, div, sum_3, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, mean_1, convolution_11, relu_9, div_1, sum_6, avg_pool2d, convolution_13, avg_pool2d_1, convolution_14, relu_10, convolution_15, relu_11, convolution_16, relu_12, mean_2, convolution_17, relu_13, div_2, sum_9, avg_pool2d_2, convolution_19, avg_pool2d_3, convolution_20, relu_14, convolution_21, relu_15, convolution_22, relu_16, mean_3, convolution_23, relu_17, div_3, sum_12, avg_pool2d_4, convolution_25, avg_pool2d_5, convolution_26, view_24, permute_5, le, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_resnest', benchmark_compiled_module)
