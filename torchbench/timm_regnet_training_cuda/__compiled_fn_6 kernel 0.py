
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


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqrvvirqeflvbk7pfp34gwq3xjyrkjaakkcyxbd4ljenpbiw2mw.py
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
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2240
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (109760*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2240*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (49*x0) + (109760*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (r1 + (49*x0) + (109760*r2)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp20 = tmp18 - tmp19
    tmp21 = tmp5 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = tl.math.rsqrt(tmp28)
    tmp30 = tmp17 * tmp29
    tmp32 = tmp31 + tmp27
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp25 * tmp33
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp30, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp34, xmask)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hytf3wlkqwusut36ackan2ud7iqzmc7it7dgrxxo3ohrtr7ar6.py
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
    xnumel = 439040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2240
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxbrfnshhpgjxf4nycihqymjbww4i6vmox7fy3ud7kscrawnt2y.py
# Source Nodes: [sigmoid_18], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_18 => sigmoid_18
triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
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
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tmp6 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vp/cvpvw3vsne3zjedydyeb3uavbjvyye75l3vrctvj7m2ss5jdrrit.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (2240 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (4480 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (6720 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bf/cbf2z3jjubxjfrz4bhpxlaclsnxnjl7xbxjyltvpj4j525e646mp.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdtb5amcpfccdzlw2v63eodlyxi5t4xbkxdna22ogs6v2n3u3ji.py
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
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (224 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (448 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (672 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozvcnswhvvgk5x426mluwcd6isk7fpixi3fn3hnqs2on6ewl3h7.py
# Source Nodes: [sigmoid_18], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_18 => sigmoid_18
triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2240
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (109760*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (109760*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (x0 + (2240*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0 + (2240*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (49*x0) + (109760*r2)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 49.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp18 = tmp16 - tmp17
    tmp19 = tmp11 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = tl.math.rsqrt(tmp26)
    tmp28 = tmp23 * tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp28, xmask)
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqsabuzg4o3b25wsc5wtxmblpyxoj72fjmd6wc7luaxfy7kxi66.py
# Source Nodes: [sigmoid_18], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_18 => sigmoid_18
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 439040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2240
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 49.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/cey55hd7u73xjulm4c7z53ydeofmyxocdaftdfxa4injp5bpukiu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 2240
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (439040*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (439040*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (196*x0) + (439040*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ay/caylh4qhjuxpxtjgvyx4od5ipksox5gwmdlzrdveccuxvgxacfnf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1756160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2240
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpd7nnsgca7kvv7pouzw4bbbule6oiagax6jbeyp5jpwndk6lfga.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 896
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2pxgaoequrqzjd2dlihrnnsmaqi2tqmmzabzvovvj6ah67gtoj.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceusgrw42uhtlxppwlrlzpehechwggfepvkmucp3pi4eur626iyr.py
# Source Nodes: [sigmoid_17], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_17 => sigmoid_17
triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
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
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tmp6 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pcmho2ay2uytsklqdwv3wot7twgn3oqwgcnwicefptf2k422ct.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (896 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (1792 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (2688 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cofhcciedtzvwwoyqxrfurchlpevixqz7tyfb5ifwjectnnd7a4v.py
# Source Nodes: [sigmoid_17], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_17 => sigmoid_17
triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel):
    xnumel = 896
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (x0 + (896*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0 + (896*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 196.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp18 = tmp16 - tmp17
    tmp19 = tmp11 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = tl.math.rsqrt(tmp26)
    tmp28 = tmp23 * tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp28, xmask)
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpf6tbnqfwngia3n55nd4za3vf77yz75ya53fm2p4flfctfjtjfg.py
# Source Nodes: [sigmoid_17], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_17 => sigmoid_17
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 196.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkhcoq6n34iepfyx62zzxcbn5ofiwljzubqmnptxxmzq72w7im3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 896
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/h5/ch55tireve65xf4efkujp4dgypnts3u6nknsjs33idxxi22oqpag.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
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


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3e25zhyi5il2jgxqfxubkeavv5q5343iywjlo3aseite3wksfx.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 196) % 896
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp6 = tl.load(in_out_ptr0 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cuxlkzerqq4rt3cnmymdmtrqqq3btj6soboh2mbsxju6uppdnmoq.py
# Source Nodes: [sigmoid_16], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_16 => sigmoid_16
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 196.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmv2g2jk6iji5mj5xndmjhocicfip2xch4kddczgkm4mx2wwmjeu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pa/cpanzhhkljj7opgcibhnlgovbjmog36xakjv2qqn5wxhxapbhviy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 896
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 0.0
    tmp15 = tmp13 <= tmp14
    tmp17 = tmp0 + tmp16
    tmp18 = tl.where(tmp15, tmp14, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp25 = tmp23 - tmp24
    tmp26 = tmp18 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp12 * tmp34
    tmp37 = tmp36 + tmp32
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp35, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp39, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7soi5ygankwyuvvi3k7tnwwuxfsx6krqdr4noaevpmdlnydjlfg.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 196) % 896
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/contbt5zntk3ubxd4uyqy54ho3j6gtyfd2tgsdm6uji5lgk4mymr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 896
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 0.0
    tmp15 = tmp13 <= tmp14
    tmp17 = tmp0 + tmp16
    tmp18 = tl.where(tmp15, tmp14, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp25 = tmp23 - tmp24
    tmp26 = tmp18 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp33 = tmp31 - tmp32
    tmp34 = tmp18 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = tl.math.rsqrt(tmp41)
    tmp43 = tmp12 * tmp42
    tmp45 = tmp44 + tmp40
    tmp46 = tl.math.rsqrt(tmp45)
    tmp47 = tmp30 * tmp46
    tmp49 = tmp48 + tmp40
    tmp50 = tl.math.rsqrt(tmp49)
    tmp51 = tmp38 * tmp50
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp43, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp47, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp51, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/caph5tgq3axaq2jmazvikmb7lpkklarytxm7qiklpqumtbtyipcm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tmp15 = tmp14 + tmp8
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp6 * tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/cifmumnkp3fd3sf3ktf3n2qpudmyfmo7fqhb5rjaugbw7mesuwqi.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gtvrzwqygbgvow4snttzol4z3ed6oynhel7wfp372a623tv4ui.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (112 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (224 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (336 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqakjzgodcqrx5kplqdwtlfjuxpb6yompx5dilw52qpwffbzwcz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (702464*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (702464*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (702464*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdpwhyihuxfodlytxnyjzcmtg2pyr7uib6srj5f4tcld3ha3zij.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 896
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


# kernel path: /tmp/torchinductor_youkaichao/oq/coqs2zyad4sgftqgbdywnltebqsbba3spbxbj2scjjxpwb43shka.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp15 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/chalc4jigd22nci3xhks7zwbwgpbdrokj65dfyvi7j4duooowdei.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cenhd47btr6sg64tcxqf3n6wdn22zp4i4etjq3sezgraawiqzisw.py
# Source Nodes: [sigmoid_6], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_6 => sigmoid_6
triton_per_fused_mul_sigmoid_sigmoid_backward_sum_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_sum_32', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
    xnumel = 1792
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (784*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tmp6 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/can3dx5ov55mnnhx2e622mkejrghtztjkl6sioslja5rssmigpz5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (448 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (896 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (1344 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7k/c7kkjp2hxskw5l4wyll6i7il426d4id6kgztqfe6mp55dlwzmocm.py
# Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_6 => sigmoid_6
triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (448*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (448*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp8 = 784.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cs/ccs5kvbz4p2up6qaqqojyhesvaa5scxllbtwmm447nw63qj64aoc.py
# Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_6 => sigmoid_6
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 448
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 784.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4et7edic5my5gau56af5mytvql5yvtwrm2pteassxtqd4tznwg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ma/cma4whmoi7jgjhpner3gv7rdv3jzpjawiysaa4svd6tqqapbbdwu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
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


# kernel path: /tmp/torchinductor_youkaichao/yr/cyr4ddxej5quvxn6p6pahkgcrn65y3b6vfgzjzll7gbjryua4teq.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 784) % 448
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpylbyqcwqhkvkqt6gpix5c5gggl4fuuhdxgx2pmxlgvs36uni27.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_5 => sigmoid_5
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 448
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 784.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xm/cxm2s3aodgvfjtfy4sbwsbfqtacep6wmfwdxgzvg7stbvma6e4sp.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdftcmqoyl42mpj5k4boa75hk47neiponyi4lmfj3diwfqchmm7k.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = 0.0
        tmp13 = tmp11 <= tmp12
        tmp15 = tmp0 + tmp14
        tmp16 = tl.where(tmp13, tmp12, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp27 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp9 * tmp30
    tmp33 = tmp32 + tmp28
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp25 * tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7a/c7adm4uveuynli2nv7gbu26cwk53spn4dtpcrsjpfggfzr3znlmu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp28 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr7 + (r1 + (784*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = 0.0
        tmp13 = tmp11 <= tmp12
        tmp15 = tmp0 + tmp14
        tmp16 = tl.where(tmp13, tmp12, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tmp29 = tmp27 - tmp28
        tmp30 = tmp16 * tmp29
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tmp34 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp9 * tmp37
    tmp40 = tmp39 + tmp35
    tmp41 = tl.math.rsqrt(tmp40)
    tmp42 = tmp25 * tmp41
    tmp44 = tmp43 + tmp35
    tmp45 = tl.math.rsqrt(tmp44)
    tmp46 = tmp32 * tmp45
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp38, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp42, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp46, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dpsjqk6darvfdeq3ofwis5fxnmqoksbrmm3o32l27pps4zxajk.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tmp15 = tmp14 + tmp8
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp6 * tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/czefm5ibhvrjebj7izqzvrizx35rckyndtjaicngkjamrwxlhk3z.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv5hyrnx6sc3sw37k27bdvywavq53vbm7l6j7wbldej2suc3hooi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (56 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (112 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (168 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/26/c26m76dcvl7ybv4t6gcrvs5cwdmtynn6ku5ewcr3jmz7f2qupynx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 12544
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1404928*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (1404928*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (1404928*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/km/ckmko37nzrgosfc5e3qmu6beiqe4hqsjzecauckuzqbgg5igjzpq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5619712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 448
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


# kernel path: /tmp/torchinductor_youkaichao/hv/chvyl5npaidgyxl75nvogphlsf5r6rg6okexi6y7yzwz7rgzip4j.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p6/cp63cuu3iyumv32b63s2k5v6fwrru6lgbnljuprabparua43io2g.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czo2zv7whtc2m6nzxpp3bojuf32xne2lzhmlkga4vqxehtcdqe3a.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ab/cabupwjlzbnmmt2ihgbbaflofrorremwy54k2n3wvzjfwsgkfcdz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 224
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhaachwkdoblc5tcpaixn6vxhdnvf5hrnxibwhmu7ldsjs2od75.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_1 => sigmoid_1
triton_red_fused_mul_sigmoid_sigmoid_backward_sum_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sigmoid_sigmoid_backward_sum_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwilbmxnoawnpsmcytbmljzqnv62jcrxzibrrrtn5m3o3xhcrvxl.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_1 => sigmoid_1
triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (224*(r2 // 3136)) + (448*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (224*(r2 // 3136)) + (448*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp8 = 3136.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vobru2a7y56f772xlghu6ot5caujb4d2xk63htn4fbvdlk7lpn.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
# sigmoid_1 => sigmoid_1
triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 3136)
    x1 = (xindex // 3136) % 224
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 3136.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czas5ephbsvqtbvd2ycwokxqe7zfu3ad44df2bih7fbniga2i4iu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4l/c4luiv2leps7fddsg7fvyuw7o3u5qkbg45n3am7cjhrschtihf6e.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 224
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


# kernel path: /tmp/torchinductor_youkaichao/gl/cglxryhymm6einp7brxmvxdhrr5gdlliq4tsphhcugg2ngv5yzrt.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 3136) % 224
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19 + tmp13
    tmp21 = tl.math.rsqrt(tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp11 * tmp23
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kh/ckh44rmvc5qdf4k2rmp2hj2jhq62yeyou5e3l7txjzpjauqnzfge.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + ((3136*x0) + (702464*(r2 // 3136)) + (1404928*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp0 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyu63n3bxnitwmgoeph7ffxznh2own4jsfujrscxebgbvxknng2.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5c/c5cbzyfb5mcgjnr4i6gkrkkoy3fvfm5r2iobxg2bilcj5px2nsy3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (8 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (16 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (24 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcde7e77udrcr4hbrd7ignpa23sscc5gktordvbotxoqhjsob22.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x2 = xindex % 224
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (12544*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (12544*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (12544*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = tl.math.rsqrt(tmp17)
        tmp20 = tmp18 * tmp19
        tmp21 = tmp4 * tmp20
        tl.store(out_ptr2 + (r1 + (12544*x0)), tmp21, rmask & xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iy/ciy6guwijvb7nhrd75hrh5a5ykt6nltagpz3qjw37x7fdpjgh24p.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4vx5r2gaule6s25bs72ju5d255rymn6zyd3hxnmnbf62apbn4f.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_63', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/g6/cg6ukm47jm4zrf4cikq7ubkrknqs33yxzh2ls2y3et3fghlcsq2q.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (401408*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (401408*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (401408*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (401408*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77j3k6iiopr4zprdwhqeedyrm6zp3hj5epkuctmkv5sqbg2tcpt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5ibuazthn56gmisyblcb4u5jv3hck5q2l46tsz4vhtrrjozyzr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/yg/cygqpnp3kkw3fpf65kfyinvevhqc5tbzcq4vmfsi7bokuptavg5p.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_198, primals_199, primals_200, primals_201, primals_203, primals_205, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_233, primals_234, primals_235, primals_236, primals_238, primals_240, primals_241, primals_242, primals_243, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_254, primals_255, primals_256, primals_257, primals_259, primals_261, primals_262, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, mean, relu_3, convolution_4, mul_9, convolution_5, convolution_6, relu_4, convolution_7, relu_5, convolution_8, relu_6, mean_1, relu_7, convolution_10, mul_22, convolution_11, relu_8, convolution_12, relu_9, convolution_13, relu_10, mean_2, relu_11, convolution_15, mul_32, convolution_16, convolution_17, relu_12, convolution_18, relu_13, convolution_19, relu_14, mean_3, relu_15, convolution_21, mul_45, convolution_22, relu_16, convolution_23, relu_17, convolution_24, relu_18, mean_4, relu_19, convolution_26, mul_55, convolution_27, relu_20, convolution_28, relu_21, convolution_29, relu_22, mean_5, relu_23, convolution_31, mul_65, convolution_32, relu_24, convolution_33, relu_25, convolution_34, relu_26, mean_6, relu_27, convolution_36, mul_75, convolution_37, relu_28, convolution_38, relu_29, convolution_39, relu_30, mean_7, relu_31, convolution_41, mul_85, convolution_42, convolution_43, relu_32, convolution_44, relu_33, convolution_45, relu_34, mean_8, relu_35, convolution_47, mul_98, convolution_48, relu_36, convolution_49, relu_37, convolution_50, relu_38, mean_9, relu_39, convolution_52, mul_108, convolution_53, relu_40, convolution_54, relu_41, convolution_55, relu_42, mean_10, relu_43, convolution_57, mul_118, convolution_58, relu_44, convolution_59, relu_45, convolution_60, relu_46, mean_11, relu_47, convolution_62, mul_128, convolution_63, relu_48, convolution_64, relu_49, convolution_65, relu_50, mean_12, relu_51, convolution_67, mul_138, convolution_68, relu_52, convolution_69, relu_53, convolution_70, relu_54, mean_13, relu_55, convolution_72, mul_148, convolution_73, relu_56, convolution_74, relu_57, convolution_75, relu_58, mean_14, relu_59, convolution_77, mul_158, convolution_78, relu_60, convolution_79, relu_61, convolution_80, relu_62, mean_15, relu_63, convolution_82, mul_168, convolution_83, relu_64, convolution_84, relu_65, convolution_85, relu_66, mean_16, relu_67, convolution_87, mul_178, convolution_88, relu_68, convolution_89, relu_69, convolution_90, relu_70, mean_17, relu_71, convolution_92, mul_188, convolution_93, relu_72, convolution_94, relu_73, convolution_95, relu_74, mean_18, relu_75, convolution_97, mul_198, convolution_98, convolution_99, clone, permute_1, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (224, ), (1, ))
    assert_size_stride(primals_5, (224, ), (1, ))
    assert_size_stride(primals_7, (224, ), (1, ))
    assert_size_stride(primals_9, (224, ), (1, ))
    assert_size_stride(primals_11, (224, ), (1, ))
    assert_size_stride(primals_13, (224, ), (1, ))
    assert_size_stride(primals_15, (224, ), (1, ))
    assert_size_stride(primals_17, (448, ), (1, ))
    assert_size_stride(primals_19, (448, ), (1, ))
    assert_size_stride(primals_21, (448, ), (1, ))
    assert_size_stride(primals_23, (448, ), (1, ))
    assert_size_stride(primals_25, (448, ), (1, ))
    assert_size_stride(primals_27, (448, ), (1, ))
    assert_size_stride(primals_29, (448, ), (1, ))
    assert_size_stride(primals_31, (448, ), (1, ))
    assert_size_stride(primals_33, (448, ), (1, ))
    assert_size_stride(primals_35, (448, ), (1, ))
    assert_size_stride(primals_37, (448, ), (1, ))
    assert_size_stride(primals_39, (448, ), (1, ))
    assert_size_stride(primals_41, (448, ), (1, ))
    assert_size_stride(primals_43, (448, ), (1, ))
    assert_size_stride(primals_45, (448, ), (1, ))
    assert_size_stride(primals_47, (448, ), (1, ))
    assert_size_stride(primals_49, (896, ), (1, ))
    assert_size_stride(primals_51, (896, ), (1, ))
    assert_size_stride(primals_53, (896, ), (1, ))
    assert_size_stride(primals_55, (896, ), (1, ))
    assert_size_stride(primals_57, (896, ), (1, ))
    assert_size_stride(primals_59, (896, ), (1, ))
    assert_size_stride(primals_61, (896, ), (1, ))
    assert_size_stride(primals_63, (896, ), (1, ))
    assert_size_stride(primals_65, (896, ), (1, ))
    assert_size_stride(primals_67, (896, ), (1, ))
    assert_size_stride(primals_69, (896, ), (1, ))
    assert_size_stride(primals_71, (896, ), (1, ))
    assert_size_stride(primals_73, (896, ), (1, ))
    assert_size_stride(primals_75, (896, ), (1, ))
    assert_size_stride(primals_77, (896, ), (1, ))
    assert_size_stride(primals_79, (896, ), (1, ))
    assert_size_stride(primals_81, (896, ), (1, ))
    assert_size_stride(primals_83, (896, ), (1, ))
    assert_size_stride(primals_85, (896, ), (1, ))
    assert_size_stride(primals_87, (896, ), (1, ))
    assert_size_stride(primals_89, (896, ), (1, ))
    assert_size_stride(primals_91, (896, ), (1, ))
    assert_size_stride(primals_93, (896, ), (1, ))
    assert_size_stride(primals_95, (896, ), (1, ))
    assert_size_stride(primals_97, (896, ), (1, ))
    assert_size_stride(primals_99, (896, ), (1, ))
    assert_size_stride(primals_101, (896, ), (1, ))
    assert_size_stride(primals_103, (896, ), (1, ))
    assert_size_stride(primals_105, (896, ), (1, ))
    assert_size_stride(primals_107, (896, ), (1, ))
    assert_size_stride(primals_109, (896, ), (1, ))
    assert_size_stride(primals_111, (896, ), (1, ))
    assert_size_stride(primals_113, (896, ), (1, ))
    assert_size_stride(primals_115, (896, ), (1, ))
    assert_size_stride(primals_117, (2240, ), (1, ))
    assert_size_stride(primals_119, (2240, ), (1, ))
    assert_size_stride(primals_121, (2240, ), (1, ))
    assert_size_stride(primals_123, (2240, ), (1, ))
    assert_size_stride(primals_125, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_126, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_127, (224, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_128, (8, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_130, (224, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_132, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_133, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_134, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_135, (224, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_136, (56, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_138, (224, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_140, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_141, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_142, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_143, (56, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_145, (448, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_147, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_148, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_149, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_150, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_151, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_153, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_155, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_156, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_157, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_158, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_160, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_162, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_163, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_164, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_165, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_167, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_169, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_170, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_171, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_172, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_174, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_176, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_177, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_178, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_179, (112, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_181, (896, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_183, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_184, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_185, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_186, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_187, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_189, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_191, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_192, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_193, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_194, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_196, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_198, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_199, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_200, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_201, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_203, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_205, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_206, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_207, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_208, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_210, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_212, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_213, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_214, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_215, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_217, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_219, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_220, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_221, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_222, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_224, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_226, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_227, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_228, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_229, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_231, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_233, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_234, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_235, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_236, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_238, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_240, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_241, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_242, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_243, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_245, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_247, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_248, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_249, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_250, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_252, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_254, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_255, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_256, (2240, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(primals_257, (224, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_259, (2240, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_261, (2240, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_262, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, ), (1, ))
    assert_size_stride(primals_267, (224, ), (1, ))
    assert_size_stride(primals_268, (224, ), (1, ))
    assert_size_stride(primals_269, (224, ), (1, ))
    assert_size_stride(primals_270, (224, ), (1, ))
    assert_size_stride(primals_271, (224, ), (1, ))
    assert_size_stride(primals_272, (224, ), (1, ))
    assert_size_stride(primals_273, (224, ), (1, ))
    assert_size_stride(primals_274, (224, ), (1, ))
    assert_size_stride(primals_275, (224, ), (1, ))
    assert_size_stride(primals_276, (224, ), (1, ))
    assert_size_stride(primals_277, (224, ), (1, ))
    assert_size_stride(primals_278, (224, ), (1, ))
    assert_size_stride(primals_279, (224, ), (1, ))
    assert_size_stride(primals_280, (224, ), (1, ))
    assert_size_stride(primals_281, (448, ), (1, ))
    assert_size_stride(primals_282, (448, ), (1, ))
    assert_size_stride(primals_283, (448, ), (1, ))
    assert_size_stride(primals_284, (448, ), (1, ))
    assert_size_stride(primals_285, (448, ), (1, ))
    assert_size_stride(primals_286, (448, ), (1, ))
    assert_size_stride(primals_287, (448, ), (1, ))
    assert_size_stride(primals_288, (448, ), (1, ))
    assert_size_stride(primals_289, (448, ), (1, ))
    assert_size_stride(primals_290, (448, ), (1, ))
    assert_size_stride(primals_291, (448, ), (1, ))
    assert_size_stride(primals_292, (448, ), (1, ))
    assert_size_stride(primals_293, (448, ), (1, ))
    assert_size_stride(primals_294, (448, ), (1, ))
    assert_size_stride(primals_295, (448, ), (1, ))
    assert_size_stride(primals_296, (448, ), (1, ))
    assert_size_stride(primals_297, (448, ), (1, ))
    assert_size_stride(primals_298, (448, ), (1, ))
    assert_size_stride(primals_299, (448, ), (1, ))
    assert_size_stride(primals_300, (448, ), (1, ))
    assert_size_stride(primals_301, (448, ), (1, ))
    assert_size_stride(primals_302, (448, ), (1, ))
    assert_size_stride(primals_303, (448, ), (1, ))
    assert_size_stride(primals_304, (448, ), (1, ))
    assert_size_stride(primals_305, (448, ), (1, ))
    assert_size_stride(primals_306, (448, ), (1, ))
    assert_size_stride(primals_307, (448, ), (1, ))
    assert_size_stride(primals_308, (448, ), (1, ))
    assert_size_stride(primals_309, (448, ), (1, ))
    assert_size_stride(primals_310, (448, ), (1, ))
    assert_size_stride(primals_311, (448, ), (1, ))
    assert_size_stride(primals_312, (448, ), (1, ))
    assert_size_stride(primals_313, (896, ), (1, ))
    assert_size_stride(primals_314, (896, ), (1, ))
    assert_size_stride(primals_315, (896, ), (1, ))
    assert_size_stride(primals_316, (896, ), (1, ))
    assert_size_stride(primals_317, (896, ), (1, ))
    assert_size_stride(primals_318, (896, ), (1, ))
    assert_size_stride(primals_319, (896, ), (1, ))
    assert_size_stride(primals_320, (896, ), (1, ))
    assert_size_stride(primals_321, (896, ), (1, ))
    assert_size_stride(primals_322, (896, ), (1, ))
    assert_size_stride(primals_323, (896, ), (1, ))
    assert_size_stride(primals_324, (896, ), (1, ))
    assert_size_stride(primals_325, (896, ), (1, ))
    assert_size_stride(primals_326, (896, ), (1, ))
    assert_size_stride(primals_327, (896, ), (1, ))
    assert_size_stride(primals_328, (896, ), (1, ))
    assert_size_stride(primals_329, (896, ), (1, ))
    assert_size_stride(primals_330, (896, ), (1, ))
    assert_size_stride(primals_331, (896, ), (1, ))
    assert_size_stride(primals_332, (896, ), (1, ))
    assert_size_stride(primals_333, (896, ), (1, ))
    assert_size_stride(primals_334, (896, ), (1, ))
    assert_size_stride(primals_335, (896, ), (1, ))
    assert_size_stride(primals_336, (896, ), (1, ))
    assert_size_stride(primals_337, (896, ), (1, ))
    assert_size_stride(primals_338, (896, ), (1, ))
    assert_size_stride(primals_339, (896, ), (1, ))
    assert_size_stride(primals_340, (896, ), (1, ))
    assert_size_stride(primals_341, (896, ), (1, ))
    assert_size_stride(primals_342, (896, ), (1, ))
    assert_size_stride(primals_343, (896, ), (1, ))
    assert_size_stride(primals_344, (896, ), (1, ))
    assert_size_stride(primals_345, (896, ), (1, ))
    assert_size_stride(primals_346, (896, ), (1, ))
    assert_size_stride(primals_347, (896, ), (1, ))
    assert_size_stride(primals_348, (896, ), (1, ))
    assert_size_stride(primals_349, (896, ), (1, ))
    assert_size_stride(primals_350, (896, ), (1, ))
    assert_size_stride(primals_351, (896, ), (1, ))
    assert_size_stride(primals_352, (896, ), (1, ))
    assert_size_stride(primals_353, (896, ), (1, ))
    assert_size_stride(primals_354, (896, ), (1, ))
    assert_size_stride(primals_355, (896, ), (1, ))
    assert_size_stride(primals_356, (896, ), (1, ))
    assert_size_stride(primals_357, (896, ), (1, ))
    assert_size_stride(primals_358, (896, ), (1, ))
    assert_size_stride(primals_359, (896, ), (1, ))
    assert_size_stride(primals_360, (896, ), (1, ))
    assert_size_stride(primals_361, (896, ), (1, ))
    assert_size_stride(primals_362, (896, ), (1, ))
    assert_size_stride(primals_363, (896, ), (1, ))
    assert_size_stride(primals_364, (896, ), (1, ))
    assert_size_stride(primals_365, (896, ), (1, ))
    assert_size_stride(primals_366, (896, ), (1, ))
    assert_size_stride(primals_367, (896, ), (1, ))
    assert_size_stride(primals_368, (896, ), (1, ))
    assert_size_stride(primals_369, (896, ), (1, ))
    assert_size_stride(primals_370, (896, ), (1, ))
    assert_size_stride(primals_371, (896, ), (1, ))
    assert_size_stride(primals_372, (896, ), (1, ))
    assert_size_stride(primals_373, (896, ), (1, ))
    assert_size_stride(primals_374, (896, ), (1, ))
    assert_size_stride(primals_375, (896, ), (1, ))
    assert_size_stride(primals_376, (896, ), (1, ))
    assert_size_stride(primals_377, (896, ), (1, ))
    assert_size_stride(primals_378, (896, ), (1, ))
    assert_size_stride(primals_379, (896, ), (1, ))
    assert_size_stride(primals_380, (896, ), (1, ))
    assert_size_stride(primals_381, (2240, ), (1, ))
    assert_size_stride(primals_382, (2240, ), (1, ))
    assert_size_stride(primals_383, (2240, ), (1, ))
    assert_size_stride(primals_384, (2240, ), (1, ))
    assert_size_stride(primals_385, (2240, ), (1, ))
    assert_size_stride(primals_386, (2240, ), (1, ))
    assert_size_stride(primals_387, (2240, ), (1, ))
    assert_size_stride(primals_388, (2240, ), (1, ))
    assert_size_stride(primals_389, (4, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (4, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(relu, (4, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(convolution_1, (4, 224, 112, 112), (2809856, 12544, 112, 1))
    assert_size_stride(relu_1, (4, 224, 112, 112), (2809856, 12544, 112, 1))
    assert_size_stride(convolution_2, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(relu_2, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(mean, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(relu_3, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(convolution_4, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(mul_9, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(convolution_5, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(convolution_6, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(relu_4, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(convolution_7, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(relu_5, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(convolution_8, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(relu_6, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(mean_1, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(relu_7, (4, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(convolution_10, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(mul_22, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(convolution_11, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(relu_8, (4, 224, 56, 56), (702464, 3136, 56, 1))
    assert_size_stride(convolution_12, (4, 448, 56, 56), (1404928, 3136, 56, 1))
    assert_size_stride(relu_9, (4, 448, 56, 56), (1404928, 3136, 56, 1))
    assert_size_stride(convolution_13, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_10, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(mean_2, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(relu_11, (4, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(convolution_15, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(mul_32, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_16, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_17, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_12, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_18, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_13, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_19, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_14, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(mean_3, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(relu_15, (4, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(convolution_21, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(mul_45, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_22, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_16, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_23, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_17, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_24, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_18, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(mean_4, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(relu_19, (4, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(convolution_26, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(mul_55, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_27, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_20, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_28, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_21, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_29, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_22, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(mean_5, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(relu_23, (4, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(convolution_31, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(mul_65, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_32, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_24, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_33, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_25, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_34, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_26, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(mean_6, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(relu_27, (4, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(convolution_36, (4, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(mul_75, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_37, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(relu_28, (4, 448, 28, 28), (351232, 784, 28, 1))
    assert_size_stride(convolution_38, (4, 896, 28, 28), (702464, 784, 28, 1))
    assert_size_stride(relu_29, (4, 896, 28, 28), (702464, 784, 28, 1))
    assert_size_stride(convolution_39, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_30, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_7, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_31, (4, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(convolution_41, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_85, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_42, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_43, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_32, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_44, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_33, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_45, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_34, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_8, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_35, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_47, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_98, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_48, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_36, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_49, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_37, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_50, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_38, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_9, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_39, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_52, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_108, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_53, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_40, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_54, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_41, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_55, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_42, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_10, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_43, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_57, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_118, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_58, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_44, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_59, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_45, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_60, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_46, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_11, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_47, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_62, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_128, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_63, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_48, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_64, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_49, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_65, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_50, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_12, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_51, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_67, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_138, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_68, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_52, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_69, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_53, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_70, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_54, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_13, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_55, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_72, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_148, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_73, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_56, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_74, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_57, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_75, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_58, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_14, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_59, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_77, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_158, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_78, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_60, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_79, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_61, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_80, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_62, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_15, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_63, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_82, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_168, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_83, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_64, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_84, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_65, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_85, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_66, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_16, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_67, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_87, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_178, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_88, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_68, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_89, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_69, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_90, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_70, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(mean_17, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(relu_71, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_92, (4, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(mul_188, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_93, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(relu_72, (4, 896, 14, 14), (175616, 196, 14, 1))
    assert_size_stride(convolution_94, (4, 2240, 14, 14), (439040, 196, 14, 1))
    assert_size_stride(relu_73, (4, 2240, 14, 14), (439040, 196, 14, 1))
    assert_size_stride(convolution_95, (4, 2240, 7, 7), (109760, 49, 7, 1))
    assert_size_stride(relu_74, (4, 2240, 7, 7), (109760, 49, 7, 1))
    assert_size_stride(mean_18, (4, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(relu_75, (4, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(convolution_97, (4, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(mul_198, (4, 2240, 7, 7), (109760, 49, 7, 1))
    assert_size_stride(convolution_98, (4, 2240, 7, 7), (109760, 49, 7, 1))
    assert_size_stride(convolution_99, (4, 2240, 7, 7), (109760, 49, 7, 1))
    assert_size_stride(clone, (4, 2240), (2240, 1))
    assert_size_stride(permute_1, (1000, 2240), (2240, 1))
    assert_size_stride(le, (4, 2240, 7, 7), (109760, 49, 7, 1))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 2240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 2240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf4 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf10 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf5 = buf4; del buf4  # reuse
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(buf5, buf11, le, buf0, convolution_99, primals_387, convolution_98, primals_385, primals_388, primals_386, buf3, 2240, 196, grid=grid(2240), stream=stream0)
        del convolution_98
        del convolution_99
        del primals_385
        del primals_387
        buf6 = empty((4, 2240, 7, 7), device='cuda', dtype=torch.float32)
        buf12 = empty((4, 2240, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2.run(le, buf0, primals_388, primals_123, primals_386, primals_121, buf6, buf12, 439040, grid=grid(439040), stream=stream0)
        del le
        del primals_121
        del primals_123
        del primals_386
        del primals_388
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf7 = aten.convolution_backward(buf6, relu_72, primals_262, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf6
        del primals_262
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf13 = aten.convolution_backward(buf12, mul_198, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf12
        del mul_198
        del primals_261
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        buf16 = reinterpret_tensor(buf0, (4, 2240, 1, 1), (2240, 1, 8960, 8960), 0); del buf0  # reuse
        buf17 = reinterpret_tensor(buf16, (4, 2240, 1, 1), (2240, 1, 1, 1), 0); del buf16  # reuse
        # Source Nodes: [sigmoid_18], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_3.run(buf17, buf14, relu_74, convolution_97, 8960, 49, grid=grid(8960), stream=stream0)
        buf18 = empty((2240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_4.run(buf17, buf18, 2240, grid=grid(2240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf19 = aten.convolution_backward(buf17, relu_75, primals_259, [2240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf17
        del primals_259
        buf20 = buf19[0]
        buf21 = buf19[1]
        del buf19
        buf22 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf22, relu_75, 896, grid=grid(896), stream=stream0)
        del relu_75
        buf23 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf22, buf23, 224, grid=grid(224), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf24 = aten.convolution_backward(buf22, mean_18, primals_257, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_18
        del primals_257
        buf25 = buf24[0]
        buf26 = buf24[1]
        del buf24
        buf27 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf28 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf29 = buf28; del buf28  # reuse
        # Source Nodes: [sigmoid_18], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7.run(buf29, relu_74, buf14, convolution_97, buf25, convolution_95, primals_383, primals_384, buf27, 2240, 196, grid=grid(2240), stream=stream0)
        del convolution_95
        del primals_383
        buf30 = buf14; del buf14  # reuse
        # Source Nodes: [sigmoid_18], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8.run(buf30, relu_74, convolution_97, buf25, primals_384, primals_119, 439040, grid=grid(439040), stream=stream0)
        del buf25
        del convolution_97
        del primals_119
        del primals_384
        del relu_74
        # Source Nodes: [sigmoid_18], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf31 = aten.convolution_backward(buf30, relu_73, primals_256, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False])
        del buf30
        del primals_256
        buf32 = buf31[0]
        buf33 = buf31[1]
        del buf31
        buf34 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf35 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf36 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf36, relu_73, buf32, convolution_94, primals_381, primals_382, buf34, 2240, 784, grid=grid(2240), stream=stream0)
        del convolution_94
        del primals_381
        buf37 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf37, relu_73, primals_382, primals_117, 1756160, grid=grid(1756160), stream=stream0)
        del primals_117
        del primals_382
        del relu_73
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf38 = aten.convolution_backward(buf37, relu_72, primals_255, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf37
        del primals_255
        buf39 = buf38[0]
        buf40 = buf38[1]
        del buf38
        buf41 = reinterpret_tensor(buf22, (896, ), (1, ), 0); del buf22  # reuse
        buf42 = empty((896, ), device='cuda', dtype=torch.float32)
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_11.run(buf43, relu_72, buf8, buf39, convolution_93, primals_379, primals_380, buf41, 896, 784, grid=grid(896), stream=stream0)
        del convolution_93
        del primals_379
        buf44 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_72, buf8, buf39, primals_380, primals_115, buf44, 702464, grid=grid(702464), stream=stream0)
        del primals_115
        del primals_380
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf45 = aten.convolution_backward(buf44, mul_188, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf44
        del mul_188
        del primals_254
        buf46 = buf45[0]
        buf47 = buf45[1]
        del buf45
        buf48 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cuda', dtype=torch.float32)
        buf49 = reinterpret_tensor(buf48, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf48  # reuse
        # Source Nodes: [sigmoid_17], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf49, buf46, relu_70, convolution_92, 3584, 196, grid=grid(3584), stream=stream0)
        buf50 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf49, buf50, 896, grid=grid(896), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf51 = aten.convolution_backward(buf49, relu_71, primals_252, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf49
        del primals_252
        buf52 = buf51[0]
        buf53 = buf51[1]
        del buf51
        buf54 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf54, relu_71, 896, grid=grid(896), stream=stream0)
        del relu_71
        buf55 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf54, buf55, 224, grid=grid(224), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf56 = aten.convolution_backward(buf54, mean_17, primals_250, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_17
        del primals_250
        buf57 = buf56[0]
        buf58 = buf56[1]
        del buf56
        buf59 = reinterpret_tensor(buf54, (896, ), (1, ), 0); del buf54  # reuse
        buf60 = empty((896, ), device='cuda', dtype=torch.float32)
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [sigmoid_17], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf61, relu_70, buf46, convolution_92, buf57, convolution_90, primals_377, primals_378, buf59, 896, 784, grid=grid(896), stream=stream0)
        del convolution_90
        del primals_377
        buf62 = buf46; del buf46  # reuse
        # Source Nodes: [sigmoid_17], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_16.run(buf62, relu_70, convolution_92, buf57, primals_378, primals_113, 702464, grid=grid(702464), stream=stream0)
        del convolution_92
        del primals_113
        del primals_378
        del relu_70
        # Source Nodes: [sigmoid_17], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf63 = aten.convolution_backward(buf62, relu_69, primals_249, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf62
        del primals_249
        buf64 = buf63[0]
        buf65 = buf63[1]
        del buf63
        buf66 = empty((896, ), device='cuda', dtype=torch.float32)
        buf67 = empty((896, ), device='cuda', dtype=torch.float32)
        buf68 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf68, relu_69, buf64, convolution_89, primals_375, primals_376, buf66, 896, 784, grid=grid(896), stream=stream0)
        del convolution_89
        del primals_375
        buf69 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(buf69, relu_69, primals_376, primals_111, 702464, grid=grid(702464), stream=stream0)
        del primals_111
        del primals_376
        del relu_69
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf70 = aten.convolution_backward(buf69, relu_68, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_248
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = buf39; del buf39  # reuse
        buf77 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf73, relu_68, relu_72, buf8, buf71, primals_374, primals_109, buf77, 702464, grid=grid(702464), stream=stream0)
        del buf71
        del buf8
        del primals_109
        del relu_68
        del relu_72
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf78 = aten.convolution_backward(buf77, mul_178, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_178
        del primals_247
        buf79 = buf78[0]
        buf81 = reinterpret_tensor(buf57, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf57  # reuse
        buf82 = reinterpret_tensor(buf81, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf81  # reuse
        # Source Nodes: [sigmoid_16], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf82, buf79, relu_66, convolution_87, 3584, 196, grid=grid(3584), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf84 = aten.convolution_backward(buf82, relu_67, primals_245, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_245
        buf85 = buf84[0]
        buf87 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf87, relu_67, 896, grid=grid(896), stream=stream0)
        del relu_67
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf89 = aten.convolution_backward(buf87, mean_16, primals_243, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_16
        del primals_243
        buf90 = buf89[0]
        buf95 = buf77; del buf77  # reuse
        # Source Nodes: [sigmoid_16], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_20.run(relu_66, buf79, convolution_87, buf90, primals_372, primals_107, buf95, 702464, grid=grid(702464), stream=stream0)
        del primals_107
        # Source Nodes: [sigmoid_16], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf96 = aten.convolution_backward(buf95, relu_65, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del primals_242
        buf97 = buf96[0]
        buf102 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_65, buf97, primals_370, primals_105, buf102, 702464, grid=grid(702464), stream=stream0)
        del primals_105
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf103 = aten.convolution_backward(buf102, relu_64, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf102
        del primals_241
        buf104 = buf103[0]
        buf74 = empty((896, ), device='cuda', dtype=torch.float32)
        buf75 = empty((896, ), device='cuda', dtype=torch.float32)
        buf106 = empty((896, ), device='cuda', dtype=torch.float32)
        buf107 = empty((896, ), device='cuda', dtype=torch.float32)
        buf76 = buf75; del buf75  # reuse
        buf108 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf76, buf108, buf73, convolution_88, primals_373, relu_64, buf104, convolution_83, primals_367, primals_374, primals_368, buf74, buf106, 896, 784, grid=grid(896), stream=stream0)
        del convolution_83
        del convolution_88
        del primals_367
        del primals_373
        del primals_374
        buf80 = buf78[1]
        del buf78
        buf83 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf82, buf83, 896, grid=grid(896), stream=stream0)
        del buf82
        buf86 = buf84[1]
        del buf84
        buf88 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf87, buf88, 224, grid=grid(224), stream=stream0)
        buf91 = buf89[1]
        del buf89
        buf92 = reinterpret_tensor(buf87, (896, ), (1, ), 0); del buf87  # reuse
        buf93 = empty((896, ), device='cuda', dtype=torch.float32)
        buf94 = buf93; del buf93  # reuse
        # Source Nodes: [sigmoid_16], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf94, relu_66, buf79, convolution_87, buf90, convolution_85, primals_371, primals_372, buf92, 896, 784, grid=grid(896), stream=stream0)
        del buf79
        del convolution_85
        del convolution_87
        del primals_371
        del primals_372
        del relu_66
        buf98 = buf96[1]
        del buf96
        buf99 = empty((896, ), device='cuda', dtype=torch.float32)
        buf100 = empty((896, ), device='cuda', dtype=torch.float32)
        buf101 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf101, relu_65, buf97, convolution_84, primals_369, primals_370, buf99, 896, 784, grid=grid(896), stream=stream0)
        del convolution_84
        del primals_369
        del primals_370
        del relu_65
        buf105 = buf103[1]
        del buf103
        buf109 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_64, buf73, buf104, primals_368, primals_103, buf109, 702464, grid=grid(702464), stream=stream0)
        del primals_103
        del primals_368
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf110 = aten.convolution_backward(buf109, mul_168, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf109
        del mul_168
        del primals_240
        buf111 = buf110[0]
        buf112 = buf110[1]
        del buf110
        buf113 = reinterpret_tensor(buf90, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf90  # reuse
        buf114 = reinterpret_tensor(buf113, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf113  # reuse
        # Source Nodes: [sigmoid_15], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf114, buf111, relu_62, convolution_82, 3584, 196, grid=grid(3584), stream=stream0)
        buf115 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf114, buf115, 896, grid=grid(896), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf116 = aten.convolution_backward(buf114, relu_63, primals_238, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf114
        del primals_238
        buf117 = buf116[0]
        buf118 = buf116[1]
        del buf116
        buf119 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf119, relu_63, 896, grid=grid(896), stream=stream0)
        del relu_63
        buf120 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf119, buf120, 224, grid=grid(224), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf121 = aten.convolution_backward(buf119, mean_15, primals_236, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_15
        del primals_236
        buf122 = buf121[0]
        buf123 = buf121[1]
        del buf121
        buf124 = reinterpret_tensor(buf119, (896, ), (1, ), 0); del buf119  # reuse
        buf125 = empty((896, ), device='cuda', dtype=torch.float32)
        buf126 = buf125; del buf125  # reuse
        # Source Nodes: [sigmoid_15], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf126, relu_62, buf111, convolution_82, buf122, convolution_80, primals_365, primals_366, buf124, 896, 784, grid=grid(896), stream=stream0)
        del convolution_80
        del primals_365
        buf127 = buf111; del buf111  # reuse
        # Source Nodes: [sigmoid_15], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_16.run(buf127, relu_62, convolution_82, buf122, primals_366, primals_101, 702464, grid=grid(702464), stream=stream0)
        del convolution_82
        del primals_101
        del primals_366
        del relu_62
        # Source Nodes: [sigmoid_15], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf128 = aten.convolution_backward(buf127, relu_61, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf127
        del primals_235
        buf129 = buf128[0]
        buf130 = buf128[1]
        del buf128
        buf131 = empty((896, ), device='cuda', dtype=torch.float32)
        buf132 = empty((896, ), device='cuda', dtype=torch.float32)
        buf133 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf133, relu_61, buf129, convolution_79, primals_363, primals_364, buf131, 896, 784, grid=grid(896), stream=stream0)
        del convolution_79
        del primals_363
        buf134 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(buf134, relu_61, primals_364, primals_99, 702464, grid=grid(702464), stream=stream0)
        del primals_364
        del primals_99
        del relu_61
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf135 = aten.convolution_backward(buf134, relu_60, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_234
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        buf138 = buf104; del buf104  # reuse
        buf142 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf138, relu_60, relu_64, buf73, buf136, primals_362, primals_97, buf142, 702464, grid=grid(702464), stream=stream0)
        del buf136
        del buf73
        del primals_97
        del relu_60
        del relu_64
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf143 = aten.convolution_backward(buf142, mul_158, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_158
        del primals_233
        buf144 = buf143[0]
        buf146 = reinterpret_tensor(buf122, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf122  # reuse
        buf147 = reinterpret_tensor(buf146, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf146  # reuse
        # Source Nodes: [sigmoid_14], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf147, buf144, relu_58, convolution_77, 3584, 196, grid=grid(3584), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf149 = aten.convolution_backward(buf147, relu_59, primals_231, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_231
        buf150 = buf149[0]
        buf152 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf152, relu_59, 896, grid=grid(896), stream=stream0)
        del relu_59
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf154 = aten.convolution_backward(buf152, mean_14, primals_229, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_14
        del primals_229
        buf155 = buf154[0]
        buf160 = buf142; del buf142  # reuse
        # Source Nodes: [sigmoid_14], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_20.run(relu_58, buf144, convolution_77, buf155, primals_360, primals_95, buf160, 702464, grid=grid(702464), stream=stream0)
        del primals_95
        # Source Nodes: [sigmoid_14], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf161 = aten.convolution_backward(buf160, relu_57, primals_228, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del primals_228
        buf162 = buf161[0]
        buf167 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_57, buf162, primals_358, primals_93, buf167, 702464, grid=grid(702464), stream=stream0)
        del primals_93
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf168 = aten.convolution_backward(buf167, relu_56, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf167
        del primals_227
        buf169 = buf168[0]
        buf139 = empty((896, ), device='cuda', dtype=torch.float32)
        buf140 = empty((896, ), device='cuda', dtype=torch.float32)
        buf171 = empty((896, ), device='cuda', dtype=torch.float32)
        buf172 = empty((896, ), device='cuda', dtype=torch.float32)
        buf141 = buf140; del buf140  # reuse
        buf173 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf141, buf173, buf138, convolution_78, primals_361, relu_56, buf169, convolution_73, primals_355, primals_362, primals_356, buf139, buf171, 896, 784, grid=grid(896), stream=stream0)
        del convolution_73
        del convolution_78
        del primals_355
        del primals_361
        del primals_362
        buf145 = buf143[1]
        del buf143
        buf148 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf147, buf148, 896, grid=grid(896), stream=stream0)
        del buf147
        buf151 = buf149[1]
        del buf149
        buf153 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf152, buf153, 224, grid=grid(224), stream=stream0)
        buf156 = buf154[1]
        del buf154
        buf157 = reinterpret_tensor(buf152, (896, ), (1, ), 0); del buf152  # reuse
        buf158 = empty((896, ), device='cuda', dtype=torch.float32)
        buf159 = buf158; del buf158  # reuse
        # Source Nodes: [sigmoid_14], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf159, relu_58, buf144, convolution_77, buf155, convolution_75, primals_359, primals_360, buf157, 896, 784, grid=grid(896), stream=stream0)
        del buf144
        del convolution_75
        del convolution_77
        del primals_359
        del primals_360
        del relu_58
        buf163 = buf161[1]
        del buf161
        buf164 = empty((896, ), device='cuda', dtype=torch.float32)
        buf165 = empty((896, ), device='cuda', dtype=torch.float32)
        buf166 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf166, relu_57, buf162, convolution_74, primals_357, primals_358, buf164, 896, 784, grid=grid(896), stream=stream0)
        del convolution_74
        del primals_357
        del primals_358
        del relu_57
        buf170 = buf168[1]
        del buf168
        buf174 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_56, buf138, buf169, primals_356, primals_91, buf174, 702464, grid=grid(702464), stream=stream0)
        del primals_356
        del primals_91
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf175 = aten.convolution_backward(buf174, mul_148, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf174
        del mul_148
        del primals_226
        buf176 = buf175[0]
        buf177 = buf175[1]
        del buf175
        buf178 = reinterpret_tensor(buf155, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf155  # reuse
        buf179 = reinterpret_tensor(buf178, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf178  # reuse
        # Source Nodes: [sigmoid_13], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf179, buf176, relu_54, convolution_72, 3584, 196, grid=grid(3584), stream=stream0)
        buf180 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf179, buf180, 896, grid=grid(896), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf181 = aten.convolution_backward(buf179, relu_55, primals_224, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf179
        del primals_224
        buf182 = buf181[0]
        buf183 = buf181[1]
        del buf181
        buf184 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf184, relu_55, 896, grid=grid(896), stream=stream0)
        del relu_55
        buf185 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf184, buf185, 224, grid=grid(224), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf186 = aten.convolution_backward(buf184, mean_13, primals_222, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_13
        del primals_222
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf189 = reinterpret_tensor(buf184, (896, ), (1, ), 0); del buf184  # reuse
        buf190 = empty((896, ), device='cuda', dtype=torch.float32)
        buf191 = buf190; del buf190  # reuse
        # Source Nodes: [sigmoid_13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf191, relu_54, buf176, convolution_72, buf187, convolution_70, primals_353, primals_354, buf189, 896, 784, grid=grid(896), stream=stream0)
        del convolution_70
        del primals_353
        buf192 = buf176; del buf176  # reuse
        # Source Nodes: [sigmoid_13], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_16.run(buf192, relu_54, convolution_72, buf187, primals_354, primals_89, 702464, grid=grid(702464), stream=stream0)
        del convolution_72
        del primals_354
        del primals_89
        del relu_54
        # Source Nodes: [sigmoid_13], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf193 = aten.convolution_backward(buf192, relu_53, primals_221, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf192
        del primals_221
        buf194 = buf193[0]
        buf195 = buf193[1]
        del buf193
        buf196 = empty((896, ), device='cuda', dtype=torch.float32)
        buf197 = empty((896, ), device='cuda', dtype=torch.float32)
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf198, relu_53, buf194, convolution_69, primals_351, primals_352, buf196, 896, 784, grid=grid(896), stream=stream0)
        del convolution_69
        del primals_351
        buf199 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(buf199, relu_53, primals_352, primals_87, 702464, grid=grid(702464), stream=stream0)
        del primals_352
        del primals_87
        del relu_53
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf200 = aten.convolution_backward(buf199, relu_52, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_220
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = buf138; del buf138  # reuse
        buf207 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23.run(buf203, relu_52, relu_56, buf169, buf201, primals_350, primals_85, buf207, 702464, grid=grid(702464), stream=stream0)
        del buf169
        del buf201
        del primals_85
        del relu_52
        del relu_56
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf208 = aten.convolution_backward(buf207, mul_138, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_138
        del primals_219
        buf209 = buf208[0]
        buf211 = reinterpret_tensor(buf187, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf187  # reuse
        buf212 = reinterpret_tensor(buf211, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf211  # reuse
        # Source Nodes: [sigmoid_12], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf212, buf209, relu_50, convolution_67, 3584, 196, grid=grid(3584), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf214 = aten.convolution_backward(buf212, relu_51, primals_217, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_217
        buf215 = buf214[0]
        buf217 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf217, relu_51, 896, grid=grid(896), stream=stream0)
        del relu_51
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf219 = aten.convolution_backward(buf217, mean_12, primals_215, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_12
        del primals_215
        buf220 = buf219[0]
        buf225 = buf207; del buf207  # reuse
        # Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_20.run(relu_50, buf209, convolution_67, buf220, primals_348, primals_83, buf225, 702464, grid=grid(702464), stream=stream0)
        del primals_83
        # Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf226 = aten.convolution_backward(buf225, relu_49, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del primals_214
        buf227 = buf226[0]
        buf232 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_49, buf227, primals_346, primals_81, buf232, 702464, grid=grid(702464), stream=stream0)
        del primals_81
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf233 = aten.convolution_backward(buf232, relu_48, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf232
        del primals_213
        buf234 = buf233[0]
        buf204 = empty((896, ), device='cuda', dtype=torch.float32)
        buf205 = empty((896, ), device='cuda', dtype=torch.float32)
        buf236 = empty((896, ), device='cuda', dtype=torch.float32)
        buf237 = empty((896, ), device='cuda', dtype=torch.float32)
        buf206 = buf205; del buf205  # reuse
        buf238 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf206, buf238, buf203, convolution_68, primals_349, relu_48, buf234, convolution_63, primals_343, primals_350, primals_344, buf204, buf236, 896, 784, grid=grid(896), stream=stream0)
        del convolution_63
        del convolution_68
        del primals_343
        del primals_349
        del primals_350
        buf210 = buf208[1]
        del buf208
        buf213 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf212, buf213, 896, grid=grid(896), stream=stream0)
        del buf212
        buf216 = buf214[1]
        del buf214
        buf218 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf217, buf218, 224, grid=grid(224), stream=stream0)
        buf221 = buf219[1]
        del buf219
        buf222 = reinterpret_tensor(buf217, (896, ), (1, ), 0); del buf217  # reuse
        buf223 = empty((896, ), device='cuda', dtype=torch.float32)
        buf224 = buf223; del buf223  # reuse
        # Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf224, relu_50, buf209, convolution_67, buf220, convolution_65, primals_347, primals_348, buf222, 896, 784, grid=grid(896), stream=stream0)
        del buf209
        del convolution_65
        del convolution_67
        del primals_347
        del primals_348
        del relu_50
        buf228 = buf226[1]
        del buf226
        buf229 = empty((896, ), device='cuda', dtype=torch.float32)
        buf230 = empty((896, ), device='cuda', dtype=torch.float32)
        buf231 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf231, relu_49, buf227, convolution_64, primals_345, primals_346, buf229, 896, 784, grid=grid(896), stream=stream0)
        del convolution_64
        del primals_345
        del primals_346
        del relu_49
        buf235 = buf233[1]
        del buf233
        buf239 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_48, buf203, buf234, primals_344, primals_79, buf239, 702464, grid=grid(702464), stream=stream0)
        del primals_344
        del primals_79
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf240 = aten.convolution_backward(buf239, mul_128, primals_212, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf239
        del mul_128
        del primals_212
        buf241 = buf240[0]
        buf242 = buf240[1]
        del buf240
        buf243 = reinterpret_tensor(buf220, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf220  # reuse
        buf244 = reinterpret_tensor(buf243, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf243  # reuse
        # Source Nodes: [sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf244, buf241, relu_46, convolution_62, 3584, 196, grid=grid(3584), stream=stream0)
        buf245 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf244, buf245, 896, grid=grid(896), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf246 = aten.convolution_backward(buf244, relu_47, primals_210, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf244
        del primals_210
        buf247 = buf246[0]
        buf248 = buf246[1]
        del buf246
        buf249 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf249, relu_47, 896, grid=grid(896), stream=stream0)
        del relu_47
        buf250 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf249, buf250, 224, grid=grid(224), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf251 = aten.convolution_backward(buf249, mean_11, primals_208, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_11
        del primals_208
        buf252 = buf251[0]
        buf253 = buf251[1]
        del buf251
        buf254 = reinterpret_tensor(buf249, (896, ), (1, ), 0); del buf249  # reuse
        buf255 = empty((896, ), device='cuda', dtype=torch.float32)
        buf256 = buf255; del buf255  # reuse
        # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf256, relu_46, buf241, convolution_62, buf252, convolution_60, primals_341, primals_342, buf254, 896, 784, grid=grid(896), stream=stream0)
        del convolution_60
        del primals_341
        buf257 = buf241; del buf241  # reuse
        # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_16.run(buf257, relu_46, convolution_62, buf252, primals_342, primals_77, 702464, grid=grid(702464), stream=stream0)
        del convolution_62
        del primals_342
        del primals_77
        del relu_46
        # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf258 = aten.convolution_backward(buf257, relu_45, primals_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf257
        del primals_207
        buf259 = buf258[0]
        buf260 = buf258[1]
        del buf258
        buf261 = empty((896, ), device='cuda', dtype=torch.float32)
        buf262 = empty((896, ), device='cuda', dtype=torch.float32)
        buf263 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf263, relu_45, buf259, convolution_59, primals_339, primals_340, buf261, 896, 784, grid=grid(896), stream=stream0)
        del convolution_59
        del primals_339
        buf264 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(buf264, relu_45, primals_340, primals_75, 702464, grid=grid(702464), stream=stream0)
        del primals_340
        del primals_75
        del relu_45
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf265 = aten.convolution_backward(buf264, relu_44, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_206
        buf266 = buf265[0]
        buf267 = buf265[1]
        del buf265
        buf268 = buf203; del buf203  # reuse
        buf272 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23.run(buf268, relu_44, relu_48, buf234, buf266, primals_338, primals_73, buf272, 702464, grid=grid(702464), stream=stream0)
        del buf234
        del buf266
        del primals_73
        del relu_44
        del relu_48
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf273 = aten.convolution_backward(buf272, mul_118, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_118
        del primals_205
        buf274 = buf273[0]
        buf276 = reinterpret_tensor(buf252, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf252  # reuse
        buf277 = reinterpret_tensor(buf276, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf276  # reuse
        # Source Nodes: [sigmoid_10], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf277, buf274, relu_42, convolution_57, 3584, 196, grid=grid(3584), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf279 = aten.convolution_backward(buf277, relu_43, primals_203, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_203
        buf280 = buf279[0]
        buf282 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf282, relu_43, 896, grid=grid(896), stream=stream0)
        del relu_43
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf284 = aten.convolution_backward(buf282, mean_10, primals_201, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_10
        del primals_201
        buf285 = buf284[0]
        buf290 = buf272; del buf272  # reuse
        # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_20.run(relu_42, buf274, convolution_57, buf285, primals_336, primals_71, buf290, 702464, grid=grid(702464), stream=stream0)
        del primals_71
        # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf291 = aten.convolution_backward(buf290, relu_41, primals_200, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del primals_200
        buf292 = buf291[0]
        buf297 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_41, buf292, primals_334, primals_69, buf297, 702464, grid=grid(702464), stream=stream0)
        del primals_69
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf298 = aten.convolution_backward(buf297, relu_40, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf297
        del primals_199
        buf299 = buf298[0]
        buf269 = empty((896, ), device='cuda', dtype=torch.float32)
        buf270 = empty((896, ), device='cuda', dtype=torch.float32)
        buf301 = empty((896, ), device='cuda', dtype=torch.float32)
        buf302 = empty((896, ), device='cuda', dtype=torch.float32)
        buf271 = buf270; del buf270  # reuse
        buf303 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf271, buf303, buf268, convolution_58, primals_337, relu_40, buf299, convolution_53, primals_331, primals_338, primals_332, buf269, buf301, 896, 784, grid=grid(896), stream=stream0)
        del convolution_53
        del convolution_58
        del primals_331
        del primals_337
        del primals_338
        buf275 = buf273[1]
        del buf273
        buf278 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf277, buf278, 896, grid=grid(896), stream=stream0)
        del buf277
        buf281 = buf279[1]
        del buf279
        buf283 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf282, buf283, 224, grid=grid(224), stream=stream0)
        buf286 = buf284[1]
        del buf284
        buf287 = reinterpret_tensor(buf282, (896, ), (1, ), 0); del buf282  # reuse
        buf288 = empty((896, ), device='cuda', dtype=torch.float32)
        buf289 = buf288; del buf288  # reuse
        # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf289, relu_42, buf274, convolution_57, buf285, convolution_55, primals_335, primals_336, buf287, 896, 784, grid=grid(896), stream=stream0)
        del buf274
        del convolution_55
        del convolution_57
        del primals_335
        del primals_336
        del relu_42
        buf293 = buf291[1]
        del buf291
        buf294 = empty((896, ), device='cuda', dtype=torch.float32)
        buf295 = empty((896, ), device='cuda', dtype=torch.float32)
        buf296 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf296, relu_41, buf292, convolution_54, primals_333, primals_334, buf294, 896, 784, grid=grid(896), stream=stream0)
        del convolution_54
        del primals_333
        del primals_334
        del relu_41
        buf300 = buf298[1]
        del buf298
        buf304 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_40, buf268, buf299, primals_332, primals_67, buf304, 702464, grid=grid(702464), stream=stream0)
        del primals_332
        del primals_67
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf305 = aten.convolution_backward(buf304, mul_108, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf304
        del mul_108
        del primals_198
        buf306 = buf305[0]
        buf307 = buf305[1]
        del buf305
        buf308 = reinterpret_tensor(buf285, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf285  # reuse
        buf309 = reinterpret_tensor(buf308, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf308  # reuse
        # Source Nodes: [sigmoid_9], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf309, buf306, relu_38, convolution_52, 3584, 196, grid=grid(3584), stream=stream0)
        buf310 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf309, buf310, 896, grid=grid(896), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf311 = aten.convolution_backward(buf309, relu_39, primals_196, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf309
        del primals_196
        buf312 = buf311[0]
        buf313 = buf311[1]
        del buf311
        buf314 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf314, relu_39, 896, grid=grid(896), stream=stream0)
        del relu_39
        buf315 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf314, buf315, 224, grid=grid(224), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf316 = aten.convolution_backward(buf314, mean_9, primals_194, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_9
        del primals_194
        buf317 = buf316[0]
        buf318 = buf316[1]
        del buf316
        buf319 = reinterpret_tensor(buf314, (896, ), (1, ), 0); del buf314  # reuse
        buf320 = empty((896, ), device='cuda', dtype=torch.float32)
        buf321 = buf320; del buf320  # reuse
        # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf321, relu_38, buf306, convolution_52, buf317, convolution_50, primals_329, primals_330, buf319, 896, 784, grid=grid(896), stream=stream0)
        del convolution_50
        del primals_329
        buf322 = buf306; del buf306  # reuse
        # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_16.run(buf322, relu_38, convolution_52, buf317, primals_330, primals_65, 702464, grid=grid(702464), stream=stream0)
        del convolution_52
        del primals_330
        del primals_65
        del relu_38
        # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf323 = aten.convolution_backward(buf322, relu_37, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf322
        del primals_193
        buf324 = buf323[0]
        buf325 = buf323[1]
        del buf323
        buf326 = empty((896, ), device='cuda', dtype=torch.float32)
        buf327 = empty((896, ), device='cuda', dtype=torch.float32)
        buf328 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf328, relu_37, buf324, convolution_49, primals_327, primals_328, buf326, 896, 784, grid=grid(896), stream=stream0)
        del convolution_49
        del primals_327
        buf329 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(buf329, relu_37, primals_328, primals_63, 702464, grid=grid(702464), stream=stream0)
        del primals_328
        del primals_63
        del relu_37
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf330 = aten.convolution_backward(buf329, relu_36, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_192
        buf331 = buf330[0]
        buf332 = buf330[1]
        del buf330
        buf333 = buf268; del buf268  # reuse
        buf337 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23.run(buf333, relu_36, relu_40, buf299, buf331, primals_326, primals_61, buf337, 702464, grid=grid(702464), stream=stream0)
        del buf299
        del buf331
        del primals_61
        del relu_36
        del relu_40
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf338 = aten.convolution_backward(buf337, mul_98, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_98
        del primals_191
        buf339 = buf338[0]
        buf341 = reinterpret_tensor(buf317, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf317  # reuse
        buf342 = reinterpret_tensor(buf341, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf341  # reuse
        # Source Nodes: [sigmoid_8], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf342, buf339, relu_34, convolution_47, 3584, 196, grid=grid(3584), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf344 = aten.convolution_backward(buf342, relu_35, primals_189, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_189
        buf345 = buf344[0]
        buf347 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_5.run(buf347, relu_35, 896, grid=grid(896), stream=stream0)
        del relu_35
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf349 = aten.convolution_backward(buf347, mean_8, primals_187, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_8
        del primals_187
        buf350 = buf349[0]
        buf355 = buf337; del buf337  # reuse
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_20.run(relu_34, buf339, convolution_47, buf350, primals_324, primals_59, buf355, 702464, grid=grid(702464), stream=stream0)
        del primals_59
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf356 = aten.convolution_backward(buf355, relu_33, primals_186, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del primals_186
        buf357 = buf356[0]
        buf362 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_33, buf357, primals_322, primals_57, buf362, 702464, grid=grid(702464), stream=stream0)
        del primals_57
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf363 = aten.convolution_backward(buf362, relu_32, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf362
        del primals_185
        buf364 = buf363[0]
        buf334 = empty((896, ), device='cuda', dtype=torch.float32)
        buf335 = empty((896, ), device='cuda', dtype=torch.float32)
        buf366 = empty((896, ), device='cuda', dtype=torch.float32)
        buf367 = empty((896, ), device='cuda', dtype=torch.float32)
        buf373 = empty((896, ), device='cuda', dtype=torch.float32)
        buf336 = buf335; del buf335  # reuse
        buf368 = buf367; del buf367  # reuse
        buf374 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_24.run(buf336, buf368, buf374, buf333, convolution_48, primals_325, relu_32, buf364, convolution_43, primals_319, convolution_42, primals_317, primals_326, primals_320, primals_318, buf334, buf366, 896, 784, grid=grid(896), stream=stream0)
        del convolution_42
        del convolution_43
        del convolution_48
        del primals_317
        del primals_319
        del primals_325
        del primals_326
        buf340 = buf338[1]
        del buf338
        buf343 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf342, buf343, 896, grid=grid(896), stream=stream0)
        del buf342
        buf346 = buf344[1]
        del buf344
        buf348 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf347, buf348, 224, grid=grid(224), stream=stream0)
        buf351 = buf349[1]
        del buf349
        buf352 = reinterpret_tensor(buf347, (896, ), (1, ), 0); del buf347  # reuse
        buf353 = empty((896, ), device='cuda', dtype=torch.float32)
        buf354 = buf353; del buf353  # reuse
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf354, relu_34, buf339, convolution_47, buf350, convolution_45, primals_323, primals_324, buf352, 896, 784, grid=grid(896), stream=stream0)
        del convolution_45
        del convolution_47
        del primals_323
        del primals_324
        del relu_34
        buf358 = buf356[1]
        del buf356
        buf359 = empty((896, ), device='cuda', dtype=torch.float32)
        buf360 = empty((896, ), device='cuda', dtype=torch.float32)
        buf361 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf361, relu_33, buf357, convolution_44, primals_321, primals_322, buf359, 896, 784, grid=grid(896), stream=stream0)
        del convolution_44
        del primals_321
        del primals_322
        del relu_33
        buf365 = buf363[1]
        del buf363
        buf369 = buf357; del buf357  # reuse
        buf375 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(relu_32, buf333, buf364, primals_320, primals_55, primals_318, primals_53, buf369, buf375, 702464, grid=grid(702464), stream=stream0)
        del buf333
        del buf364
        del primals_318
        del primals_320
        del primals_53
        del primals_55
        del relu_32
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf370 = aten.convolution_backward(buf369, relu_28, primals_184, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf369
        del primals_184
        buf371 = buf370[0]
        buf372 = buf370[1]
        del buf370
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf376 = aten.convolution_backward(buf375, mul_85, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf375
        del mul_85
        del primals_183
        buf377 = buf376[0]
        buf378 = buf376[1]
        del buf376
        buf379 = reinterpret_tensor(buf350, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf350  # reuse
        buf380 = reinterpret_tensor(buf379, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf379  # reuse
        # Source Nodes: [sigmoid_7], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_13.run(buf380, buf377, relu_30, convolution_41, 3584, 196, grid=grid(3584), stream=stream0)
        buf381 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf380, buf381, 896, grid=grid(896), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf382 = aten.convolution_backward(buf380, relu_31, primals_181, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf380
        del primals_181
        buf383 = buf382[0]
        buf384 = buf382[1]
        del buf382
        buf385 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_26.run(buf385, relu_31, 448, grid=grid(448), stream=stream0)
        del relu_31
        buf386 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_27.run(buf385, buf386, 112, grid=grid(112), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf387 = aten.convolution_backward(buf385, mean_7, primals_179, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_7
        del primals_179
        buf388 = buf387[0]
        buf389 = buf387[1]
        del buf387
        buf390 = empty((896, ), device='cuda', dtype=torch.float32)
        buf391 = empty((896, ), device='cuda', dtype=torch.float32)
        buf392 = buf391; del buf391  # reuse
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_15.run(buf392, relu_30, buf377, convolution_41, buf388, convolution_39, primals_315, primals_316, buf390, 896, 784, grid=grid(896), stream=stream0)
        del convolution_39
        del primals_315
        buf393 = buf377; del buf377  # reuse
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_16.run(buf393, relu_30, convolution_41, buf388, primals_316, primals_51, 702464, grid=grid(702464), stream=stream0)
        del buf388
        del convolution_41
        del primals_316
        del primals_51
        del relu_30
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf394 = aten.convolution_backward(buf393, relu_29, primals_178, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf393
        del primals_178
        buf395 = buf394[0]
        buf396 = buf394[1]
        del buf394
        buf397 = empty((896, ), device='cuda', dtype=torch.float32)
        buf398 = empty((896, ), device='cuda', dtype=torch.float32)
        buf399 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf399, relu_29, buf395, convolution_38, primals_313, primals_314, buf397, 896, 3136, grid=grid(896), stream=stream0)
        del convolution_38
        del primals_313
        buf400 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf400, relu_29, primals_314, primals_49, 2809856, grid=grid(2809856), stream=stream0)
        del primals_314
        del primals_49
        del relu_29
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf401 = aten.convolution_backward(buf400, relu_28, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_177
        buf402 = buf401[0]
        buf403 = buf401[1]
        del buf401
        buf404 = reinterpret_tensor(buf385, (448, ), (1, ), 0); del buf385  # reuse
        buf405 = empty((448, ), device='cuda', dtype=torch.float32)
        buf406 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_30.run(buf406, relu_28, buf371, buf402, convolution_37, primals_311, primals_312, buf404, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_37
        del primals_311
        buf407 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_28, buf371, buf402, primals_312, primals_47, buf407, 1404928, grid=grid(1404928), stream=stream0)
        del primals_312
        del primals_47
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf408 = aten.convolution_backward(buf407, mul_75, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf407
        del mul_75
        del primals_176
        buf409 = buf408[0]
        buf410 = buf408[1]
        del buf408
        buf411 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf412 = reinterpret_tensor(buf411, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf411  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_32.run(buf412, buf409, relu_26, convolution_36, 1792, 784, grid=grid(1792), stream=stream0)
        buf413 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_33.run(buf412, buf413, 448, grid=grid(448), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf414 = aten.convolution_backward(buf412, relu_27, primals_174, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf412
        del primals_174
        buf415 = buf414[0]
        buf416 = buf414[1]
        del buf414
        buf417 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_26.run(buf417, relu_27, 448, grid=grid(448), stream=stream0)
        del relu_27
        buf418 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_27.run(buf417, buf418, 112, grid=grid(112), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf419 = aten.convolution_backward(buf417, mean_6, primals_172, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_6
        del primals_172
        buf420 = buf419[0]
        buf421 = buf419[1]
        del buf419
        buf422 = reinterpret_tensor(buf417, (448, ), (1, ), 0); del buf417  # reuse
        buf423 = empty((448, ), device='cuda', dtype=torch.float32)
        buf424 = buf423; del buf423  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34.run(buf424, relu_26, buf409, convolution_36, buf420, convolution_34, primals_309, primals_310, buf422, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_34
        del primals_309
        buf425 = buf409; del buf409  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_35.run(buf425, relu_26, convolution_36, buf420, primals_310, primals_45, 1404928, grid=grid(1404928), stream=stream0)
        del convolution_36
        del primals_310
        del primals_45
        del relu_26
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf426 = aten.convolution_backward(buf425, relu_25, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del buf425
        del primals_171
        buf427 = buf426[0]
        buf428 = buf426[1]
        del buf426
        buf429 = empty((448, ), device='cuda', dtype=torch.float32)
        buf430 = empty((448, ), device='cuda', dtype=torch.float32)
        buf431 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf431, relu_25, buf427, convolution_33, primals_307, primals_308, buf429, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_33
        del primals_307
        buf432 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37.run(buf432, relu_25, primals_308, primals_43, 1404928, grid=grid(1404928), stream=stream0)
        del primals_308
        del primals_43
        del relu_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf433 = aten.convolution_backward(buf432, relu_24, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_170
        buf434 = buf433[0]
        buf435 = buf433[1]
        del buf433
        buf436 = buf371; del buf371  # reuse
        buf440 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf436, relu_24, relu_28, buf402, buf434, primals_306, primals_41, buf440, 1404928, grid=grid(1404928), stream=stream0)
        del buf402
        del buf434
        del primals_41
        del relu_24
        del relu_28
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf441 = aten.convolution_backward(buf440, mul_65, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_65
        del primals_169
        buf442 = buf441[0]
        buf444 = reinterpret_tensor(buf420, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf420  # reuse
        buf445 = reinterpret_tensor(buf444, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf444  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_32.run(buf445, buf442, relu_22, convolution_31, 1792, 784, grid=grid(1792), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf447 = aten.convolution_backward(buf445, relu_23, primals_167, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_167
        buf448 = buf447[0]
        buf450 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_26.run(buf450, relu_23, 448, grid=grid(448), stream=stream0)
        del relu_23
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf452 = aten.convolution_backward(buf450, mean_5, primals_165, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_5
        del primals_165
        buf453 = buf452[0]
        buf458 = buf440; del buf440  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_39.run(relu_22, buf442, convolution_31, buf453, primals_304, primals_39, buf458, 1404928, grid=grid(1404928), stream=stream0)
        del primals_39
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf459 = aten.convolution_backward(buf458, relu_21, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del primals_164
        buf460 = buf459[0]
        buf465 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(relu_21, buf460, primals_302, primals_37, buf465, 1404928, grid=grid(1404928), stream=stream0)
        del primals_37
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf466 = aten.convolution_backward(buf465, relu_20, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf465
        del primals_163
        buf467 = buf466[0]
        buf437 = empty((448, ), device='cuda', dtype=torch.float32)
        buf438 = empty((448, ), device='cuda', dtype=torch.float32)
        buf469 = empty((448, ), device='cuda', dtype=torch.float32)
        buf470 = empty((448, ), device='cuda', dtype=torch.float32)
        buf439 = buf438; del buf438  # reuse
        buf471 = buf470; del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_41.run(buf439, buf471, buf436, convolution_32, primals_305, relu_20, buf467, convolution_27, primals_299, primals_306, primals_300, buf437, buf469, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_27
        del convolution_32
        del primals_299
        del primals_305
        del primals_306
        buf443 = buf441[1]
        del buf441
        buf446 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_33.run(buf445, buf446, 448, grid=grid(448), stream=stream0)
        del buf445
        buf449 = buf447[1]
        del buf447
        buf451 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_27.run(buf450, buf451, 112, grid=grid(112), stream=stream0)
        buf454 = buf452[1]
        del buf452
        buf455 = reinterpret_tensor(buf450, (448, ), (1, ), 0); del buf450  # reuse
        buf456 = empty((448, ), device='cuda', dtype=torch.float32)
        buf457 = buf456; del buf456  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34.run(buf457, relu_22, buf442, convolution_31, buf453, convolution_29, primals_303, primals_304, buf455, 448, 3136, grid=grid(448), stream=stream0)
        del buf442
        del convolution_29
        del convolution_31
        del primals_303
        del primals_304
        del relu_22
        buf461 = buf459[1]
        del buf459
        buf462 = empty((448, ), device='cuda', dtype=torch.float32)
        buf463 = empty((448, ), device='cuda', dtype=torch.float32)
        buf464 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf464, relu_21, buf460, convolution_28, primals_301, primals_302, buf462, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_28
        del primals_301
        del primals_302
        del relu_21
        buf468 = buf466[1]
        del buf466
        buf472 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_20, buf436, buf467, primals_300, primals_35, buf472, 1404928, grid=grid(1404928), stream=stream0)
        del primals_300
        del primals_35
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf473 = aten.convolution_backward(buf472, mul_55, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf472
        del mul_55
        del primals_162
        buf474 = buf473[0]
        buf475 = buf473[1]
        del buf473
        buf476 = reinterpret_tensor(buf453, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf453  # reuse
        buf477 = reinterpret_tensor(buf476, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf476  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_32.run(buf477, buf474, relu_18, convolution_26, 1792, 784, grid=grid(1792), stream=stream0)
        buf478 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_33.run(buf477, buf478, 448, grid=grid(448), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf479 = aten.convolution_backward(buf477, relu_19, primals_160, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf477
        del primals_160
        buf480 = buf479[0]
        buf481 = buf479[1]
        del buf479
        buf482 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_26.run(buf482, relu_19, 448, grid=grid(448), stream=stream0)
        del relu_19
        buf483 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_27.run(buf482, buf483, 112, grid=grid(112), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf484 = aten.convolution_backward(buf482, mean_4, primals_158, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_4
        del primals_158
        buf485 = buf484[0]
        buf486 = buf484[1]
        del buf484
        buf487 = reinterpret_tensor(buf482, (448, ), (1, ), 0); del buf482  # reuse
        buf488 = empty((448, ), device='cuda', dtype=torch.float32)
        buf489 = buf488; del buf488  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34.run(buf489, relu_18, buf474, convolution_26, buf485, convolution_24, primals_297, primals_298, buf487, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_24
        del primals_297
        buf490 = buf474; del buf474  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_35.run(buf490, relu_18, convolution_26, buf485, primals_298, primals_33, 1404928, grid=grid(1404928), stream=stream0)
        del convolution_26
        del primals_298
        del primals_33
        del relu_18
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf491 = aten.convolution_backward(buf490, relu_17, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del buf490
        del primals_157
        buf492 = buf491[0]
        buf493 = buf491[1]
        del buf491
        buf494 = empty((448, ), device='cuda', dtype=torch.float32)
        buf495 = empty((448, ), device='cuda', dtype=torch.float32)
        buf496 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf496, relu_17, buf492, convolution_23, primals_295, primals_296, buf494, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_23
        del primals_295
        buf497 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37.run(buf497, relu_17, primals_296, primals_31, 1404928, grid=grid(1404928), stream=stream0)
        del primals_296
        del primals_31
        del relu_17
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf498 = aten.convolution_backward(buf497, relu_16, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_156
        buf499 = buf498[0]
        buf500 = buf498[1]
        del buf498
        buf501 = buf436; del buf436  # reuse
        buf505 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf501, relu_16, relu_20, buf467, buf499, primals_294, primals_29, buf505, 1404928, grid=grid(1404928), stream=stream0)
        del buf467
        del buf499
        del primals_29
        del relu_16
        del relu_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf506 = aten.convolution_backward(buf505, mul_45, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_45
        del primals_155
        buf507 = buf506[0]
        buf509 = reinterpret_tensor(buf485, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf485  # reuse
        buf510 = reinterpret_tensor(buf509, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf509  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_32.run(buf510, buf507, relu_14, convolution_21, 1792, 784, grid=grid(1792), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf512 = aten.convolution_backward(buf510, relu_15, primals_153, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_153
        buf513 = buf512[0]
        buf515 = buf513; del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_26.run(buf515, relu_15, 448, grid=grid(448), stream=stream0)
        del relu_15
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf517 = aten.convolution_backward(buf515, mean_3, primals_151, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_3
        del primals_151
        buf518 = buf517[0]
        buf523 = buf505; del buf505  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_39.run(relu_14, buf507, convolution_21, buf518, primals_292, primals_27, buf523, 1404928, grid=grid(1404928), stream=stream0)
        del primals_27
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf524 = aten.convolution_backward(buf523, relu_13, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del primals_150
        buf525 = buf524[0]
        buf530 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(relu_13, buf525, primals_290, primals_25, buf530, 1404928, grid=grid(1404928), stream=stream0)
        del primals_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf531 = aten.convolution_backward(buf530, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf530
        del primals_149
        buf532 = buf531[0]
        buf502 = empty((448, ), device='cuda', dtype=torch.float32)
        buf503 = empty((448, ), device='cuda', dtype=torch.float32)
        buf534 = empty((448, ), device='cuda', dtype=torch.float32)
        buf535 = empty((448, ), device='cuda', dtype=torch.float32)
        buf541 = empty((448, ), device='cuda', dtype=torch.float32)
        buf504 = buf503; del buf503  # reuse
        buf536 = buf535; del buf535  # reuse
        buf542 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_42.run(buf504, buf536, buf542, buf501, convolution_22, primals_293, relu_12, buf532, convolution_17, primals_287, convolution_16, primals_285, primals_294, primals_288, primals_286, buf502, buf534, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_16
        del convolution_17
        del convolution_22
        del primals_285
        del primals_287
        del primals_293
        del primals_294
        buf508 = buf506[1]
        del buf506
        buf511 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_33.run(buf510, buf511, 448, grid=grid(448), stream=stream0)
        del buf510
        buf514 = buf512[1]
        del buf512
        buf516 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_27.run(buf515, buf516, 112, grid=grid(112), stream=stream0)
        buf519 = buf517[1]
        del buf517
        buf520 = reinterpret_tensor(buf515, (448, ), (1, ), 0); del buf515  # reuse
        buf521 = empty((448, ), device='cuda', dtype=torch.float32)
        buf522 = buf521; del buf521  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34.run(buf522, relu_14, buf507, convolution_21, buf518, convolution_19, primals_291, primals_292, buf520, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_19
        del convolution_21
        del primals_291
        del primals_292
        del relu_14
        buf526 = buf524[1]
        del buf524
        buf527 = empty((448, ), device='cuda', dtype=torch.float32)
        buf528 = empty((448, ), device='cuda', dtype=torch.float32)
        buf529 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf529, relu_13, buf525, convolution_18, primals_289, primals_290, buf527, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_18
        del primals_289
        del primals_290
        del relu_13
        buf533 = buf531[1]
        del buf531
        buf537 = buf525; del buf525  # reuse
        buf543 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(relu_12, buf501, buf532, primals_288, primals_23, primals_286, primals_21, buf537, buf543, 1404928, grid=grid(1404928), stream=stream0)
        del buf501
        del buf532
        del primals_21
        del primals_23
        del primals_286
        del primals_288
        del relu_12
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf538 = aten.convolution_backward(buf537, relu_8, primals_148, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf537
        del primals_148
        buf539 = buf538[0]
        buf540 = buf538[1]
        del buf538
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf544 = aten.convolution_backward(buf543, mul_32, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf543
        del mul_32
        del primals_147
        buf545 = buf544[0]
        buf546 = buf544[1]
        del buf544
        buf547 = reinterpret_tensor(buf518, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf518  # reuse
        buf548 = reinterpret_tensor(buf547, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf547  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_32.run(buf548, buf545, relu_10, convolution_15, 1792, 784, grid=grid(1792), stream=stream0)
        buf549 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_33.run(buf548, buf549, 448, grid=grid(448), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf550 = aten.convolution_backward(buf548, relu_11, primals_145, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf548
        del primals_145
        buf551 = buf550[0]
        buf552 = buf550[1]
        del buf550
        buf553 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_44.run(buf553, relu_11, 224, grid=grid(224), stream=stream0)
        del relu_11
        buf554 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_45.run(buf553, buf554, 56, grid=grid(56), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf555 = aten.convolution_backward(buf553, mean_2, primals_143, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_2
        del primals_143
        buf556 = buf555[0]
        buf557 = buf555[1]
        del buf555
        buf558 = empty((448, ), device='cuda', dtype=torch.float32)
        buf559 = empty((448, ), device='cuda', dtype=torch.float32)
        buf560 = buf559; del buf559  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34.run(buf560, relu_10, buf545, convolution_15, buf556, convolution_13, primals_283, primals_284, buf558, 448, 3136, grid=grid(448), stream=stream0)
        del convolution_13
        del primals_283
        buf561 = buf545; del buf545  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_35.run(buf561, relu_10, convolution_15, buf556, primals_284, primals_19, 1404928, grid=grid(1404928), stream=stream0)
        del buf556
        del convolution_15
        del primals_19
        del primals_284
        del relu_10
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf562 = aten.convolution_backward(buf561, relu_9, primals_142, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del buf561
        del primals_142
        buf563 = buf562[0]
        buf564 = buf562[1]
        del buf562
        buf565 = empty((448, ), device='cuda', dtype=torch.float32)
        buf566 = empty((448, ), device='cuda', dtype=torch.float32)
        buf567 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(buf567, relu_9, buf563, convolution_12, primals_281, primals_282, buf565, 448, 12544, grid=grid(448), stream=stream0)
        del convolution_12
        del primals_281
        buf568 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47.run(buf568, relu_9, primals_282, primals_17, 5619712, grid=grid(5619712), stream=stream0)
        del primals_17
        del primals_282
        del relu_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf569 = aten.convolution_backward(buf568, relu_8, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf568
        del primals_141
        buf570 = buf569[0]
        buf571 = buf569[1]
        del buf569
        buf572 = empty_strided((224, 2), (1, 224), device='cuda', dtype=torch.float32)
        buf574 = empty_strided((224, 2), (1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(relu_8, buf539, buf570, convolution_11, primals_279, buf572, buf574, 448, 6272, grid=grid(448), stream=stream0)
        del convolution_11
        del primals_279
        buf573 = reinterpret_tensor(buf553, (224, ), (1, ), 0); del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_49.run(buf572, buf573, 224, 2, grid=grid(224), stream=stream0)
        buf575 = empty((224, ), device='cuda', dtype=torch.float32)
        buf576 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf576, buf574, primals_280, 224, 2, grid=grid(224), stream=stream0)
        buf577 = reinterpret_tensor(buf400, (4, 224, 56, 56), (702464, 3136, 56, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51.run(relu_8, buf539, buf570, primals_280, primals_15, buf577, 2809856, grid=grid(2809856), stream=stream0)
        del primals_15
        del primals_280
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf578 = aten.convolution_backward(buf577, mul_22, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf577
        del mul_22
        del primals_140
        buf579 = buf578[0]
        buf580 = buf578[1]
        del buf578
        buf581 = empty_strided((4, 224, 1, 1), (224, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf582 = reinterpret_tensor(buf581, (4, 224, 1, 1), (224, 1, 1, 1), 0); del buf581  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_mul_sigmoid_sigmoid_backward_sum_52.run(buf582, buf579, relu_6, convolution_10, 896, 3136, grid=grid(896), stream=stream0)
        buf583 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf582, buf583, 224, grid=grid(224), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf584 = aten.convolution_backward(buf582, relu_7, primals_138, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf582
        del primals_138
        buf585 = buf584[0]
        buf586 = buf584[1]
        del buf584
        buf587 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_44.run(buf587, relu_7, 224, grid=grid(224), stream=stream0)
        del relu_7
        buf588 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_45.run(buf587, buf588, 56, grid=grid(56), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf589 = aten.convolution_backward(buf587, mean_1, primals_136, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_136
        buf590 = buf589[0]
        buf591 = buf589[1]
        del buf589
        buf592 = buf574; del buf574  # reuse
        buf594 = buf572; del buf572  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_53.run(relu_6, buf579, convolution_10, buf590, convolution_8, primals_277, buf592, buf594, 448, 6272, grid=grid(448), stream=stream0)
        del convolution_8
        del primals_277
        buf593 = reinterpret_tensor(buf587, (224, ), (1, ), 0); del buf587  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_49.run(buf592, buf593, 224, 2, grid=grid(224), stream=stream0)
        buf595 = empty((224, ), device='cuda', dtype=torch.float32)
        buf596 = buf595; del buf595  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf596, buf594, primals_278, 224, 2, grid=grid(224), stream=stream0)
        buf597 = buf579; del buf579  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_54.run(buf597, relu_6, convolution_10, buf590, primals_278, primals_13, 2809856, grid=grid(2809856), stream=stream0)
        del convolution_10
        del primals_13
        del primals_278
        del relu_6
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf598 = aten.convolution_backward(buf597, relu_5, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_135
        buf599 = buf598[0]
        buf600 = buf598[1]
        del buf598
        buf601 = buf594; del buf594  # reuse
        buf603 = buf592; del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_55.run(relu_5, buf599, convolution_7, primals_275, buf601, buf603, 448, 6272, grid=grid(448), stream=stream0)
        del convolution_7
        del primals_275
        buf602 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_49.run(buf601, buf602, 224, 2, grid=grid(224), stream=stream0)
        buf604 = empty((224, ), device='cuda', dtype=torch.float32)
        buf605 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf605, buf603, primals_276, 224, 2, grid=grid(224), stream=stream0)
        buf606 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf606, relu_5, primals_276, primals_11, 2809856, grid=grid(2809856), stream=stream0)
        del primals_11
        del primals_276
        del relu_5
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf607 = aten.convolution_backward(buf606, relu_4, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_134
        buf608 = buf607[0]
        buf609 = buf607[1]
        del buf607
        buf610 = buf539; del buf539  # reuse
        buf616 = buf606; del buf606  # reuse
        buf623 = buf597; del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_57.run(buf610, relu_4, relu_8, buf570, buf608, primals_274, primals_9, primals_272, primals_7, buf616, buf623, 2809856, grid=grid(2809856), stream=stream0)
        del buf570
        del buf608
        del primals_7
        del primals_9
        del relu_4
        del relu_8
        buf611 = buf603; del buf603  # reuse
        buf613 = buf601; del buf601  # reuse
        buf620 = empty_strided((224, 2), (1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf610, convolution_6, primals_273, convolution_5, primals_271, buf611, buf613, buf620, 448, 6272, grid=grid(448), stream=stream0)
        del buf610
        del convolution_5
        del convolution_6
        del primals_271
        del primals_273
        buf612 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_49.run(buf611, buf612, 224, 2, grid=grid(224), stream=stream0)
        del buf611
        buf614 = empty((224, ), device='cuda', dtype=torch.float32)
        buf615 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf615, buf613, primals_274, 224, 2, grid=grid(224), stream=stream0)
        del primals_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf617 = aten.convolution_backward(buf616, relu, primals_133, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf616
        del primals_133
        buf618 = buf617[0]
        buf619 = buf617[1]
        del buf617
        buf621 = empty((224, ), device='cuda', dtype=torch.float32)
        buf622 = buf621; del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf622, buf620, primals_272, 224, 2, grid=grid(224), stream=stream0)
        del primals_272
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf624 = aten.convolution_backward(buf623, mul_9, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf623
        del mul_9
        del primals_132
        buf625 = buf624[0]
        buf626 = buf624[1]
        del buf624
        buf627 = reinterpret_tensor(buf590, (4, 224, 1, 1), (224, 1, 896, 896), 0); del buf590  # reuse
        buf628 = reinterpret_tensor(buf627, (4, 224, 1, 1), (224, 1, 1, 1), 0); del buf627  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_mul_sigmoid_sigmoid_backward_sum_52.run(buf628, buf625, relu_2, convolution_4, 896, 3136, grid=grid(896), stream=stream0)
        buf629 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_6.run(buf628, buf629, 224, grid=grid(224), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf630 = aten.convolution_backward(buf628, relu_3, primals_130, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_130
        buf631 = buf630[0]
        buf632 = buf630[1]
        del buf630
        buf633 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_59.run(buf633, relu_3, 32, grid=grid(32), stream=stream0)
        del relu_3
        buf634 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_60.run(buf633, buf634, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf635 = aten.convolution_backward(buf633, mean, primals_128, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_128
        buf636 = buf635[0]
        buf637 = buf635[1]
        del buf635
        buf638 = buf620; del buf620  # reuse
        buf640 = buf613; del buf613  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_53.run(relu_2, buf625, convolution_4, buf636, convolution_2, primals_269, buf638, buf640, 448, 6272, grid=grid(448), stream=stream0)
        del convolution_2
        del primals_269
        buf639 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_49.run(buf638, buf639, 224, 2, grid=grid(224), stream=stream0)
        del buf638
        buf641 = empty((224, ), device='cuda', dtype=torch.float32)
        buf642 = buf641; del buf641  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf642, buf640, primals_270, 224, 2, grid=grid(224), stream=stream0)
        del buf640
        buf643 = buf625; del buf625  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_54.run(buf643, relu_2, convolution_4, buf636, primals_270, primals_5, 2809856, grid=grid(2809856), stream=stream0)
        del convolution_4
        del primals_270
        del primals_5
        del relu_2
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
        buf644 = aten.convolution_backward(buf643, relu_1, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf643
        del primals_127
        buf645 = buf644[0]
        buf646 = buf644[1]
        del buf644
        buf647 = reinterpret_tensor(buf636, (224, 4), (1, 224), 0); del buf636  # reuse
        buf649 = reinterpret_tensor(buf628, (224, 4), (1, 224), 0); del buf628  # reuse
        buf652 = empty((4, 224, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61.run(relu_1, buf645, convolution_1, primals_267, primals_268, primals_3, buf647, buf649, buf652, 896, 12544, grid=grid(896), stream=stream0)
        del buf645
        del convolution_1
        del primals_267
        del primals_3
        del relu_1
        buf648 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_62.run(buf647, buf648, 224, 4, grid=grid(224), stream=stream0)
        del buf647
        buf650 = empty((224, ), device='cuda', dtype=torch.float32)
        buf651 = buf650; del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_63.run(buf651, buf649, primals_268, 224, 4, grid=grid(224), stream=stream0)
        del buf649
        del primals_268
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf653 = aten.convolution_backward(buf652, relu, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf652
        del primals_126
        buf654 = buf653[0]
        buf655 = buf653[1]
        del buf653
        buf656 = empty((32, 7), device='cuda', dtype=torch.float32)
        buf658 = empty((32, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_64.run(relu, buf618, buf654, convolution, primals_265, buf656, buf658, 224, 7168, grid=grid(224), stream=stream0)
        del convolution
        del primals_265
        buf657 = reinterpret_tensor(buf633, (32, ), (1, ), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_65.run(buf656, buf657, 32, 7, grid=grid(32), stream=stream0)
        del buf656
        buf659 = empty((32, ), device='cuda', dtype=torch.float32)
        buf660 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_66.run(buf660, buf658, primals_266, 32, 7, grid=grid(32), stream=stream0)
        del buf658
        buf661 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf661, relu, buf654, primals_266, primals_1, 1605632, grid=grid(1605632), stream=stream0)
        del buf654
        del primals_1
        del primals_266
        del relu
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf662 = aten.convolution_backward(buf661, primals_389, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf661
        del primals_125
        del primals_389
        buf663 = buf662[1]
        return (buf660, buf657, buf651, buf648, buf642, buf639, buf622, buf612, buf615, buf612, buf605, buf602, buf596, buf593, buf576, buf573, buf567, buf565, buf560, buf558, buf542, buf534, buf536, buf534, buf529, buf527, buf522, buf520, buf504, buf502, buf496, buf494, buf489, buf487, buf471, buf469, buf464, buf462, buf457, buf455, buf439, buf437, buf431, buf429, buf424, buf422, buf406, buf404, buf399, buf397, buf392, buf390, buf374, buf366, buf368, buf366, buf361, buf359, buf354, buf352, buf336, buf334, buf328, buf326, buf321, buf319, buf303, buf301, buf296, buf294, buf289, buf287, buf271, buf269, buf263, buf261, buf256, buf254, buf238, buf236, buf231, buf229, buf224, buf222, buf206, buf204, buf198, buf196, buf191, buf189, buf173, buf171, buf166, buf164, buf159, buf157, buf141, buf139, buf133, buf131, buf126, buf124, buf108, buf106, buf101, buf99, buf94, buf92, buf76, buf74, buf68, buf66, buf61, buf59, buf43, buf41, buf36, buf34, buf29, buf27, buf11, buf3, buf5, buf3, buf663, buf655, buf646, buf637, buf634, buf632, buf629, buf626, buf619, buf609, buf600, buf591, buf588, buf586, buf583, buf580, buf571, buf564, buf557, buf554, buf552, buf549, buf546, buf540, buf533, buf526, buf519, buf516, buf514, buf511, buf508, buf500, buf493, buf486, buf483, buf481, buf478, buf475, buf468, buf461, buf454, buf451, buf449, buf446, buf443, buf435, buf428, buf421, buf418, buf416, buf413, buf410, buf403, buf396, buf389, buf386, buf384, buf381, buf378, buf372, buf365, buf358, buf351, buf348, buf346, buf343, buf340, buf332, buf325, buf318, buf315, buf313, buf310, buf307, buf300, buf293, buf286, buf283, buf281, buf278, buf275, buf267, buf260, buf253, buf250, buf248, buf245, buf242, buf235, buf228, buf221, buf218, buf216, buf213, buf210, buf202, buf195, buf188, buf185, buf183, buf180, buf177, buf170, buf163, buf156, buf153, buf151, buf148, buf145, buf137, buf130, buf123, buf120, buf118, buf115, buf112, buf105, buf98, buf91, buf88, buf86, buf83, buf80, buf72, buf65, buf58, buf55, buf53, buf50, buf47, buf40, buf33, buf26, buf23, buf21, buf18, buf15, buf9, reinterpret_tensor(buf1, (1000, 2240), (2240, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((224, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((8, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((224, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((224, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((56, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((224, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((56, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((448, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((112, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((896, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((2240, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((224, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((2240, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((2240, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((4, 224, 112, 112), (2809856, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((4, 224, 112, 112), (2809856, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((4, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_22 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 224, 56, 56), (702464, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((4, 448, 56, 56), (1404928, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((4, 448, 56, 56), (1404928, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_32 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((4, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((4, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((4, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((4, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((4, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((4, 448, 28, 28), (351232, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((4, 896, 28, 28), (702464, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((4, 896, 28, 28), (702464, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((4, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_98 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_108 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_118 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_48 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_49 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_12 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_138 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_52 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_53 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_54 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_13 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_148 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_56 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_57 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_58 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_14 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_59 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_158 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_60 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_61 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_62 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_15 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_63 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_168 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_64 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_66 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_16 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_67 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_178 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_68 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_69 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_90 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    mean_17 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_71 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((4, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_188 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_72 = rand_strided((4, 896, 14, 14), (175616, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_94 = rand_strided((4, 2240, 14, 14), (439040, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_73 = rand_strided((4, 2240, 14, 14), (439040, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((4, 2240, 7, 7), (109760, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_74 = rand_strided((4, 2240, 7, 7), (109760, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    mean_18 = rand_strided((4, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((4, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_97 = rand_strided((4, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_198 = rand_strided((4, 2240, 7, 7), (109760, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_98 = rand_strided((4, 2240, 7, 7), (109760, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_99 = rand_strided((4, 2240, 7, 7), (109760, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((4, 2240), (2240, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2240), (2240, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((4, 2240, 7, 7), (109760, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_198, primals_199, primals_200, primals_201, primals_203, primals_205, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_233, primals_234, primals_235, primals_236, primals_238, primals_240, primals_241, primals_242, primals_243, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_254, primals_255, primals_256, primals_257, primals_259, primals_261, primals_262, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, mean, relu_3, convolution_4, mul_9, convolution_5, convolution_6, relu_4, convolution_7, relu_5, convolution_8, relu_6, mean_1, relu_7, convolution_10, mul_22, convolution_11, relu_8, convolution_12, relu_9, convolution_13, relu_10, mean_2, relu_11, convolution_15, mul_32, convolution_16, convolution_17, relu_12, convolution_18, relu_13, convolution_19, relu_14, mean_3, relu_15, convolution_21, mul_45, convolution_22, relu_16, convolution_23, relu_17, convolution_24, relu_18, mean_4, relu_19, convolution_26, mul_55, convolution_27, relu_20, convolution_28, relu_21, convolution_29, relu_22, mean_5, relu_23, convolution_31, mul_65, convolution_32, relu_24, convolution_33, relu_25, convolution_34, relu_26, mean_6, relu_27, convolution_36, mul_75, convolution_37, relu_28, convolution_38, relu_29, convolution_39, relu_30, mean_7, relu_31, convolution_41, mul_85, convolution_42, convolution_43, relu_32, convolution_44, relu_33, convolution_45, relu_34, mean_8, relu_35, convolution_47, mul_98, convolution_48, relu_36, convolution_49, relu_37, convolution_50, relu_38, mean_9, relu_39, convolution_52, mul_108, convolution_53, relu_40, convolution_54, relu_41, convolution_55, relu_42, mean_10, relu_43, convolution_57, mul_118, convolution_58, relu_44, convolution_59, relu_45, convolution_60, relu_46, mean_11, relu_47, convolution_62, mul_128, convolution_63, relu_48, convolution_64, relu_49, convolution_65, relu_50, mean_12, relu_51, convolution_67, mul_138, convolution_68, relu_52, convolution_69, relu_53, convolution_70, relu_54, mean_13, relu_55, convolution_72, mul_148, convolution_73, relu_56, convolution_74, relu_57, convolution_75, relu_58, mean_14, relu_59, convolution_77, mul_158, convolution_78, relu_60, convolution_79, relu_61, convolution_80, relu_62, mean_15, relu_63, convolution_82, mul_168, convolution_83, relu_64, convolution_84, relu_65, convolution_85, relu_66, mean_16, relu_67, convolution_87, mul_178, convolution_88, relu_68, convolution_89, relu_69, convolution_90, relu_70, mean_17, relu_71, convolution_92, mul_188, convolution_93, relu_72, convolution_94, relu_73, convolution_95, relu_74, mean_18, relu_75, convolution_97, mul_198, convolution_98, convolution_99, clone, permute_1, le, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_regnet', benchmark_compiled_module)
