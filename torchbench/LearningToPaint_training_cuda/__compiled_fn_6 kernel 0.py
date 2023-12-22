
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


# kernel path: /tmp/torchinductor_youkaichao/wv/cwvlv6pvqzunh3isxii3ca6kpojkzus3gx4su427ntaiynk4tfh4.py
# Source Nodes: [], Original ATen: [aten.sigmoid_backward]

triton_poi_fused_sigmoid_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 260
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hiwkztjufjnfhhd66niexswxoibyaky5nhifbf5v4oxmxtgj75.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (65 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (130 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (195 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhdxvybttegrgihzt4tlfyihloxljuawkg3fkizo6mot7iucyks.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 8192)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 16
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp7
    tmp9 = tl.where(tmp8, tmp4, tmp1)
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozm5vdrdp2vfylowea5utq2ql5zydero4tlcmejkgtkxdaccver.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (16*x2) + (8192*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cev7woymeuz6njahpqhunotqcaxpaxecp43xg5xk4qx2tcutta4a.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(18, 19))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r2 = (rindex // 16)
    r1 = rindex % 16
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (r1 + (16*x0) + (8192*r2)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr6 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr8 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp41 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 16
    tmp5 = tl.full([1, 1], 0, tl.int32)
    tmp6 = tl.full([1, 1], 1, tl.int32)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp7
    tmp9 = tl.where(tmp8, tmp4, tmp1)
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tmp15 - tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp23 <= tmp1
    tmp26 = tmp10 + tmp25
    tmp27 = tl.where(tmp24, tmp1, tmp26)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp34 = tmp32 - tmp33
    tmp35 = tmp27 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp42 = tmp40 - tmp41
    tmp43 = tmp27 * tmp42
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK, RBLOCK])
    tmp46 = tl.where(rmask & xmask, tmp44, 0)
    tmp47 = tl.sum(tmp46, 1)[:, None]
    tmp49 = 1e-05
    tmp50 = tmp48 + tmp49
    tmp51 = tl.math.rsqrt(tmp50)
    tmp52 = tmp22 * tmp51
    tmp54 = tmp53 + tmp49
    tmp55 = tl.math.rsqrt(tmp54)
    tmp56 = tmp39 * tmp55
    tmp58 = tmp57 + tmp49
    tmp59 = tl.math.rsqrt(tmp58)
    tmp60 = tmp47 * tmp59
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp52, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp56, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp60, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7q76chxdiyeyjozdwwvdwpptujw4r4ljkn4inhtqknb67tp3xvj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 16
    r2 = (rindex // 16)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (16*x0) + (8192*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbqm5zq7v2omidne6k3w7qk5i35amybsiva6tdwdzuuiahd6nzx.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 16)
    y0 = yindex % 16
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0 + (16*x2) + (8192*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tmp5 / 16
    tmp7 = tl.full([1, 1], 0, tl.int32)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp9
    tmp11 = tl.where(tmp10, tmp6, tmp1)
    tmp12 = tl.where(tmp4, tmp1, tmp11)
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp22, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (512*y3)), tmp28, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zsus6ykklxjohur3mdvb3gv26zv22tuz44xskvwi3xyxahwto7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 64
    r2 = (rindex // 64)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (16384*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (64*x0) + (16384*r2)), rmask & xmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/52/c52imkjhrzdxskj55m3oiupubfobxn7axba4jwf6mkwlstc3ielw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp8 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (16384*(r2 // 64)) + (32768*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((64*x0) + (16384*(r2 // 64)) + (32768*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4cppuo6xyxktmftw7pk4toq27ezj22wurfrakkopwzxrlyxsz3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zf/czfzgts7gwyylyl6b4f4edzut6ccr53ytywz36f7cscyqncttgnk.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (16384*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (256*x2) + (16384*y1)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jbxv2d25gwt7hni4jqpm4fxjsmeswifvqwfqvn52koj5uzpbeu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (16384*(r2 // 64)) + (32768*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/s4/cs45gm6wqqklmhfdddy6gkbdkfu5k4tieczyacefidt5jsqjctqc.py
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
    size_hints=[256, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/ccoblpbboupkfnolkugmelaiba3xufxdueujocdrrsfi3v2kzlwi.py
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
    size_hints=[256, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask)
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (16384*y1)), xmask & ymask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqkifpsjdxr4rfsgvhyd2uvonqelsewg3c2k4yo5y7ilyuilmfa.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (16384*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (16384*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp11, xmask)
    tl.store(out_ptr0 + (y0 + (256*x2) + (16384*y1)), tmp18, xmask)
    tl.store(out_ptr1 + (y0 + (256*x2) + (16384*y1)), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqmtnmtjrvtf3iqbg34sksjto763hukoqr35li6fephv6jnfkcmw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (16384*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nq/cnqruhgyzkplxrqecyct264xgng477l3psapwltk4adzzugn63nd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (16384*(r2 // 64)) + (32768*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafjv72gpgohwh3r72mc526mb4eolp5w72pfpsh4jyomgprl3idp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 256
    r2 = (rindex // 256)
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cickbtlsmnau5n3g7wxbtoklrr4hje4hosydzktbol3tyfb6ddxv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (32768*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (32768*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chsejsgqs5goawr2ut4jgtalx4ggfosnjfxqvbpz2wiurwul5qfd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (8*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/cr/ccreuq22ua4x3fkqyxyqsrb3pb7plq6veqxpz63x3wvece4vj7nc.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (32768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (128*x2) + (32768*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ce/cce7tmn4h23myo4myeamunjcd3uvcixdbfvtspoqjusgspw7qbi5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (32768*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgksbnqz32vha6ty7ynuwfmnghav7hgvkz62pgkumwx2mi5xyzwo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (8*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhdngdil2etefhfnfhugrkpbwd4ayemg4wfjleqy5e7y7hwg4on.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (32768*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2g7yys6ynmgcgf3cnyu3pheu73oegi4ulvw3cb4aq6p5lf5joq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hudtbz76xm3b5jkkipwvfycf7vqcmslw3hogngdqsjj3t2ghdq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lx/clxzm4sf65ahrmzfquwri56u3x6uxlxzgugj65war52b22k5acsd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (32768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (128*x2) + (32768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp11, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (128*x2) + (32768*y1)), tmp18, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (128*x2) + (32768*y1)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wz/cwz35c4fqezszf32pmps645g7vtjjukzkkfhzgm7agtbix426kyp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 256
    r2 = (rindex // 256)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chyo4esjagdqdis3eeebpuemcugyap7fzybnkkfkiigovmvdbz4f.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (32768*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cvilkgmmm7bjlre32k6q2d6jlb3go4khpi5yxq35feoufdj7gkod.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cstx3qv2kyf7kejcnrqymg7jzhpsuc5ma3rb2jcypddidq56oh56.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (65536*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (65536*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rr/crrmyrlrxyzh3kajhsn3q3cqca2nqm4px753acxaroo3d5lnekcj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (32*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ot/cotkd6aqxaybkhutua5gotzv44bdpp2uchmfgtersj2y3jozzsz3.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (65536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (64*x2) + (65536*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnrlgfl2bykntbxg5joui5n2bi5t7c3mkmqlfegmqurdgeinwj5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (65536*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfedbwq4yjarec3ehvyo54527afpwu5y3y7pvz3iz7a6vv4g4js.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (32*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cge6b7k76is45ztd5mfqwbqnjtfemwdmvfrddetni5p56kmco2wo.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (65536*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kq/ckql4nbticw7r7h4vrquiq4mf5c6hpbrzatdhqbjzdvsgsbyq3ey.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 32
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/ou/couebeho45iqij3eydnq7thrx7kj5yeocbsmfv2im4yhiyqr7cbh.py
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
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nu/cnumthwheinf6lg5gdb2fwx7xtshwgiqce326yyhmyejyn2xviok.py
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
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (65536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (65536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp11, xmask & ymask)
    tl.store(out_ptr0 + (y0 + (64*x2) + (65536*y1)), tmp18, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (64*x2) + (65536*y1)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rv/crv5pktgll2enfirv3nw5mrv4moqrqe6zxb2a7sseaovk53ffn4r.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ji/cji7iihliwsfbqhjwezymhz3eckkghaqnykdkf2pvxfixndrg2k2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (65536*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvayhjjhuflmrqlcdcevyfpncqrhgqd6arlkdt4p65wtibxr7u3o.py
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
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (524288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bg/cbg32moxplc5qfcgzult6ujqjlklqirbyvon4reg3hwwqp72i2r5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zz/czzvh46se2tlhui4g22r37oftobn4im63dga6eoswapmfuysymhl.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5uzzfmnelznwhjovbz3mulfwdna73bzu7c4j6zpuhkptuxpga4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ru/crupfvi6xunkbj6kqqufhq2ngn2cq3nniawl6upknkhxot3w7z4l.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp13, ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, convolution, relu, convolution_1, relu_1, convolution_2, convolution_3, relu_2, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, convolution_11, relu_9, convolution_12, convolution_13, relu_10, convolution_14, relu_11, convolution_15, relu_12, convolution_16, relu_13, convolution_17, convolution_18, relu_14, convolution_19, relu_15, convolution_20, relu_16, view, sigmoid, permute_1, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 9, 3, 3), (81, 1, 27, 9))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_22, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_28, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_34, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_40, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_43, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_46, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_52, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_58, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_59, (512, ), (1, ))
    assert_size_stride(primals_61, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (64, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_129, (4, 9, 128, 128), (147456, 1, 1152, 9))
    assert_size_stride(convolution, (4, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(relu, (4, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_1, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(relu_1, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_2, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_3, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(relu_2, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_4, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(relu_3, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_5, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(relu_4, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_6, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(relu_5, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_7, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_8, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(relu_6, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_9, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(relu_7, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_10, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(relu_8, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_11, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(relu_9, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_12, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_13, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(relu_10, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_14, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(relu_11, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_15, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(relu_12, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_16, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(relu_13, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(convolution_17, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(convolution_18, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(relu_14, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(convolution_19, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(relu_15, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(convolution_20, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(relu_16, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(view, (4, 512), (512, 1))
    assert_size_stride(sigmoid, (4, 65), (65, 1))
    assert_size_stride(permute_1, (65, 512), (512, 1))
    assert_size_stride(tangents_1, (4, 65), (65, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 65), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sigmoid_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sigmoid_backward_0.run(tangents_1, sigmoid, buf0, 260, grid=grid(260), stream=stream0)
        del sigmoid
        del tangents_1
        buf1 = empty((4, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_1, out=buf1)
        del permute_1
        buf2 = empty((65, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (65, 4), (1, 65), 0), view, out=buf2)
        del view
        buf3 = empty((65, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_1.run(buf0, buf3, 65, grid=grid(65), stream=stream0)
        del buf0
        buf7 = empty_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_2.run(relu_16, buf1, primals_127, primals_62, buf7, 32768, grid=grid(32768), stream=stream0)
        del primals_62
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf8 = aten.convolution_backward(buf7, relu_15, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_61
        buf9 = buf8[0]
        buf14 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3.run(relu_15, buf9, primals_124, primals_59, buf14, 64, 512, grid=grid(64, 512), stream=stream0)
        del primals_59
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf15 = aten.convolution_backward(buf14, relu_14, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_58
        buf16 = buf15[0]
        buf4 = empty((512, ), device='cuda', dtype=torch.float32)
        buf5 = empty((512, ), device='cuda', dtype=torch.float32)
        buf18 = empty((512, ), device='cuda', dtype=torch.float32)
        buf19 = empty((512, ), device='cuda', dtype=torch.float32)
        buf25 = empty((512, ), device='cuda', dtype=torch.float32)
        buf6 = buf5; del buf5  # reuse
        buf20 = buf19; del buf19  # reuse
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_4.run(buf6, buf20, buf26, relu_16, buf1, convolution_20, primals_126, relu_14, buf16, convolution_18, primals_120, convolution_17, primals_117, primals_127, primals_121, primals_118, buf4, buf18, 512, 64, grid=grid(512), stream=stream0)
        del convolution_17
        del convolution_18
        del convolution_20
        del primals_117
        del primals_120
        del primals_126
        del primals_127
        buf10 = buf8[1]
        del buf8
        buf11 = empty((512, ), device='cuda', dtype=torch.float32)
        buf12 = empty((512, ), device='cuda', dtype=torch.float32)
        buf13 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_5.run(buf13, relu_15, buf9, convolution_19, primals_123, primals_124, buf11, 512, 64, grid=grid(512), stream=stream0)
        del convolution_19
        del primals_123
        del primals_124
        del relu_15
        buf17 = buf15[1]
        del buf15
        buf21 = reinterpret_tensor(buf9, (4, 512, 4, 4), (8192, 1, 2048, 512), 0); del buf9  # reuse
        buf27 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_6.run(relu_14, relu_16, buf1, buf16, primals_121, primals_56, primals_118, primals_53, buf21, buf27, 64, 512, grid=grid(64, 512), stream=stream0)
        del buf16
        del primals_118
        del primals_121
        del primals_53
        del primals_56
        del relu_14
        del relu_16
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf22 = aten.convolution_backward(buf21, relu_12, primals_55, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf21
        del primals_55
        buf23 = buf22[0]
        buf24 = buf22[1]
        del buf22
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf28 = aten.convolution_backward(buf27, relu_13, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_52
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = empty((512, ), device='cuda', dtype=torch.float32)
        buf32 = empty((512, ), device='cuda', dtype=torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_5.run(buf33, relu_13, buf29, convolution_16, primals_114, primals_115, buf31, 512, 64, grid=grid(512), stream=stream0)
        del convolution_16
        del primals_114
        buf34 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3.run(relu_13, buf29, primals_115, primals_50, buf34, 64, 512, grid=grid(64, 512), stream=stream0)
        del buf29
        del primals_115
        del primals_50
        del relu_13
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf35 = aten.convolution_backward(buf34, relu_12, primals_49, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf34
        del primals_49
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(relu_12, buf23, buf36, buf38, 256, 256, grid=grid(256), stream=stream0)
        buf39 = empty_strided((256, 2), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_8.run(relu_12, buf23, buf36, convolution_15, primals_111, buf39, 512, 128, grid=grid(512), stream=stream0)
        del convolution_15
        del primals_111
        buf40 = empty((256, ), device='cuda', dtype=torch.float32)
        buf41 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_9.run(buf41, buf39, primals_112, 256, 2, grid=grid(256), stream=stream0)
        buf42 = empty_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_12, buf23, buf36, primals_112, primals_47, buf42, 1024, 64, grid=grid(1024, 64), stream=stream0)
        del primals_112
        del primals_47
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf43 = aten.convolution_backward(buf42, relu_11, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_46
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        buf46 = buf39; del buf39  # reuse
        buf48 = empty_strided((256, 2), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_11.run(relu_11, buf44, convolution_14, primals_108, buf46, buf48, 512, 128, grid=grid(512), stream=stream0)
        del convolution_14
        del primals_108
        buf47 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_12.run(buf46, buf47, 256, 2, grid=grid(256), stream=stream0)
        buf49 = empty((256, ), device='cuda', dtype=torch.float32)
        buf50 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_9.run(buf50, buf48, primals_109, 256, 2, grid=grid(256), stream=stream0)
        buf51 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13.run(relu_11, buf44, primals_109, primals_44, buf51, 256, 256, grid=grid(256, 256), stream=stream0)
        del primals_109
        del primals_44
        del relu_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf52 = aten.convolution_backward(buf51, relu_10, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_43
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf55 = buf23; del buf23  # reuse
        buf60 = buf51; del buf51  # reuse
        buf67 = reinterpret_tensor(buf44, (4, 256, 8, 8), (16384, 1, 2048, 256), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf55, relu_10, relu_12, buf36, buf53, primals_106, primals_41, primals_103, primals_38, buf60, buf67, 1024, 64, grid=grid(1024, 64), stream=stream0)
        del buf36
        del buf53
        del primals_38
        del primals_41
        del relu_10
        del relu_12
        buf56 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_15.run(buf55, buf56, 256, 256, grid=grid(256), stream=stream0)
        buf57 = buf48; del buf48  # reuse
        buf64 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_16.run(buf55, convolution_13, primals_105, convolution_12, primals_102, buf57, buf64, 512, 128, grid=grid(512), stream=stream0)
        del buf55
        del convolution_12
        del convolution_13
        del primals_102
        del primals_105
        buf58 = empty((256, ), device='cuda', dtype=torch.float32)
        buf59 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_9.run(buf59, buf57, primals_106, 256, 2, grid=grid(256), stream=stream0)
        del primals_106
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf61 = aten.convolution_backward(buf60, relu_8, primals_40, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf60
        del primals_40
        buf62 = buf61[0]
        buf63 = buf61[1]
        del buf61
        buf65 = empty((256, ), device='cuda', dtype=torch.float32)
        buf66 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_9.run(buf66, buf64, primals_103, 256, 2, grid=grid(256), stream=stream0)
        del primals_103
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf68 = aten.convolution_backward(buf67, relu_9, primals_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_37
        buf69 = buf68[0]
        buf70 = buf68[1]
        del buf68
        buf71 = buf64; del buf64  # reuse
        buf73 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_11.run(relu_9, buf69, convolution_11, primals_99, buf71, buf73, 512, 128, grid=grid(512), stream=stream0)
        del convolution_11
        del primals_99
        buf72 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_12.run(buf71, buf72, 256, 2, grid=grid(256), stream=stream0)
        del buf71
        buf74 = empty((256, ), device='cuda', dtype=torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_9.run(buf75, buf73, primals_100, 256, 2, grid=grid(256), stream=stream0)
        del buf73
        buf76 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13.run(relu_9, buf69, primals_100, primals_35, buf76, 256, 256, grid=grid(256, 256), stream=stream0)
        del buf69
        del primals_100
        del primals_35
        del relu_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf77 = aten.convolution_backward(buf76, relu_8, primals_34, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf76
        del primals_34
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf80 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_17.run(relu_8, buf62, buf78, buf80, 128, 1024, grid=grid(128), stream=stream0)
        buf81 = empty((128, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(relu_8, buf62, buf78, convolution_10, primals_96, buf81, 1024, 128, grid=grid(1024), stream=stream0)
        del convolution_10
        del primals_96
        buf82 = empty((128, ), device='cuda', dtype=torch.float32)
        buf83 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf83, buf81, primals_97, 128, 8, grid=grid(128), stream=stream0)
        buf84 = empty_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20.run(relu_8, buf62, buf78, primals_97, primals_32, buf84, 512, 256, grid=grid(512, 256), stream=stream0)
        del primals_32
        del primals_97
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf85 = aten.convolution_backward(buf84, relu_7, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_31
        buf86 = buf85[0]
        buf87 = buf85[1]
        del buf85
        buf88 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_21.run(relu_7, buf86, buf88, 1024, 128, grid=grid(1024), stream=stream0)
        buf89 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_22.run(buf88, buf89, 128, 8, grid=grid(128), stream=stream0)
        buf90 = reinterpret_tensor(buf88, (128, 8), (1, 128), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(relu_7, buf86, convolution_9, primals_93, buf90, 1024, 128, grid=grid(1024), stream=stream0)
        del convolution_9
        del primals_93
        buf91 = empty((128, ), device='cuda', dtype=torch.float32)
        buf92 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_24.run(buf92, buf90, primals_94, 128, 8, grid=grid(128), stream=stream0)
        buf93 = buf84; del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(relu_7, buf86, primals_94, primals_29, buf93, 1024, 128, grid=grid(1024, 128), stream=stream0)
        del primals_29
        del primals_94
        del relu_7
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf94 = aten.convolution_backward(buf93, relu_6, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_28
        buf95 = buf94[0]
        buf96 = buf94[1]
        del buf94
        buf97 = buf62; del buf62  # reuse
        buf102 = buf93; del buf93  # reuse
        buf109 = reinterpret_tensor(buf86, (4, 128, 16, 16), (32768, 1, 2048, 128), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(buf97, relu_6, relu_8, buf78, buf95, primals_91, primals_26, primals_88, primals_23, buf102, buf109, 512, 256, grid=grid(512, 256), stream=stream0)
        del buf78
        del buf95
        del primals_23
        del primals_26
        del relu_6
        del relu_8
        buf98 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_27.run(buf97, buf98, 128, 1024, grid=grid(128), stream=stream0)
        buf99 = reinterpret_tensor(buf90, (128, 8), (8, 1), 0); del buf90  # reuse
        buf106 = empty((128, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_28.run(buf97, convolution_8, primals_90, convolution_7, primals_87, buf99, buf106, 1024, 128, grid=grid(1024), stream=stream0)
        del buf97
        del convolution_7
        del convolution_8
        del primals_87
        del primals_90
        buf100 = empty((128, ), device='cuda', dtype=torch.float32)
        buf101 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf101, buf99, primals_91, 128, 8, grid=grid(128), stream=stream0)
        del buf99
        del primals_91
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf103 = aten.convolution_backward(buf102, relu_4, primals_25, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf102
        del primals_25
        buf104 = buf103[0]
        buf105 = buf103[1]
        del buf103
        buf107 = empty((128, ), device='cuda', dtype=torch.float32)
        buf108 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf108, buf106, primals_88, 128, 8, grid=grid(128), stream=stream0)
        del primals_88
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf110 = aten.convolution_backward(buf109, relu_5, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        buf111 = buf110[0]
        buf112 = buf110[1]
        del buf110
        buf113 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_21.run(relu_5, buf111, buf113, 1024, 128, grid=grid(1024), stream=stream0)
        buf114 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_22.run(buf113, buf114, 128, 8, grid=grid(128), stream=stream0)
        buf115 = reinterpret_tensor(buf113, (128, 8), (1, 128), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(relu_5, buf111, convolution_6, primals_84, buf115, 1024, 128, grid=grid(1024), stream=stream0)
        del convolution_6
        del primals_84
        buf116 = empty((128, ), device='cuda', dtype=torch.float32)
        buf117 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_24.run(buf117, buf115, primals_85, 128, 8, grid=grid(128), stream=stream0)
        del buf115
        buf118 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25.run(relu_5, buf111, primals_85, primals_20, buf118, 1024, 128, grid=grid(1024, 128), stream=stream0)
        del buf111
        del primals_20
        del primals_85
        del relu_5
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf119 = aten.convolution_backward(buf118, relu_4, primals_19, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf118
        del primals_19
        buf120 = buf119[0]
        buf121 = buf119[1]
        del buf119
        buf122 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_29.run(relu_4, buf104, buf120, buf122, 64, 4096, grid=grid(64), stream=stream0)
        buf123 = reinterpret_tensor(buf1, (64, 32), (32, 1), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_30.run(relu_4, buf104, buf120, convolution_5, primals_81, buf123, 2048, 128, grid=grid(2048), stream=stream0)
        del convolution_5
        del primals_81
        buf124 = empty((64, ), device='cuda', dtype=torch.float32)
        buf125 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_31.run(buf125, buf123, primals_82, 64, 32, grid=grid(64), stream=stream0)
        buf126 = empty_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32.run(relu_4, buf104, buf120, primals_82, primals_17, buf126, 256, 1024, grid=grid(256, 1024), stream=stream0)
        del primals_17
        del primals_82
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf127 = aten.convolution_backward(buf126, relu_3, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_16
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_33.run(relu_3, buf128, buf130, 2048, 128, grid=grid(2048), stream=stream0)
        buf131 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_34.run(buf130, buf131, 64, 32, grid=grid(64), stream=stream0)
        buf132 = reinterpret_tensor(buf130, (64, 32), (1, 64), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_3, buf128, convolution_4, primals_78, buf132, 2048, 128, grid=grid(2048), stream=stream0)
        del convolution_4
        del primals_78
        buf133 = empty((64, ), device='cuda', dtype=torch.float32)
        buf134 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf134, buf132, primals_79, 64, 32, grid=grid(64), stream=stream0)
        buf135 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37.run(relu_3, buf128, primals_79, primals_14, buf135, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del primals_14
        del primals_79
        del relu_3
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf136 = aten.convolution_backward(buf135, relu_2, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_13
        buf137 = buf136[0]
        buf138 = buf136[1]
        del buf136
        buf139 = buf104; del buf104  # reuse
        buf144 = buf135; del buf135  # reuse
        buf151 = reinterpret_tensor(buf128, (4, 64, 32, 32), (65536, 1, 2048, 64), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf139, relu_2, relu_4, buf120, buf137, primals_76, primals_11, primals_73, primals_8, buf144, buf151, 256, 1024, grid=grid(256, 1024), stream=stream0)
        del buf120
        del buf137
        del primals_11
        del primals_8
        del relu_2
        del relu_4
        buf140 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_39.run(buf139, buf140, 64, 4096, grid=grid(64), stream=stream0)
        buf141 = reinterpret_tensor(buf132, (64, 32), (32, 1), 0); del buf132  # reuse
        buf148 = empty((64, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_40.run(buf139, convolution_3, primals_75, convolution_2, primals_72, buf141, buf148, 2048, 128, grid=grid(2048), stream=stream0)
        del buf139
        del convolution_2
        del convolution_3
        del primals_72
        del primals_75
        buf142 = empty((64, ), device='cuda', dtype=torch.float32)
        buf143 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_31.run(buf143, buf141, primals_76, 64, 32, grid=grid(64), stream=stream0)
        del buf141
        del primals_76
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf145 = aten.convolution_backward(buf144, relu, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf144
        del primals_10
        buf146 = buf145[0]
        buf147 = buf145[1]
        del buf145
        buf149 = empty((64, ), device='cuda', dtype=torch.float32)
        buf150 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_31.run(buf150, buf148, primals_73, 64, 32, grid=grid(64), stream=stream0)
        del primals_73
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf152 = aten.convolution_backward(buf151, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_7
        buf153 = buf152[0]
        buf154 = buf152[1]
        del buf152
        buf155 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_33.run(relu_1, buf153, buf155, 2048, 128, grid=grid(2048), stream=stream0)
        buf156 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_34.run(buf155, buf156, 64, 32, grid=grid(64), stream=stream0)
        buf157 = reinterpret_tensor(buf155, (64, 32), (1, 64), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_1, buf153, convolution_1, primals_69, buf157, 2048, 128, grid=grid(2048), stream=stream0)
        del convolution_1
        del primals_69
        buf158 = empty((64, ), device='cuda', dtype=torch.float32)
        buf159 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf159, buf157, primals_70, 64, 32, grid=grid(64), stream=stream0)
        del buf157
        buf160 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37.run(relu_1, buf153, primals_70, primals_5, buf160, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del buf153
        del primals_5
        del primals_70
        del relu_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf161 = aten.convolution_backward(buf160, relu, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf160
        del primals_4
        buf162 = buf161[0]
        buf163 = buf161[1]
        del buf161
        buf164 = empty_strided((64, 2), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_41.run(relu, buf146, buf162, buf164, 128, 8192, grid=grid(128), stream=stream0)
        buf165 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_42.run(buf164, buf165, 64, 2, grid=grid(64), stream=stream0)
        del buf164
        buf166 = empty((64, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_43.run(relu, buf146, buf162, convolution, primals_66, buf166, 8192, 128, grid=grid(8192), stream=stream0)
        del convolution
        del primals_66
        buf167 = empty((64, ), device='cuda', dtype=torch.float32)
        buf168 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_44.run(buf168, buf166, primals_67, 64, 128, grid=grid(64), stream=stream0)
        del buf166
        buf169 = empty_strided((4, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(relu, buf146, buf162, primals_67, primals_2, buf169, 256, 4096, grid=grid(256, 4096), stream=stream0)
        del buf146
        del buf162
        del primals_2
        del primals_67
        del relu
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf170 = aten.convolution_backward(buf169, primals_129, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf169
        del primals_1
        del primals_129
        buf171 = buf170[1]
        return (buf171, buf168, buf165, buf163, buf159, buf156, buf154, buf150, buf140, buf147, buf143, buf140, buf138, buf134, buf131, buf129, buf125, buf122, buf121, buf117, buf114, buf112, buf108, buf98, buf105, buf101, buf98, buf96, buf92, buf89, buf87, buf83, buf80, buf79, buf75, buf72, buf70, buf66, buf56, buf63, buf59, buf56, buf54, buf50, buf47, buf45, buf41, buf38, buf37, buf33, buf31, buf30, buf26, buf18, buf24, buf20, buf18, buf17, buf13, buf11, buf10, buf6, buf4, reinterpret_tensor(buf2, (65, 512), (512, 1), 0), buf3, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 9, 3, 3), (81, 1, 27, 9), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((4, 9, 128, 128), (147456, 1, 1152, 9), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float32)
    view = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    sigmoid = rand_strided((4, 65), (65, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((65, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 65), (65, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, convolution, relu, convolution_1, relu_1, convolution_2, convolution_3, relu_2, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, convolution_11, relu_9, convolution_12, convolution_13, relu_10, convolution_14, relu_11, convolution_15, relu_12, convolution_16, relu_13, convolution_17, convolution_18, relu_14, convolution_19, relu_15, convolution_20, relu_16, view, sigmoid, permute_1, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LearningToPaint', benchmark_compiled_module)
