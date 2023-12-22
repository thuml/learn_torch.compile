
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


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbtr5mx5ogt657xy56hkwej47wgbn2kppjian6xnngxt65sbgz4.py
# Source Nodes: [x_8, y], Original ATen: [aten.add, aten.native_layer_norm]
# x_8 => add
# y => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
triton_red_fused_add_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196) % 16
    x2 = (xindex // 3136)
    x4 = xindex % 3136
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x1 % 4)) + (56*(x0 // 14)) + (784*(x1 // 4)) + (3136*r3) + (401408*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp9 = tl.load(in_ptr0 + ((14*(x1 % 4)) + (56*(x0 // 14)) + (784*(x1 // 4)) + (3136*r3) + (401408*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r3 + (128*x5)), tmp24, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/mb/cmb3fmfwv5sptizhvtzqnn6wubrtwt3ynorgde4ddfp6b4xgmk5z.py
# Source Nodes: [x_10], Original ATen: [aten.clone, aten.mul]
# x_10 => clone_1, mul_2
triton_poi_fused_clone_mul_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 3136
    x2 = (xindex // 100352) % 4
    x3 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (384*x1) + (1204224*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhscfrjo7sgmyvxg2pbld3r7ur4xzafoobcsywqknvkpdnd2sow.py
# Source Nodes: [x_10], Original ATen: [aten.clone, aten.mul]
# x_10 => clone_2, mul_3
triton_poi_fused_clone_mul_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32) % 16
    y2 = (yindex // 512) % 4
    y3 = (yindex // 2048)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (32*y2) + (384*x4) + (75264*y1) + (1204224*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + y0 + (32*y2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4 + (196*y5)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmkenugsqxcdoyfeexxuz3sxbq4vwiwgbkdwh4do54e74m675rp.py
# Source Nodes: [x_10], Original ATen: [aten._softmax]
# x_10 => amax, div, exp, sub_1, sum_1
triton_per_fused__softmax_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp11, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/cre2ofuwdxyfueuiuoaxo3mfu5fgbtrrjq7ojqp7xdmuvqcuqtfo.py
# Source Nodes: [x_10], Original ATen: [aten.clone]
# x_10 => clone_3
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 3136
    x2 = (xindex // 100352) % 4
    x3 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (32*x2) + (384*x1) + (1204224*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jw/cjw35jjyintq7eg3yrwspe37pcrtmpz3chwukofy4f4ocrroafp3.py
# Source Nodes: [x_11], Original ATen: [aten.clone]
# x_11 => clone_4
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 802816
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 100352
    y1 = (yindex // 100352)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (100352*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (4*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qnspbihnq3yaeoakjpapcgydnlh24xrzduvcglmeernie7vrhz.py
# Source Nodes: [x_14, x_15, x_8], Original ATen: [aten.add, aten.native_layer_norm]
# x_14 => add_3
# x_15 => add_4, add_5, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
# x_8 => add
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196) % 16
    x2 = (xindex // 3136)
    x4 = xindex % 3136
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*(x1 % 4)) + (56*(x0 // 14)) + (784*(x1 // 4)) + (3136*r3) + (401408*x2) + (x0 % 14)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r3 + (128*x5)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r3 + (128*x5)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (128*x5)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkfnn5cl5d7xhyecp2mvyjyh5fk7v2zuy55wkfbn6icg7qz7hka.py
# Source Nodes: [x_17], Original ATen: [aten.gelu]
# x_17 => add_6, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2yru63re2tnjsaptvrmz32mrsubkaqqgqc7pjwuqw2572x2oie.py
# Source Nodes: [x_22, y_1], Original ATen: [aten.add, aten.native_layer_norm]
# x_22 => add_7
# y_1 => add_8, add_9, mul_10, mul_9, rsqrt_2, sub_3, var_mean_2
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciubaqqiq6okhmozssbk7bxc6buhysj4lu5nhoze2kbucwh4ckow.py
# Source Nodes: [x_22, x_28, x_29], Original ATen: [aten.add, aten.native_layer_norm]
# x_22 => add_7
# x_28 => add_10
# x_29 => add_11, add_12, mul_13, mul_14, rsqrt_3, sub_5, var_mean_3
triton_per_fused_add_native_layer_norm_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxbk437c62vgchy6jb27jn54rotpzzhpv2ecwu3r7gx7ky4hzvc.py
# Source Nodes: [x_41], Original ATen: [aten.convolution]
# x_41 => convolution_1
triton_poi_fused_convolution_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 56
    x2 = (xindex // 7168) % 56
    x3 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*(x1 % 14)) + (1792*(x2 % 14)) + (25088*(x1 // 14)) + (100352*(x2 // 14)) + (401408*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*(x1 % 14)) + (1792*(x2 % 14)) + (25088*(x1 // 14)) + (100352*(x2 // 14)) + (401408*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5jmqe4zcn7m63n5umzy2khqp2wvdkng4nd7cqsnydr3vvyhwmn.py
# Source Nodes: [x_41], Original ATen: [aten.convolution]
# x_41 => convolution_1
triton_poi_fused_convolution_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b4/cb4l7sjamszrocfqsuscfdwptl5ozfckdztgpkkh65ikwkvwbthw.py
# Source Nodes: [x_42], Original ATen: [aten.native_layer_norm]
# x_42 => clone_16, var_mean_4
triton_red_fused_native_layer_norm_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (802816*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr1 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddwqsweebe5wjmkluftyxh6j436kik5zjczk2rnk4udbcqclugo.py
# Source Nodes: [x_45], Original ATen: [aten.constant_pad_nd]
# x_45 => constant_pad_nd
triton_poi_fused_constant_pad_nd_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6653952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 57) % 57
    x0 = xindex % 57
    x4 = (xindex // 3249)
    x2 = (xindex // 3249) % 256
    x3 = (xindex // 831744)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 56, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (56*x1) + (3136*x4)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x0 + (56*x1) + (3136*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (x0 + (56*x1) + (3136*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp12 = 256.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp18 = tl.load(in_ptr4 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr5 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, float("-inf"), tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tl.store(out_ptr0 + (x5), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cs/ccszes6crky7braqbuuuzex5xartq3do3zuzogaz7c5bguq72pcz.py
# Source Nodes: [x_45, x_47], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_45 => constant_pad_nd
# x_47 => max_pool2d_with_indices
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (57 + (2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (58 + (2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (59 + (2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (114 + (2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (115 + (2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (116 + (2*x0) + (114*x1) + (3249*x2)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzihnh5fsuvob4kpunczet4vxkt3pp5hcz4xvl6we2ofpl6j7na.py
# Source Nodes: [x_52, y_2], Original ATen: [aten.add, aten.native_layer_norm]
# x_52 => add_17
# y_2 => var_mean_5
triton_red_fused_add_native_layer_norm_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 196
    x2 = (xindex // 392) % 4
    x3 = (xindex // 1568)
    x5 = xindex % 1568
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x2 % 2)) + (28*(x1 // 14)) + (392*(x2 // 2)) + (784*r4) + (100352*x0) + (200704*x3) + (x1 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x6), tmp4, xmask)
    tl.store(out_ptr1 + (x6), tmp5, xmask)
    tl.store(out_ptr2 + (x6), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2ao6uebdt52t7qam7uoabqjz2bd5twkb5rjtvgbicea6ryrxbz.py
# Source Nodes: [x_52, y_2], Original ATen: [aten.add, aten.native_layer_norm]
# x_52 => add_17
# y_2 => var_mean_5
triton_per_fused_add_native_layer_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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


# kernel path: /tmp/torchinductor_youkaichao/an/cantgxsawatshngiq5idd7bl3f4lztwc4s3trin5ih72og7ckeqr.py
# Source Nodes: [x_52, y_2], Original ATen: [aten.add, aten.native_layer_norm]
# x_52 => add_17
# y_2 => add_18, add_19, mul_20, mul_21, rsqrt_5, sub_7, var_mean_5
triton_poi_fused_add_native_layer_norm_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 4
    y2 = (yindex // 784)
    y4 = yindex % 784
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + ((14*(y1 % 2)) + (28*(y0 // 14)) + (392*(y1 // 2)) + (784*x3) + (200704*y2) + (y0 % 14)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (256*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y5), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y5), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3 + (256*y5)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3tilthownrslvg7v7azyfc7alca4u3vivqrwfw7ba7yqgflafu.py
# Source Nodes: [x_54], Original ATen: [aten.clone, aten.mul]
# x_54 => clone_18, mul_22
triton_poi_fused_clone_mul_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 784
    x2 = (xindex // 25088) % 8
    x3 = (xindex // 200704)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (768*x1) + (602112*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmdl7cthh4jyrwhihjnsebpgcts3c24enbj75bgbbnfutjpezt7.py
# Source Nodes: [x_54], Original ATen: [aten.clone, aten.mul]
# x_54 => clone_19, mul_23
triton_poi_fused_clone_mul_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32) % 4
    y2 = (yindex // 128) % 8
    y3 = (yindex // 1024)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (32*y2) + (768*x4) + (150528*y1) + (602112*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 + y0 + (32*y2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4 + (196*y5)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirjhegzg3yo6htekbocpi4nsuihoyxpasqcv6skmd2gaohzff6a.py
# Source Nodes: [x_54], Original ATen: [aten._softmax]
# x_54 => amax_2, div_2, exp_2, sub_8, sum_3
triton_per_fused__softmax_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp11, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4zllvzjdyy2izm5lq2kvkcz7isqjs6wlccddopdg3hka2fub6iu.py
# Source Nodes: [x_54], Original ATen: [aten.clone]
# x_54 => clone_20
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 784
    x2 = (xindex // 25088) % 8
    x3 = (xindex // 200704)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (32*x2) + (768*x1) + (602112*x3)), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clz6ritkxx6cyygurp3vi3ctg7xtvwwpq7prktyfp2yi2wl3ta2g.py
# Source Nodes: [x_55], Original ATen: [aten.clone]
# x_55 => clone_21
triton_poi_fused_clone_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 25088
    y1 = (yindex // 25088)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (25088*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (8*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjiia4qeil3ykv6obnymllv52rcxkueo3kcvai6alfx77jot66t.py
# Source Nodes: [x_52, x_58, x_59], Original ATen: [aten.add, aten.native_layer_norm]
# x_52 => add_17
# x_58 => add_20
# x_59 => var_mean_6
triton_red_fused_add_native_layer_norm_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 196
    x2 = (xindex // 392) % 4
    x3 = (xindex // 1568)
    x5 = xindex % 1568
    x6 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x2 % 2)) + (28*(x1 // 14)) + (392*(x2 // 2)) + (784*r4) + (100352*x0) + (200704*x3) + (x1 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r4 + (128*x6)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r4 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x6), tmp8, xmask)
    tl.store(out_ptr1 + (x6), tmp9, xmask)
    tl.store(out_ptr2 + (x6), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/ckleqa3ifyzg7pg67sezxhurtzq6ongsahmaerhkng2c4knapbak.py
# Source Nodes: [x_52, x_58, x_59], Original ATen: [aten.add, aten.native_layer_norm]
# x_52 => add_17
# x_58 => add_20
# x_59 => add_21, add_22, mul_24, mul_25, rsqrt_6, sub_9, var_mean_6
triton_poi_fused_add_native_layer_norm_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 4
    y2 = (yindex // 784)
    y4 = yindex % 784
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + ((14*(y1 % 2)) + (28*(y0 // 14)) + (392*(y1 // 2)) + (784*x3) + (200704*y2) + (y0 % 14)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (256*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (256*y5)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y5), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y5), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 256.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3 + (256*y5)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch2zm2uxoobalpypjbqgvj2lbbhm3kgdcvcmofbm36cp5p5u6bwz.py
# Source Nodes: [x_61], Original ATen: [aten.gelu]
# x_61 => add_23, erf_2, mul_26, mul_27, mul_28
triton_poi_fused_gelu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtpdksoa4xwdbpd65e2klekeptkruzlhi4jcanbe3lrz46fzo35.py
# Source Nodes: [x_52, x_58, x_66, y_3], Original ATen: [aten.add, aten.native_layer_norm]
# x_52 => add_17
# x_58 => add_20
# x_66 => add_24
# y_3 => add_25, add_26, mul_29, mul_30, rsqrt_7, sub_10, var_mean_7
triton_per_fused_add_native_layer_norm_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196) % 4
    x2 = (xindex // 784)
    x4 = xindex % 784
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*(x1 % 2)) + (28*(x0 // 14)) + (392*(x1 // 2)) + (784*r3) + (200704*x2) + (x0 % 14)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r3 + (256*x5)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r3 + (256*x5)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 256, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 256.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(in_out_ptr0 + (r3 + (256*x5)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (256*x5)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/cay25tdfkvkuq2usrz7hgssp7k5ax63pwamy3b4pfa6eh4snjfhh.py
# Source Nodes: [x_72, x_73], Original ATen: [aten.add, aten.native_layer_norm]
# x_72 => add_27
# x_73 => add_28, add_29, mul_33, mul_34, rsqrt_8, sub_12, var_mean_8
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 256, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 256.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lb4auhovtagw547z4a5pp6nkimtpyojgfhtjxdwdn5dt3dzr3g.py
# Source Nodes: [x_85], Original ATen: [aten.convolution]
# x_85 => convolution_2
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 28
    x2 = (xindex // 7168) % 28
    x3 = (xindex // 200704)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*(x1 % 14)) + (3584*(x2 % 14)) + (50176*(x1 // 14)) + (100352*(x2 // 14)) + (200704*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*(x1 % 14)) + (3584*(x2 % 14)) + (50176*(x1 // 14)) + (100352*(x2 // 14)) + (200704*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (256*(x1 % 14)) + (3584*(x2 % 14)) + (50176*(x1 // 14)) + (100352*(x2 // 14)) + (200704*x3)), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkitxhwutqh5cpz7x442c36yjpx7hwc7rzxfv4hqg2qoktumcgw.py
# Source Nodes: [x_85], Original ATen: [aten.convolution]
# x_85 => convolution_2
triton_poi_fused_convolution_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceatmen7vdm5m2g7fpezpwisnm66d7x6nbmnlbr5yi7jojfsssdo.py
# Source Nodes: [x_86], Original ATen: [aten.native_layer_norm]
# x_86 => clone_33, var_mean_9
triton_red_fused_native_layer_norm_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 4
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r3) + (100352*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybwkrpqvqokcjzfw74tisokc7dpcr3xx5mofx2ueedlfgcskrab.py
# Source Nodes: [x_86], Original ATen: [aten.native_layer_norm]
# x_86 => clone_33, var_mean_9
triton_per_fused_native_layer_norm_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (3136*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (784*r2) + (3136*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (784*r2) + (3136*x1)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/cio454kxhgjntp2iflxzemuop3oxcydhdclpm2opfcana6lzw37u.py
# Source Nodes: [x_89], Original ATen: [aten.constant_pad_nd]
# x_89 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3444736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 29) % 29
    x0 = xindex % 29
    x4 = (xindex // 841)
    x2 = (xindex // 841) % 512
    x3 = (xindex // 430592)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (28*x1) + (784*x4)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x0 + (28*x1) + (784*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (x0 + (28*x1) + (784*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp12 = 512.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp18 = tl.load(in_ptr4 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr5 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, float("-inf"), tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tl.store(out_ptr0 + (x5), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqn4tfhk6wwvyp5cen3zeuscfo77qa5rfxrtrja4vnrncru6zha6.py
# Source Nodes: [x_89, x_91], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# x_89 => constant_pad_nd_1
# x_91 => max_pool2d_with_indices_1
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + (2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (30 + (2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (31 + (2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (58 + (2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (59 + (2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (60 + (2*x0) + (58*x1) + (841*x2)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrbupdum4alvfhf4kqimwo4nqmdchxfwg7gspsa2tmjlxqb5kh3.py
# Source Nodes: [x_96, y_4], Original ATen: [aten.add, aten.native_layer_norm]
# x_96 => add_34
# y_4 => var_mean_10
triton_red_fused_add_native_layer_norm_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4) % 196
    x2 = (xindex // 784)
    x4 = xindex % 784
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (100352*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crkefxv7ugv5suzxucclfy3onyvbuukxdugd6xnjfpvy45gw4s4t.py
# Source Nodes: [x_96, y_4], Original ATen: [aten.add, aten.native_layer_norm]
# x_96 => add_34
# y_4 => var_mean_10
triton_per_fused_add_native_layer_norm_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (4*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gf/cgfnyv52sjh5bzidzijopi2cu2fiwwohmv7stc2and4yjb27joau.py
# Source Nodes: [x_96, y_4], Original ATen: [aten.add, aten.native_layer_norm]
# x_96 => add_34
# y_4 => add_35, add_36, mul_40, mul_41, rsqrt_10, sub_14, var_mean_10
triton_poi_fused_add_native_layer_norm_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvspygp42aycoyyotni6ex6afjslxifgbb3j6zqq5ppuszwd5gl4.py
# Source Nodes: [x_98], Original ATen: [aten.clone, aten.mul]
# x_98 => clone_34, mul_42
triton_poi_fused_clone_mul_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 16
    x3 = (xindex // 100352)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1536*x1) + (301056*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zp/czpcqqtui4xgwo373woh5lyogtvceqr4wtoktvjokauyiewpkryo.py
# Source Nodes: [x_98], Original ATen: [aten.clone, aten.mul]
# x_98 => clone_35, mul_43
triton_poi_fused_clone_mul_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5f/c5fohbo7udbrfbxzgr5h3wqm4kq5mzoh4h6l5m7whfu7utcplru7.py
# Source Nodes: [x_98], Original ATen: [aten._softmax]
# x_98 => amax_4, div_4, exp_4, sub_15, sum_5
triton_per_fused__softmax_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp11, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5m/c5mczahdcvk3uknd5a53cyxyu72rsqqn526riupkzglm6ldxtkuh.py
# Source Nodes: [x_98], Original ATen: [aten.clone]
# x_98 => clone_36
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 16
    x3 = (xindex // 100352)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (32*x2) + (1536*x1) + (301056*x3)), None)
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cuno57jm3cbwx7o5ncv62jsdocfczg5xvhi5blv7vibga2jm25qm.py
# Source Nodes: [x_99], Original ATen: [aten.clone]
# x_99 => clone_37
triton_poi_fused_clone_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 6272
    y1 = (yindex // 6272)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (6272*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckp7mlsuwmj4vlfceiefj7giyfwpheyzehioj2wnzqfslohoaekm.py
# Source Nodes: [x_102, x_103, x_96], Original ATen: [aten.add, aten.native_layer_norm]
# x_102 => add_37
# x_103 => var_mean_11
# x_96 => add_34
triton_red_fused_add_native_layer_norm_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4) % 196
    x2 = (xindex // 784)
    x4 = xindex % 784
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (100352*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp9, xmask)
    tl.store(out_ptr2 + (x5), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lks54d6tqjzruodwbuly3om4pkwuit6cnlqxpj3q24t7yo5rlm.py
# Source Nodes: [x_102, x_103, x_96], Original ATen: [aten.add, aten.native_layer_norm]
# x_102 => add_37
# x_103 => add_38, add_39, mul_44, mul_45, rsqrt_11, sub_16, var_mean_11
# x_96 => add_34
triton_poi_fused_add_native_layer_norm_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 512.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/67/c67n5pjo45bqk6f2k56hyucsg2qykkcdydmpplh5ktnyvbvxa2ys.py
# Source Nodes: [x_105], Original ATen: [aten.gelu]
# x_105 => add_40, erf_4, mul_46, mul_47, mul_48
triton_poi_fused_gelu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnkdz74rnudurlvrhh4eatwwyu7f2buwtu53djchvq3rdh3hrfe.py
# Source Nodes: [x_102, x_110, x_96, y_5], Original ATen: [aten.add, aten.native_layer_norm]
# x_102 => add_37
# x_110 => add_41
# x_96 => add_34
# y_5 => add_42, add_43, mul_49, mul_50, rsqrt_12, sub_17, var_mean_12
triton_per_fused_add_native_layer_norm_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_45', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (100352*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 512, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 512.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehjkty4mvvpnbm7alpw4cuxy5xfpxbhlanesnjvkhentw3czqy4.py
# Source Nodes: [x_116, x_117], Original ATen: [aten.add, aten.native_layer_norm]
# x_116 => add_44
# x_117 => add_45, add_46, mul_53, mul_54, rsqrt_13, sub_19, var_mean_13
triton_per_fused_add_native_layer_norm_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2cprxdyyqay4nqgd4sux5p3wjpvsjxgyz2wegv3i4kdiwakyyo.py
# Source Nodes: [x_116, x_124, y_6], Original ATen: [aten.add, aten.native_layer_norm]
# x_116 => add_44
# x_124 => add_48
# y_6 => add_49, add_50, mul_58, mul_59, rsqrt_14, sub_20, var_mean_14
triton_per_fused_add_native_layer_norm_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_47', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 512.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4wnmnvzomrqdbb4tbqjy3snq3rfs2mw6gwvp76n7ybsjpmf2vr.py
# Source Nodes: [x_368, x_377, x_382], Original ATen: [aten.add, aten.native_layer_norm]
# x_368 => add_170
# x_377 => add_174
# x_382 => var_mean_50
triton_per_fused_add_native_layer_norm_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_48', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckj7fxhuzcvc5ly5scp4fez6vl5sp22k5bxugau52dnd6elk5yod.py
# Source Nodes: [x_385], Original ATen: [aten.mean]
# x_385 => mean
triton_red_fused_mean_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 512.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-06
        tmp7 = tmp5 + tmp6
        tmp8 = tl.math.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gu/cgunm2axivwrms4bb6jadmfpeeo3655bhdboswxvfwrfojijvbdk.py
# Source Nodes: [x_385], Original ATen: [aten.mean]
# x_385 => mean
triton_per_fused_mean_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (1024*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(arg1_1, (128, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (1, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (1, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (384, 128), (128, 1))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (128, 128), (128, 1))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (512, 128), (128, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (128, 512), (512, 1))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (384, 128), (128, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (128, 128), (128, 1))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (512, 128), (128, 1))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (128, 512), (512, 1))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (768, 256), (256, 1))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (256, 256), (256, 1))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (1024, 256), (256, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (256, 1024), (1024, 1))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (768, 256), (256, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (256, 256), (256, 1))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (1024, 256), (256, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (256, 1024), (1024, 1))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (1536, 512), (512, 1))
    assert_size_stride(arg144_1, (1536, ), (1, ))
    assert_size_stride(arg145_1, (512, 512), (512, 1))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (2048, 512), (512, 1))
    assert_size_stride(arg148_1, (2048, ), (1, ))
    assert_size_stride(arg149_1, (512, 2048), (2048, 1))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (1536, 512), (512, 1))
    assert_size_stride(arg152_1, (1536, ), (1, ))
    assert_size_stride(arg153_1, (512, 512), (512, 1))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (2048, 512), (512, 1))
    assert_size_stride(arg156_1, (2048, ), (1, ))
    assert_size_stride(arg157_1, (512, 2048), (2048, 1))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (1536, 512), (512, 1))
    assert_size_stride(arg160_1, (1536, ), (1, ))
    assert_size_stride(arg161_1, (512, 512), (512, 1))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (2048, 512), (512, 1))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (512, 2048), (2048, 1))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (1536, 512), (512, 1))
    assert_size_stride(arg168_1, (1536, ), (1, ))
    assert_size_stride(arg169_1, (512, 512), (512, 1))
    assert_size_stride(arg170_1, (512, ), (1, ))
    assert_size_stride(arg171_1, (2048, 512), (512, 1))
    assert_size_stride(arg172_1, (2048, ), (1, ))
    assert_size_stride(arg173_1, (512, 2048), (2048, 1))
    assert_size_stride(arg174_1, (512, ), (1, ))
    assert_size_stride(arg175_1, (1536, 512), (512, 1))
    assert_size_stride(arg176_1, (1536, ), (1, ))
    assert_size_stride(arg177_1, (512, 512), (512, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (2048, 512), (512, 1))
    assert_size_stride(arg180_1, (2048, ), (1, ))
    assert_size_stride(arg181_1, (512, 2048), (2048, 1))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (1536, 512), (512, 1))
    assert_size_stride(arg184_1, (1536, ), (1, ))
    assert_size_stride(arg185_1, (512, 512), (512, 1))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (2048, 512), (512, 1))
    assert_size_stride(arg188_1, (2048, ), (1, ))
    assert_size_stride(arg189_1, (512, 2048), (2048, 1))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (1536, 512), (512, 1))
    assert_size_stride(arg192_1, (1536, ), (1, ))
    assert_size_stride(arg193_1, (512, 512), (512, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (2048, 512), (512, 1))
    assert_size_stride(arg196_1, (2048, ), (1, ))
    assert_size_stride(arg197_1, (512, 2048), (2048, 1))
    assert_size_stride(arg198_1, (512, ), (1, ))
    assert_size_stride(arg199_1, (1536, 512), (512, 1))
    assert_size_stride(arg200_1, (1536, ), (1, ))
    assert_size_stride(arg201_1, (512, 512), (512, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (2048, 512), (512, 1))
    assert_size_stride(arg204_1, (2048, ), (1, ))
    assert_size_stride(arg205_1, (512, 2048), (2048, 1))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (1536, 512), (512, 1))
    assert_size_stride(arg208_1, (1536, ), (1, ))
    assert_size_stride(arg209_1, (512, 512), (512, 1))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (2048, 512), (512, 1))
    assert_size_stride(arg212_1, (2048, ), (1, ))
    assert_size_stride(arg213_1, (512, 2048), (2048, 1))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (1536, 512), (512, 1))
    assert_size_stride(arg216_1, (1536, ), (1, ))
    assert_size_stride(arg217_1, (512, 512), (512, 1))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (2048, 512), (512, 1))
    assert_size_stride(arg220_1, (2048, ), (1, ))
    assert_size_stride(arg221_1, (512, 2048), (2048, 1))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (1536, 512), (512, 1))
    assert_size_stride(arg224_1, (1536, ), (1, ))
    assert_size_stride(arg225_1, (512, 512), (512, 1))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (2048, 512), (512, 1))
    assert_size_stride(arg228_1, (2048, ), (1, ))
    assert_size_stride(arg229_1, (512, 2048), (2048, 1))
    assert_size_stride(arg230_1, (512, ), (1, ))
    assert_size_stride(arg231_1, (1536, 512), (512, 1))
    assert_size_stride(arg232_1, (1536, ), (1, ))
    assert_size_stride(arg233_1, (512, 512), (512, 1))
    assert_size_stride(arg234_1, (512, ), (1, ))
    assert_size_stride(arg235_1, (2048, 512), (512, 1))
    assert_size_stride(arg236_1, (2048, ), (1, ))
    assert_size_stride(arg237_1, (512, 2048), (2048, 1))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (1536, 512), (512, 1))
    assert_size_stride(arg240_1, (1536, ), (1, ))
    assert_size_stride(arg241_1, (512, 512), (512, 1))
    assert_size_stride(arg242_1, (512, ), (1, ))
    assert_size_stride(arg243_1, (2048, 512), (512, 1))
    assert_size_stride(arg244_1, (2048, ), (1, ))
    assert_size_stride(arg245_1, (512, 2048), (2048, 1))
    assert_size_stride(arg246_1, (512, ), (1, ))
    assert_size_stride(arg247_1, (1536, 512), (512, 1))
    assert_size_stride(arg248_1, (1536, ), (1, ))
    assert_size_stride(arg249_1, (512, 512), (512, 1))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (2048, 512), (512, 1))
    assert_size_stride(arg252_1, (2048, ), (1, ))
    assert_size_stride(arg253_1, (512, 2048), (2048, 1))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (1536, 512), (512, 1))
    assert_size_stride(arg256_1, (1536, ), (1, ))
    assert_size_stride(arg257_1, (512, 512), (512, 1))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (2048, 512), (512, 1))
    assert_size_stride(arg260_1, (2048, ), (1, ))
    assert_size_stride(arg261_1, (512, 2048), (2048, 1))
    assert_size_stride(arg262_1, (512, ), (1, ))
    assert_size_stride(arg263_1, (1536, 512), (512, 1))
    assert_size_stride(arg264_1, (1536, ), (1, ))
    assert_size_stride(arg265_1, (512, 512), (512, 1))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (2048, 512), (512, 1))
    assert_size_stride(arg268_1, (2048, ), (1, ))
    assert_size_stride(arg269_1, (512, 2048), (2048, 1))
    assert_size_stride(arg270_1, (512, ), (1, ))
    assert_size_stride(arg271_1, (1536, 512), (512, 1))
    assert_size_stride(arg272_1, (1536, ), (1, ))
    assert_size_stride(arg273_1, (512, 512), (512, 1))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (2048, 512), (512, 1))
    assert_size_stride(arg276_1, (2048, ), (1, ))
    assert_size_stride(arg277_1, (512, 2048), (2048, 1))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (1536, 512), (512, 1))
    assert_size_stride(arg280_1, (1536, ), (1, ))
    assert_size_stride(arg281_1, (512, 512), (512, 1))
    assert_size_stride(arg282_1, (512, ), (1, ))
    assert_size_stride(arg283_1, (2048, 512), (512, 1))
    assert_size_stride(arg284_1, (2048, ), (1, ))
    assert_size_stride(arg285_1, (512, 2048), (2048, 1))
    assert_size_stride(arg286_1, (512, ), (1, ))
    assert_size_stride(arg287_1, (1536, 512), (512, 1))
    assert_size_stride(arg288_1, (1536, ), (1, ))
    assert_size_stride(arg289_1, (512, 512), (512, 1))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (2048, 512), (512, 1))
    assert_size_stride(arg292_1, (2048, ), (1, ))
    assert_size_stride(arg293_1, (512, 2048), (2048, 1))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (1536, 512), (512, 1))
    assert_size_stride(arg296_1, (1536, ), (1, ))
    assert_size_stride(arg297_1, (512, 512), (512, 1))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (2048, 512), (512, 1))
    assert_size_stride(arg300_1, (2048, ), (1, ))
    assert_size_stride(arg301_1, (512, 2048), (2048, 1))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (1000, 512), (512, 1))
    assert_size_stride(arg304_1, (1000, ), (1, ))
    assert_size_stride(arg305_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg305_1, arg105_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg105_1
        del arg305_1
        buf4 = empty((8, 16, 196, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8, y], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_native_layer_norm_0.run(buf0, arg106_1, arg0_1, arg1_1, arg2_1, buf4, 25088, 128, grid=grid(25088), stream=stream0)
        del arg1_1
        del arg2_1
        buf5 = empty((25088, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (25088, 128), (128, 1), 0), reinterpret_tensor(arg107_1, (128, 384), (1, 128), 0), out=buf5)
        del arg107_1
        buf6 = reinterpret_tensor(buf4, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf4  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_1.run(buf5, arg108_1, buf6, 3211264, grid=grid(3211264), stream=stream0)
        buf7 = empty((8, 4, 16, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_2.run(buf5, arg108_1, buf7, 16384, 196, grid=grid(16384, 196), stream=stream0)
        buf8 = empty((512, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf7, (512, 32, 196), (6272, 196, 1), 0), out=buf8)
        buf11 = empty((8, 4, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf8, buf11, 100352, 196, grid=grid(100352), stream=stream0)
        buf12 = reinterpret_tensor(buf7, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf7  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf5, arg108_1, buf12, 3211264, grid=grid(3211264), stream=stream0)
        del arg108_1
        buf13 = reinterpret_tensor(buf6, (512, 196, 32), (6272, 32, 1), 0); del buf6  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf12, (512, 196, 32), (6272, 32, 1), 0), out=buf13)
        buf14 = reinterpret_tensor(buf12, (8, 16, 196, 32, 4), (401408, 25088, 128, 4, 1), 0); del buf12  # reuse
        # Source Nodes: [x_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf13, buf14, 802816, 4, grid=grid(802816, 4), stream=stream0)
        buf15 = reinterpret_tensor(buf13, (25088, 128), (128, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf14, (25088, 128), (128, 1), 0), reinterpret_tensor(arg109_1, (128, 128), (1, 128), 0), out=buf15)
        del arg109_1
        buf16 = reinterpret_tensor(buf15, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf15  # reuse
        buf20 = reinterpret_tensor(buf14, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf14  # reuse
        # Source Nodes: [x_14, x_15, x_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf16, buf0, arg106_1, arg0_1, arg110_1, arg3_1, arg4_1, buf20, 25088, 128, grid=grid(25088), stream=stream0)
        del arg0_1
        del arg106_1
        del arg110_1
        del arg3_1
        del arg4_1
        buf21 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (25088, 128), (128, 1), 0), reinterpret_tensor(arg111_1, (128, 512), (1, 128), 0), out=buf21)
        del arg111_1
        buf22 = reinterpret_tensor(buf21, (8, 16, 196, 512), (1605632, 100352, 512, 1), 0); del buf21  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf22, arg112_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg112_1
        buf23 = reinterpret_tensor(buf20, (25088, 128), (128, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (25088, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 128), (1, 512), 0), out=buf23)
        del arg113_1
        buf27 = reinterpret_tensor(buf0, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf0  # reuse
        # Source Nodes: [x_22, y_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf16, buf23, arg114_1, arg5_1, arg6_1, buf27, 25088, 128, grid=grid(25088), stream=stream0)
        del arg5_1
        del arg6_1
        buf28 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (25088, 128), (128, 1), 0), reinterpret_tensor(arg115_1, (128, 384), (1, 128), 0), out=buf28)
        del arg115_1
        buf29 = reinterpret_tensor(buf27, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf27  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_1.run(buf28, arg116_1, buf29, 3211264, grid=grid(3211264), stream=stream0)
        buf30 = empty((8, 4, 16, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_2.run(buf28, arg116_1, buf30, 16384, 196, grid=grid(16384, 196), stream=stream0)
        buf31 = reinterpret_tensor(buf11, (512, 196, 196), (38416, 196, 1), 0); del buf11  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf30, (512, 32, 196), (6272, 196, 1), 0), out=buf31)
        buf34 = reinterpret_tensor(buf8, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), 0); del buf8  # reuse
        # Source Nodes: [x_24], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf31, buf34, 100352, 196, grid=grid(100352), stream=stream0)
        del buf31
        buf35 = reinterpret_tensor(buf30, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf30  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf28, arg116_1, buf35, 3211264, grid=grid(3211264), stream=stream0)
        del arg116_1
        del buf28
        buf36 = reinterpret_tensor(buf29, (512, 196, 32), (6272, 32, 1), 0); del buf29  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf34, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf35, (512, 196, 32), (6272, 32, 1), 0), out=buf36)
        del buf34
        buf37 = reinterpret_tensor(buf35, (8, 16, 196, 32, 4), (401408, 25088, 128, 4, 1), 0); del buf35  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf36, buf37, 802816, 4, grid=grid(802816, 4), stream=stream0)
        buf38 = reinterpret_tensor(buf36, (25088, 128), (128, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (25088, 128), (128, 1), 0), reinterpret_tensor(arg117_1, (128, 128), (1, 128), 0), out=buf38)
        del arg117_1
        buf39 = reinterpret_tensor(buf38, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf38  # reuse
        buf43 = reinterpret_tensor(buf37, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf37  # reuse
        # Source Nodes: [x_22, x_28, x_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf39, buf16, buf23, arg114_1, arg118_1, arg7_1, arg8_1, buf43, 25088, 128, grid=grid(25088), stream=stream0)
        del arg114_1
        del arg118_1
        del arg7_1
        del arg8_1
        del buf16
        buf44 = reinterpret_tensor(buf22, (25088, 512), (512, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf43, (25088, 128), (128, 1), 0), reinterpret_tensor(arg119_1, (128, 512), (1, 128), 0), out=buf44)
        del arg119_1
        buf45 = reinterpret_tensor(buf44, (8, 16, 196, 512), (1605632, 100352, 512, 1), 0); del buf44  # reuse
        # Source Nodes: [x_31], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf45, arg120_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg120_1
        buf46 = reinterpret_tensor(buf43, (25088, 128), (128, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (25088, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 128), (1, 512), 0), out=buf46)
        del arg121_1
        del buf45
        buf47 = reinterpret_tensor(buf23, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf23  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf39, buf46, arg122_1, buf47, 3211264, grid=grid(3211264), stream=stream0)
        del arg122_1
        del buf39
        buf48 = reinterpret_tensor(buf46, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf46  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf47, buf48, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del buf47
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg123_1
        del buf48
        buf50 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cuda', dtype=torch.float32)
        buf51 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_42], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_12.run(buf49, arg124_1, buf50, buf51, 25088, 256, grid=grid(25088), stream=stream0)
        buf53 = empty((8, 256, 57, 57), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_13.run(buf49, arg124_1, buf50, buf51, arg9_1, arg10_1, buf53, 6653952, grid=grid(6653952), stream=stream0)
        del arg10_1
        del arg124_1
        del arg9_1
        buf54 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_47], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14.run(buf53, buf54, 1605632, grid=grid(1605632), stream=stream0)
        del buf53
        buf55 = empty_strided((8, 4, 196, 1, 2), (1568, 392, 2, 12544, 1), device='cuda', dtype=torch.float32)
        buf56 = empty_strided((8, 4, 196, 1, 2), (1568, 392, 2, 12544, 1), device='cuda', dtype=torch.float32)
        buf57 = empty_strided((8, 4, 196, 1, 2), (1568, 392, 2, 12544, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52, y_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_15.run(buf54, arg11_1, buf55, buf56, buf57, 12544, 128, grid=grid(12544), stream=stream0)
        buf58 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52, y_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf55, buf56, buf57, buf58, buf59, 6272, 2, grid=grid(6272), stream=stream0)
        buf61 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52, y_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_17.run(buf54, arg11_1, buf58, buf59, arg12_1, arg13_1, buf61, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg12_1
        del arg13_1
        buf62 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (6272, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 768), (1, 256), 0), out=buf62)
        del arg125_1
        buf63 = reinterpret_tensor(buf61, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf61  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_18.run(buf62, arg126_1, buf63, 1605632, grid=grid(1605632), stream=stream0)
        buf64 = empty((8, 8, 4, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_19.run(buf62, arg126_1, buf64, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf65 = empty((256, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf63, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf64, (256, 32, 196), (6272, 196, 1), 0), out=buf65)
        buf68 = empty((8, 8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten._softmax]
        triton_per_fused__softmax_20.run(buf65, buf68, 50176, 196, grid=grid(50176), stream=stream0)
        buf69 = reinterpret_tensor(buf64, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf64  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf62, arg126_1, buf69, 1605632, grid=grid(1605632), stream=stream0)
        del arg126_1
        buf70 = reinterpret_tensor(buf63, (256, 196, 32), (6272, 32, 1), 0); del buf63  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf69, (256, 196, 32), (6272, 32, 1), 0), out=buf70)
        buf71 = reinterpret_tensor(buf69, (8, 4, 196, 32, 8), (200704, 50176, 256, 8, 1), 0); del buf69  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf70, buf71, 200704, 8, grid=grid(200704, 8), stream=stream0)
        buf72 = reinterpret_tensor(buf70, (6272, 256), (256, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (6272, 256), (256, 1), 0), reinterpret_tensor(arg127_1, (256, 256), (1, 256), 0), out=buf72)
        del arg127_1
        buf73 = buf57; del buf57  # reuse
        buf74 = buf56; del buf56  # reuse
        buf75 = buf55; del buf55  # reuse
        # Source Nodes: [x_52, x_58, x_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_23.run(buf54, arg11_1, buf72, arg128_1, buf73, buf74, buf75, 12544, 128, grid=grid(12544), stream=stream0)
        buf76 = buf59; del buf59  # reuse
        buf77 = buf58; del buf58  # reuse
        # Source Nodes: [x_52, x_58, x_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf73, buf74, buf75, buf76, buf77, 6272, 2, grid=grid(6272), stream=stream0)
        del buf73
        del buf74
        del buf75
        buf79 = reinterpret_tensor(buf71, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf71  # reuse
        # Source Nodes: [x_52, x_58, x_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_24.run(buf54, arg11_1, buf72, arg128_1, buf76, buf77, arg14_1, arg15_1, buf79, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg14_1
        del arg15_1
        buf80 = reinterpret_tensor(buf49, (6272, 1024), (1024, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (6272, 256), (256, 1), 0), reinterpret_tensor(arg129_1, (256, 1024), (1, 256), 0), out=buf80)
        del arg129_1
        buf81 = reinterpret_tensor(buf80, (8, 4, 196, 1024), (802816, 200704, 1024, 1), 0); del buf80  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf81, arg130_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg130_1
        buf82 = reinterpret_tensor(buf79, (6272, 256), (256, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg131_1, (1024, 256), (1, 1024), 0), out=buf82)
        del arg131_1
        buf83 = reinterpret_tensor(buf72, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf72  # reuse
        buf87 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52, x_58, x_66, y_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf83, buf54, arg11_1, arg128_1, buf82, arg132_1, arg16_1, arg17_1, buf87, 6272, 256, grid=grid(6272), stream=stream0)
        del arg11_1
        del arg128_1
        del arg132_1
        del arg16_1
        del arg17_1
        buf88 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (6272, 256), (256, 1), 0), reinterpret_tensor(arg133_1, (256, 768), (1, 256), 0), out=buf88)
        del arg133_1
        buf89 = reinterpret_tensor(buf87, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf87  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_18.run(buf88, arg134_1, buf89, 1605632, grid=grid(1605632), stream=stream0)
        buf90 = reinterpret_tensor(buf82, (8, 8, 4, 32, 196), (200704, 25088, 6272, 196, 1), 0); del buf82  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_19.run(buf88, arg134_1, buf90, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf91 = reinterpret_tensor(buf68, (256, 196, 196), (38416, 196, 1), 0); del buf68  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf90, (256, 32, 196), (6272, 196, 1), 0), out=buf91)
        buf94 = reinterpret_tensor(buf65, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), 0); del buf65  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._softmax]
        triton_per_fused__softmax_20.run(buf91, buf94, 50176, 196, grid=grid(50176), stream=stream0)
        del buf91
        buf95 = reinterpret_tensor(buf90, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf90  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf88, arg134_1, buf95, 1605632, grid=grid(1605632), stream=stream0)
        del arg134_1
        del buf88
        buf96 = reinterpret_tensor(buf89, (256, 196, 32), (6272, 32, 1), 0); del buf89  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf95, (256, 196, 32), (6272, 32, 1), 0), out=buf96)
        del buf94
        buf97 = reinterpret_tensor(buf95, (8, 4, 196, 32, 8), (200704, 50176, 256, 8, 1), 0); del buf95  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf96, buf97, 200704, 8, grid=grid(200704, 8), stream=stream0)
        buf98 = reinterpret_tensor(buf96, (6272, 256), (256, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (6272, 256), (256, 1), 0), reinterpret_tensor(arg135_1, (256, 256), (1, 256), 0), out=buf98)
        del arg135_1
        buf102 = reinterpret_tensor(buf97, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf97  # reuse
        # Source Nodes: [x_72, x_73], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf83, buf98, arg136_1, arg18_1, arg19_1, buf102, 6272, 256, grid=grid(6272), stream=stream0)
        del arg18_1
        del arg19_1
        buf103 = reinterpret_tensor(buf81, (6272, 1024), (1024, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (6272, 256), (256, 1), 0), reinterpret_tensor(arg137_1, (256, 1024), (1, 256), 0), out=buf103)
        del arg137_1
        buf104 = reinterpret_tensor(buf103, (8, 4, 196, 1024), (802816, 200704, 1024, 1), 0); del buf103  # reuse
        # Source Nodes: [x_75], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf104, arg138_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg138_1
        buf105 = reinterpret_tensor(buf102, (6272, 256), (256, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 256), (1, 1024), 0), out=buf105)
        del arg139_1
        del buf104
        buf106 = reinterpret_tensor(buf54, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf54  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf83, buf98, arg136_1, buf105, arg140_1, buf106, 1605632, grid=grid(1605632), stream=stream0)
        del arg136_1
        del arg140_1
        del buf105
        del buf83
        buf107 = reinterpret_tensor(buf98, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf98  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf106, buf107, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf106
        # Source Nodes: [x_85], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg141_1
        del buf107
        buf109 = reinterpret_tensor(buf51, (8, 28, 28, 1, 4), (3136, 28, 1, 25088, 784), 0); del buf51  # reuse
        buf110 = reinterpret_tensor(buf50, (8, 28, 28, 1, 4), (3136, 28, 1, 25088, 784), 0); del buf50  # reuse
        buf111 = empty_strided((8, 28, 28, 1, 4), (3136, 28, 1, 25088, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_30.run(buf108, arg142_1, buf109, buf110, buf111, 25088, 128, grid=grid(25088), stream=stream0)
        buf112 = reinterpret_tensor(buf77, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf77  # reuse
        buf113 = reinterpret_tensor(buf76, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf76  # reuse
        # Source Nodes: [x_86], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_31.run(buf109, buf110, buf111, buf112, buf113, 6272, 4, grid=grid(6272), stream=stream0)
        del buf109
        del buf110
        del buf111
        buf115 = empty((8, 512, 29, 29), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_32.run(buf108, arg142_1, buf112, buf113, arg20_1, arg21_1, buf115, 3444736, grid=grid(3444736), stream=stream0)
        del arg142_1
        del arg20_1
        del arg21_1
        buf116 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89, x_91], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_33.run(buf115, buf116, 802816, grid=grid(802816), stream=stream0)
        del buf115
        buf117 = reinterpret_tensor(buf113, (8, 1, 196, 1, 4), (784, 6272, 4, 6272, 1), 0); del buf113  # reuse
        buf118 = reinterpret_tensor(buf112, (8, 1, 196, 1, 4), (784, 6272, 4, 6272, 1), 0); del buf112  # reuse
        buf119 = empty_strided((8, 1, 196, 1, 4), (784, 6272, 4, 6272, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96, y_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_34.run(buf116, arg22_1, buf117, buf118, buf119, 6272, 128, grid=grid(6272), stream=stream0)
        buf120 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cuda', dtype=torch.float32)
        buf121 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96, y_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_35.run(buf117, buf118, buf119, buf120, buf121, 1568, 4, grid=grid(1568), stream=stream0)
        buf123 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96, y_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_36.run(buf116, arg22_1, buf120, buf121, arg23_1, arg24_1, buf123, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg23_1
        del arg24_1
        buf124 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (1568, 512), (512, 1), 0), reinterpret_tensor(arg143_1, (512, 1536), (1, 512), 0), out=buf124)
        del arg143_1
        buf125 = reinterpret_tensor(buf123, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf123  # reuse
        # Source Nodes: [x_98], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf124, arg144_1, buf125, 802816, grid=grid(802816), stream=stream0)
        buf126 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf124, arg144_1, buf126, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf127 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf126, (128, 32, 196), (6272, 196, 1), 0), out=buf127)
        buf130 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf127, buf130, 25088, 196, grid=grid(25088), stream=stream0)
        buf131 = reinterpret_tensor(buf126, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf126  # reuse
        # Source Nodes: [x_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf124, arg144_1, buf131, 802816, grid=grid(802816), stream=stream0)
        del arg144_1
        buf132 = reinterpret_tensor(buf125, (128, 196, 32), (6272, 32, 1), 0); del buf125  # reuse
        # Source Nodes: [x_98], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf131, (128, 196, 32), (6272, 32, 1), 0), out=buf132)
        buf133 = reinterpret_tensor(buf131, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf131  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf132, buf133, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf134 = reinterpret_tensor(buf132, (1568, 512), (512, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf133, (1568, 512), (512, 1), 0), reinterpret_tensor(arg145_1, (512, 512), (1, 512), 0), out=buf134)
        del arg145_1
        buf135 = buf119; del buf119  # reuse
        buf136 = buf118; del buf118  # reuse
        buf137 = buf117; del buf117  # reuse
        # Source Nodes: [x_102, x_103, x_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_42.run(buf116, arg22_1, buf134, arg146_1, buf135, buf136, buf137, 6272, 128, grid=grid(6272), stream=stream0)
        buf138 = buf121; del buf121  # reuse
        buf139 = buf120; del buf120  # reuse
        # Source Nodes: [x_102, x_103, x_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_35.run(buf135, buf136, buf137, buf138, buf139, 1568, 4, grid=grid(1568), stream=stream0)
        del buf135
        del buf136
        del buf137
        buf141 = reinterpret_tensor(buf133, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf133  # reuse
        # Source Nodes: [x_102, x_103, x_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_43.run(buf116, arg22_1, buf134, arg146_1, buf138, buf139, arg25_1, arg26_1, buf141, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg25_1
        del arg26_1
        buf142 = reinterpret_tensor(buf108, (1568, 2048), (2048, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (1568, 512), (512, 1), 0), reinterpret_tensor(arg147_1, (512, 2048), (1, 512), 0), out=buf142)
        del arg147_1
        buf143 = reinterpret_tensor(buf142, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf142  # reuse
        # Source Nodes: [x_105], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf143, arg148_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg148_1
        buf144 = reinterpret_tensor(buf141, (1568, 512), (512, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg149_1, (2048, 512), (1, 2048), 0), out=buf144)
        del arg149_1
        buf145 = reinterpret_tensor(buf134, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf134  # reuse
        buf149 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_110, x_96, y_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf145, buf116, arg22_1, arg146_1, buf144, arg150_1, arg27_1, arg28_1, buf149, 1568, 512, grid=grid(1568), stream=stream0)
        del arg146_1
        del arg150_1
        del arg22_1
        del arg27_1
        del arg28_1
        buf150 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (1568, 512), (512, 1), 0), reinterpret_tensor(arg151_1, (512, 1536), (1, 512), 0), out=buf150)
        del arg151_1
        buf151 = reinterpret_tensor(buf149, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf149  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf150, arg152_1, buf151, 802816, grid=grid(802816), stream=stream0)
        buf152 = reinterpret_tensor(buf144, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf144  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf150, arg152_1, buf152, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf153 = reinterpret_tensor(buf130, (128, 196, 196), (38416, 196, 1), 0); del buf130  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf152, (128, 32, 196), (6272, 196, 1), 0), out=buf153)
        buf156 = reinterpret_tensor(buf127, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf127  # reuse
        # Source Nodes: [x_112], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf153, buf156, 25088, 196, grid=grid(25088), stream=stream0)
        buf157 = reinterpret_tensor(buf152, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf152  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf150, arg152_1, buf157, 802816, grid=grid(802816), stream=stream0)
        del arg152_1
        buf158 = reinterpret_tensor(buf151, (128, 196, 32), (6272, 32, 1), 0); del buf151  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf157, (128, 196, 32), (6272, 32, 1), 0), out=buf158)
        buf159 = reinterpret_tensor(buf157, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf157  # reuse
        # Source Nodes: [x_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf158, buf159, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf160 = reinterpret_tensor(buf158, (1568, 512), (512, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (1568, 512), (512, 1), 0), reinterpret_tensor(arg153_1, (512, 512), (1, 512), 0), out=buf160)
        del arg153_1
        buf164 = reinterpret_tensor(buf159, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf159  # reuse
        # Source Nodes: [x_116, x_117], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf145, buf160, arg154_1, arg29_1, arg30_1, buf164, 1568, 512, grid=grid(1568), stream=stream0)
        del arg29_1
        del arg30_1
        buf165 = reinterpret_tensor(buf143, (1568, 2048), (2048, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (1568, 512), (512, 1), 0), reinterpret_tensor(arg155_1, (512, 2048), (1, 512), 0), out=buf165)
        del arg155_1
        buf166 = reinterpret_tensor(buf165, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf165  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf166, arg156_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg156_1
        buf167 = reinterpret_tensor(buf164, (1568, 512), (512, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg157_1, (2048, 512), (1, 2048), 0), out=buf167)
        del arg157_1
        buf168 = reinterpret_tensor(buf167, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf167  # reuse
        buf172 = reinterpret_tensor(buf116, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf116  # reuse
        # Source Nodes: [x_116, x_124, y_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf168, buf145, buf160, arg154_1, arg158_1, arg31_1, arg32_1, buf172, 1568, 512, grid=grid(1568), stream=stream0)
        del arg154_1
        del arg158_1
        del arg31_1
        del arg32_1
        buf173 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (1568, 512), (512, 1), 0), reinterpret_tensor(arg159_1, (512, 1536), (1, 512), 0), out=buf173)
        del arg159_1
        buf174 = reinterpret_tensor(buf172, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf172  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf173, arg160_1, buf174, 802816, grid=grid(802816), stream=stream0)
        buf175 = reinterpret_tensor(buf160, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf160  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf173, arg160_1, buf175, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf176 = reinterpret_tensor(buf156, (128, 196, 196), (38416, 196, 1), 0); del buf156  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf175, (128, 32, 196), (6272, 196, 1), 0), out=buf176)
        buf179 = reinterpret_tensor(buf153, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf153  # reuse
        # Source Nodes: [x_126], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf176, buf179, 25088, 196, grid=grid(25088), stream=stream0)
        buf180 = reinterpret_tensor(buf175, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf175  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf173, arg160_1, buf180, 802816, grid=grid(802816), stream=stream0)
        del arg160_1
        buf181 = reinterpret_tensor(buf174, (128, 196, 32), (6272, 32, 1), 0); del buf174  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf179, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf180, (128, 196, 32), (6272, 32, 1), 0), out=buf181)
        buf182 = reinterpret_tensor(buf180, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf180  # reuse
        # Source Nodes: [x_127], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf181, buf182, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (1568, 512), (512, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1568, 512), (512, 1), 0), reinterpret_tensor(arg161_1, (512, 512), (1, 512), 0), out=buf183)
        del arg161_1
        buf187 = reinterpret_tensor(buf182, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf182  # reuse
        # Source Nodes: [x_130, x_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf168, buf183, arg162_1, arg33_1, arg34_1, buf187, 1568, 512, grid=grid(1568), stream=stream0)
        del arg33_1
        del arg34_1
        buf188 = reinterpret_tensor(buf166, (1568, 2048), (2048, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1568, 512), (512, 1), 0), reinterpret_tensor(arg163_1, (512, 2048), (1, 512), 0), out=buf188)
        del arg163_1
        buf189 = reinterpret_tensor(buf188, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf188  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf189, arg164_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg164_1
        buf190 = reinterpret_tensor(buf187, (1568, 512), (512, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg165_1, (2048, 512), (1, 2048), 0), out=buf190)
        del arg165_1
        buf191 = reinterpret_tensor(buf190, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf190  # reuse
        buf195 = reinterpret_tensor(buf145, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf145  # reuse
        # Source Nodes: [x_130, x_138, y_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf191, buf168, buf183, arg162_1, arg166_1, arg35_1, arg36_1, buf195, 1568, 512, grid=grid(1568), stream=stream0)
        del arg162_1
        del arg166_1
        del arg35_1
        del arg36_1
        buf196 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (1568, 512), (512, 1), 0), reinterpret_tensor(arg167_1, (512, 1536), (1, 512), 0), out=buf196)
        del arg167_1
        buf197 = reinterpret_tensor(buf195, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf195  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf196, arg168_1, buf197, 802816, grid=grid(802816), stream=stream0)
        buf198 = reinterpret_tensor(buf183, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf183  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf196, arg168_1, buf198, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf199 = reinterpret_tensor(buf179, (128, 196, 196), (38416, 196, 1), 0); del buf179  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf197, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf198, (128, 32, 196), (6272, 196, 1), 0), out=buf199)
        buf202 = reinterpret_tensor(buf176, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf176  # reuse
        # Source Nodes: [x_140], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf199, buf202, 25088, 196, grid=grid(25088), stream=stream0)
        buf203 = reinterpret_tensor(buf198, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf198  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf196, arg168_1, buf203, 802816, grid=grid(802816), stream=stream0)
        del arg168_1
        buf204 = reinterpret_tensor(buf197, (128, 196, 32), (6272, 32, 1), 0); del buf197  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf202, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf203, (128, 196, 32), (6272, 32, 1), 0), out=buf204)
        buf205 = reinterpret_tensor(buf203, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf203  # reuse
        # Source Nodes: [x_141], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf204, buf205, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf206 = reinterpret_tensor(buf204, (1568, 512), (512, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (1568, 512), (512, 1), 0), reinterpret_tensor(arg169_1, (512, 512), (1, 512), 0), out=buf206)
        del arg169_1
        buf210 = reinterpret_tensor(buf205, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf205  # reuse
        # Source Nodes: [x_144, x_145], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf191, buf206, arg170_1, arg37_1, arg38_1, buf210, 1568, 512, grid=grid(1568), stream=stream0)
        del arg37_1
        del arg38_1
        buf211 = reinterpret_tensor(buf189, (1568, 2048), (2048, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (1568, 512), (512, 1), 0), reinterpret_tensor(arg171_1, (512, 2048), (1, 512), 0), out=buf211)
        del arg171_1
        buf212 = reinterpret_tensor(buf211, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf211  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf212, arg172_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg172_1
        buf213 = reinterpret_tensor(buf210, (1568, 512), (512, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg173_1, (2048, 512), (1, 2048), 0), out=buf213)
        del arg173_1
        buf214 = reinterpret_tensor(buf213, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf213  # reuse
        buf218 = reinterpret_tensor(buf168, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf168  # reuse
        # Source Nodes: [x_144, x_152, y_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf214, buf191, buf206, arg170_1, arg174_1, arg39_1, arg40_1, buf218, 1568, 512, grid=grid(1568), stream=stream0)
        del arg170_1
        del arg174_1
        del arg39_1
        del arg40_1
        buf219 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (1568, 512), (512, 1), 0), reinterpret_tensor(arg175_1, (512, 1536), (1, 512), 0), out=buf219)
        del arg175_1
        buf220 = reinterpret_tensor(buf218, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf218  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf219, arg176_1, buf220, 802816, grid=grid(802816), stream=stream0)
        buf221 = reinterpret_tensor(buf206, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf206  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf219, arg176_1, buf221, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf222 = reinterpret_tensor(buf202, (128, 196, 196), (38416, 196, 1), 0); del buf202  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf221, (128, 32, 196), (6272, 196, 1), 0), out=buf222)
        buf225 = reinterpret_tensor(buf199, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf199  # reuse
        # Source Nodes: [x_154], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf222, buf225, 25088, 196, grid=grid(25088), stream=stream0)
        buf226 = reinterpret_tensor(buf221, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf221  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf219, arg176_1, buf226, 802816, grid=grid(802816), stream=stream0)
        del arg176_1
        buf227 = reinterpret_tensor(buf220, (128, 196, 32), (6272, 32, 1), 0); del buf220  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf226, (128, 196, 32), (6272, 32, 1), 0), out=buf227)
        buf228 = reinterpret_tensor(buf226, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf226  # reuse
        # Source Nodes: [x_155], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf227, buf228, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf229 = reinterpret_tensor(buf227, (1568, 512), (512, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (1568, 512), (512, 1), 0), reinterpret_tensor(arg177_1, (512, 512), (1, 512), 0), out=buf229)
        del arg177_1
        buf233 = reinterpret_tensor(buf228, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf228  # reuse
        # Source Nodes: [x_158, x_159], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf214, buf229, arg178_1, arg41_1, arg42_1, buf233, 1568, 512, grid=grid(1568), stream=stream0)
        del arg41_1
        del arg42_1
        buf234 = reinterpret_tensor(buf212, (1568, 2048), (2048, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (1568, 512), (512, 1), 0), reinterpret_tensor(arg179_1, (512, 2048), (1, 512), 0), out=buf234)
        del arg179_1
        buf235 = reinterpret_tensor(buf234, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf234  # reuse
        # Source Nodes: [x_161], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf235, arg180_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg180_1
        buf236 = reinterpret_tensor(buf233, (1568, 512), (512, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg181_1, (2048, 512), (1, 2048), 0), out=buf236)
        del arg181_1
        buf237 = reinterpret_tensor(buf236, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf236  # reuse
        buf241 = reinterpret_tensor(buf191, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf191  # reuse
        # Source Nodes: [x_158, x_166, y_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf237, buf214, buf229, arg178_1, arg182_1, arg43_1, arg44_1, buf241, 1568, 512, grid=grid(1568), stream=stream0)
        del arg178_1
        del arg182_1
        del arg43_1
        del arg44_1
        buf242 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (1568, 512), (512, 1), 0), reinterpret_tensor(arg183_1, (512, 1536), (1, 512), 0), out=buf242)
        del arg183_1
        buf243 = reinterpret_tensor(buf241, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf241  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf242, arg184_1, buf243, 802816, grid=grid(802816), stream=stream0)
        buf244 = reinterpret_tensor(buf229, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf229  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf242, arg184_1, buf244, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf245 = reinterpret_tensor(buf225, (128, 196, 196), (38416, 196, 1), 0); del buf225  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf243, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf244, (128, 32, 196), (6272, 196, 1), 0), out=buf245)
        buf248 = reinterpret_tensor(buf222, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf222  # reuse
        # Source Nodes: [x_168], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf245, buf248, 25088, 196, grid=grid(25088), stream=stream0)
        buf249 = reinterpret_tensor(buf244, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf244  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf242, arg184_1, buf249, 802816, grid=grid(802816), stream=stream0)
        del arg184_1
        buf250 = reinterpret_tensor(buf243, (128, 196, 32), (6272, 32, 1), 0); del buf243  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf248, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf249, (128, 196, 32), (6272, 32, 1), 0), out=buf250)
        buf251 = reinterpret_tensor(buf249, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf249  # reuse
        # Source Nodes: [x_169], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf250, buf251, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf252 = reinterpret_tensor(buf250, (1568, 512), (512, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (1568, 512), (512, 1), 0), reinterpret_tensor(arg185_1, (512, 512), (1, 512), 0), out=buf252)
        del arg185_1
        buf256 = reinterpret_tensor(buf251, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf251  # reuse
        # Source Nodes: [x_172, x_173], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf237, buf252, arg186_1, arg45_1, arg46_1, buf256, 1568, 512, grid=grid(1568), stream=stream0)
        del arg45_1
        del arg46_1
        buf257 = reinterpret_tensor(buf235, (1568, 2048), (2048, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (1568, 512), (512, 1), 0), reinterpret_tensor(arg187_1, (512, 2048), (1, 512), 0), out=buf257)
        del arg187_1
        buf258 = reinterpret_tensor(buf257, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf257  # reuse
        # Source Nodes: [x_175], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf258, arg188_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg188_1
        buf259 = reinterpret_tensor(buf256, (1568, 512), (512, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg189_1, (2048, 512), (1, 2048), 0), out=buf259)
        del arg189_1
        buf260 = reinterpret_tensor(buf259, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf259  # reuse
        buf264 = reinterpret_tensor(buf214, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf214  # reuse
        # Source Nodes: [x_172, x_180, y_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf260, buf237, buf252, arg186_1, arg190_1, arg47_1, arg48_1, buf264, 1568, 512, grid=grid(1568), stream=stream0)
        del arg186_1
        del arg190_1
        del arg47_1
        del arg48_1
        buf265 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf264, (1568, 512), (512, 1), 0), reinterpret_tensor(arg191_1, (512, 1536), (1, 512), 0), out=buf265)
        del arg191_1
        buf266 = reinterpret_tensor(buf264, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf264  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf265, arg192_1, buf266, 802816, grid=grid(802816), stream=stream0)
        buf267 = reinterpret_tensor(buf252, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf252  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf265, arg192_1, buf267, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf268 = reinterpret_tensor(buf248, (128, 196, 196), (38416, 196, 1), 0); del buf248  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf266, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf267, (128, 32, 196), (6272, 196, 1), 0), out=buf268)
        buf271 = reinterpret_tensor(buf245, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf245  # reuse
        # Source Nodes: [x_182], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf268, buf271, 25088, 196, grid=grid(25088), stream=stream0)
        buf272 = reinterpret_tensor(buf267, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf267  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf265, arg192_1, buf272, 802816, grid=grid(802816), stream=stream0)
        del arg192_1
        buf273 = reinterpret_tensor(buf266, (128, 196, 32), (6272, 32, 1), 0); del buf266  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf271, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf272, (128, 196, 32), (6272, 32, 1), 0), out=buf273)
        buf274 = reinterpret_tensor(buf272, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf272  # reuse
        # Source Nodes: [x_183], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf273, buf274, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (1568, 512), (512, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (1568, 512), (512, 1), 0), reinterpret_tensor(arg193_1, (512, 512), (1, 512), 0), out=buf275)
        del arg193_1
        buf279 = reinterpret_tensor(buf274, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf274  # reuse
        # Source Nodes: [x_186, x_187], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf260, buf275, arg194_1, arg49_1, arg50_1, buf279, 1568, 512, grid=grid(1568), stream=stream0)
        del arg49_1
        del arg50_1
        buf280 = reinterpret_tensor(buf258, (1568, 2048), (2048, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (1568, 512), (512, 1), 0), reinterpret_tensor(arg195_1, (512, 2048), (1, 512), 0), out=buf280)
        del arg195_1
        buf281 = reinterpret_tensor(buf280, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf280  # reuse
        # Source Nodes: [x_189], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf281, arg196_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg196_1
        buf282 = reinterpret_tensor(buf279, (1568, 512), (512, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg197_1, (2048, 512), (1, 2048), 0), out=buf282)
        del arg197_1
        buf283 = reinterpret_tensor(buf282, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf282  # reuse
        buf287 = reinterpret_tensor(buf237, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf237  # reuse
        # Source Nodes: [x_186, x_194, y_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf283, buf260, buf275, arg194_1, arg198_1, arg51_1, arg52_1, buf287, 1568, 512, grid=grid(1568), stream=stream0)
        del arg194_1
        del arg198_1
        del arg51_1
        del arg52_1
        buf288 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (1568, 512), (512, 1), 0), reinterpret_tensor(arg199_1, (512, 1536), (1, 512), 0), out=buf288)
        del arg199_1
        buf289 = reinterpret_tensor(buf287, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf287  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf288, arg200_1, buf289, 802816, grid=grid(802816), stream=stream0)
        buf290 = reinterpret_tensor(buf275, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf275  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf288, arg200_1, buf290, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf291 = reinterpret_tensor(buf271, (128, 196, 196), (38416, 196, 1), 0); del buf271  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf290, (128, 32, 196), (6272, 196, 1), 0), out=buf291)
        buf294 = reinterpret_tensor(buf268, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf268  # reuse
        # Source Nodes: [x_196], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf291, buf294, 25088, 196, grid=grid(25088), stream=stream0)
        buf295 = reinterpret_tensor(buf290, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf290  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf288, arg200_1, buf295, 802816, grid=grid(802816), stream=stream0)
        del arg200_1
        buf296 = reinterpret_tensor(buf289, (128, 196, 32), (6272, 32, 1), 0); del buf289  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf295, (128, 196, 32), (6272, 32, 1), 0), out=buf296)
        buf297 = reinterpret_tensor(buf295, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf295  # reuse
        # Source Nodes: [x_197], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf296, buf297, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf298 = reinterpret_tensor(buf296, (1568, 512), (512, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf297, (1568, 512), (512, 1), 0), reinterpret_tensor(arg201_1, (512, 512), (1, 512), 0), out=buf298)
        del arg201_1
        buf302 = reinterpret_tensor(buf297, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf297  # reuse
        # Source Nodes: [x_200, x_201], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf283, buf298, arg202_1, arg53_1, arg54_1, buf302, 1568, 512, grid=grid(1568), stream=stream0)
        del arg53_1
        del arg54_1
        buf303 = reinterpret_tensor(buf281, (1568, 2048), (2048, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (1568, 512), (512, 1), 0), reinterpret_tensor(arg203_1, (512, 2048), (1, 512), 0), out=buf303)
        del arg203_1
        buf304 = reinterpret_tensor(buf303, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf303  # reuse
        # Source Nodes: [x_203], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf304, arg204_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg204_1
        buf305 = reinterpret_tensor(buf302, (1568, 512), (512, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg205_1, (2048, 512), (1, 2048), 0), out=buf305)
        del arg205_1
        buf306 = reinterpret_tensor(buf305, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf305  # reuse
        buf310 = reinterpret_tensor(buf260, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf260  # reuse
        # Source Nodes: [x_200, x_208, y_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf306, buf283, buf298, arg202_1, arg206_1, arg55_1, arg56_1, buf310, 1568, 512, grid=grid(1568), stream=stream0)
        del arg202_1
        del arg206_1
        del arg55_1
        del arg56_1
        buf311 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (1568, 512), (512, 1), 0), reinterpret_tensor(arg207_1, (512, 1536), (1, 512), 0), out=buf311)
        del arg207_1
        buf312 = reinterpret_tensor(buf310, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf310  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf311, arg208_1, buf312, 802816, grid=grid(802816), stream=stream0)
        buf313 = reinterpret_tensor(buf298, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf298  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf311, arg208_1, buf313, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf314 = reinterpret_tensor(buf294, (128, 196, 196), (38416, 196, 1), 0); del buf294  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf312, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf313, (128, 32, 196), (6272, 196, 1), 0), out=buf314)
        buf317 = reinterpret_tensor(buf291, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf291  # reuse
        # Source Nodes: [x_210], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf314, buf317, 25088, 196, grid=grid(25088), stream=stream0)
        buf318 = reinterpret_tensor(buf313, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf313  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf311, arg208_1, buf318, 802816, grid=grid(802816), stream=stream0)
        del arg208_1
        buf319 = reinterpret_tensor(buf312, (128, 196, 32), (6272, 32, 1), 0); del buf312  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf317, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf318, (128, 196, 32), (6272, 32, 1), 0), out=buf319)
        buf320 = reinterpret_tensor(buf318, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf318  # reuse
        # Source Nodes: [x_211], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf319, buf320, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf321 = reinterpret_tensor(buf319, (1568, 512), (512, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf320, (1568, 512), (512, 1), 0), reinterpret_tensor(arg209_1, (512, 512), (1, 512), 0), out=buf321)
        del arg209_1
        buf325 = reinterpret_tensor(buf320, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf320  # reuse
        # Source Nodes: [x_214, x_215], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf306, buf321, arg210_1, arg57_1, arg58_1, buf325, 1568, 512, grid=grid(1568), stream=stream0)
        del arg57_1
        del arg58_1
        buf326 = reinterpret_tensor(buf304, (1568, 2048), (2048, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf325, (1568, 512), (512, 1), 0), reinterpret_tensor(arg211_1, (512, 2048), (1, 512), 0), out=buf326)
        del arg211_1
        buf327 = reinterpret_tensor(buf326, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf326  # reuse
        # Source Nodes: [x_217], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf327, arg212_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg212_1
        buf328 = reinterpret_tensor(buf325, (1568, 512), (512, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf327, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg213_1, (2048, 512), (1, 2048), 0), out=buf328)
        del arg213_1
        buf329 = reinterpret_tensor(buf328, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf328  # reuse
        buf333 = reinterpret_tensor(buf283, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf283  # reuse
        # Source Nodes: [x_214, x_222, y_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf329, buf306, buf321, arg210_1, arg214_1, arg59_1, arg60_1, buf333, 1568, 512, grid=grid(1568), stream=stream0)
        del arg210_1
        del arg214_1
        del arg59_1
        del arg60_1
        buf334 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf333, (1568, 512), (512, 1), 0), reinterpret_tensor(arg215_1, (512, 1536), (1, 512), 0), out=buf334)
        del arg215_1
        buf335 = reinterpret_tensor(buf333, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf333  # reuse
        # Source Nodes: [x_224], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf334, arg216_1, buf335, 802816, grid=grid(802816), stream=stream0)
        buf336 = reinterpret_tensor(buf321, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf321  # reuse
        # Source Nodes: [x_224], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf334, arg216_1, buf336, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf337 = reinterpret_tensor(buf317, (128, 196, 196), (38416, 196, 1), 0); del buf317  # reuse
        # Source Nodes: [x_224], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf336, (128, 32, 196), (6272, 196, 1), 0), out=buf337)
        buf340 = reinterpret_tensor(buf314, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf314  # reuse
        # Source Nodes: [x_224], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf337, buf340, 25088, 196, grid=grid(25088), stream=stream0)
        buf341 = reinterpret_tensor(buf336, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf336  # reuse
        # Source Nodes: [x_224], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf334, arg216_1, buf341, 802816, grid=grid(802816), stream=stream0)
        del arg216_1
        buf342 = reinterpret_tensor(buf335, (128, 196, 32), (6272, 32, 1), 0); del buf335  # reuse
        # Source Nodes: [x_224], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf341, (128, 196, 32), (6272, 32, 1), 0), out=buf342)
        buf343 = reinterpret_tensor(buf341, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf341  # reuse
        # Source Nodes: [x_225], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf342, buf343, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf344 = reinterpret_tensor(buf342, (1568, 512), (512, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (1568, 512), (512, 1), 0), reinterpret_tensor(arg217_1, (512, 512), (1, 512), 0), out=buf344)
        del arg217_1
        buf348 = reinterpret_tensor(buf343, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf343  # reuse
        # Source Nodes: [x_228, x_229], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf329, buf344, arg218_1, arg61_1, arg62_1, buf348, 1568, 512, grid=grid(1568), stream=stream0)
        del arg61_1
        del arg62_1
        buf349 = reinterpret_tensor(buf327, (1568, 2048), (2048, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf348, (1568, 512), (512, 1), 0), reinterpret_tensor(arg219_1, (512, 2048), (1, 512), 0), out=buf349)
        del arg219_1
        buf350 = reinterpret_tensor(buf349, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf349  # reuse
        # Source Nodes: [x_231], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf350, arg220_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg220_1
        buf351 = reinterpret_tensor(buf348, (1568, 512), (512, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg221_1, (2048, 512), (1, 2048), 0), out=buf351)
        del arg221_1
        buf352 = reinterpret_tensor(buf351, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf351  # reuse
        buf356 = reinterpret_tensor(buf306, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf306  # reuse
        # Source Nodes: [x_228, x_236, y_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf352, buf329, buf344, arg218_1, arg222_1, arg63_1, arg64_1, buf356, 1568, 512, grid=grid(1568), stream=stream0)
        del arg218_1
        del arg222_1
        del arg63_1
        del arg64_1
        buf357 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (1568, 512), (512, 1), 0), reinterpret_tensor(arg223_1, (512, 1536), (1, 512), 0), out=buf357)
        del arg223_1
        buf358 = reinterpret_tensor(buf356, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf356  # reuse
        # Source Nodes: [x_238], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf357, arg224_1, buf358, 802816, grid=grid(802816), stream=stream0)
        buf359 = reinterpret_tensor(buf344, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf344  # reuse
        # Source Nodes: [x_238], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf357, arg224_1, buf359, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf360 = reinterpret_tensor(buf340, (128, 196, 196), (38416, 196, 1), 0); del buf340  # reuse
        # Source Nodes: [x_238], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf358, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf359, (128, 32, 196), (6272, 196, 1), 0), out=buf360)
        buf363 = reinterpret_tensor(buf337, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf337  # reuse
        # Source Nodes: [x_238], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf360, buf363, 25088, 196, grid=grid(25088), stream=stream0)
        buf364 = reinterpret_tensor(buf359, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf359  # reuse
        # Source Nodes: [x_238], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf357, arg224_1, buf364, 802816, grid=grid(802816), stream=stream0)
        del arg224_1
        buf365 = reinterpret_tensor(buf358, (128, 196, 32), (6272, 32, 1), 0); del buf358  # reuse
        # Source Nodes: [x_238], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf363, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf364, (128, 196, 32), (6272, 32, 1), 0), out=buf365)
        buf366 = reinterpret_tensor(buf364, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf364  # reuse
        # Source Nodes: [x_239], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf365, buf366, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf367 = reinterpret_tensor(buf365, (1568, 512), (512, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf366, (1568, 512), (512, 1), 0), reinterpret_tensor(arg225_1, (512, 512), (1, 512), 0), out=buf367)
        del arg225_1
        buf371 = reinterpret_tensor(buf366, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf366  # reuse
        # Source Nodes: [x_242, x_243], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf352, buf367, arg226_1, arg65_1, arg66_1, buf371, 1568, 512, grid=grid(1568), stream=stream0)
        del arg65_1
        del arg66_1
        buf372 = reinterpret_tensor(buf350, (1568, 2048), (2048, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf371, (1568, 512), (512, 1), 0), reinterpret_tensor(arg227_1, (512, 2048), (1, 512), 0), out=buf372)
        del arg227_1
        buf373 = reinterpret_tensor(buf372, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf372  # reuse
        # Source Nodes: [x_245], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf373, arg228_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg228_1
        buf374 = reinterpret_tensor(buf371, (1568, 512), (512, 1), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf373, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg229_1, (2048, 512), (1, 2048), 0), out=buf374)
        del arg229_1
        buf375 = reinterpret_tensor(buf374, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf374  # reuse
        buf379 = reinterpret_tensor(buf329, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf329  # reuse
        # Source Nodes: [x_242, x_250, y_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf375, buf352, buf367, arg226_1, arg230_1, arg67_1, arg68_1, buf379, 1568, 512, grid=grid(1568), stream=stream0)
        del arg226_1
        del arg230_1
        del arg67_1
        del arg68_1
        buf380 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf379, (1568, 512), (512, 1), 0), reinterpret_tensor(arg231_1, (512, 1536), (1, 512), 0), out=buf380)
        del arg231_1
        buf381 = reinterpret_tensor(buf379, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf379  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf380, arg232_1, buf381, 802816, grid=grid(802816), stream=stream0)
        buf382 = reinterpret_tensor(buf367, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf367  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf380, arg232_1, buf382, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf383 = reinterpret_tensor(buf363, (128, 196, 196), (38416, 196, 1), 0); del buf363  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf381, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf382, (128, 32, 196), (6272, 196, 1), 0), out=buf383)
        buf386 = reinterpret_tensor(buf360, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf360  # reuse
        # Source Nodes: [x_252], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf383, buf386, 25088, 196, grid=grid(25088), stream=stream0)
        buf387 = reinterpret_tensor(buf382, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf382  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf380, arg232_1, buf387, 802816, grid=grid(802816), stream=stream0)
        del arg232_1
        buf388 = reinterpret_tensor(buf381, (128, 196, 32), (6272, 32, 1), 0); del buf381  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf386, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf387, (128, 196, 32), (6272, 32, 1), 0), out=buf388)
        buf389 = reinterpret_tensor(buf387, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf387  # reuse
        # Source Nodes: [x_253], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf388, buf389, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf390 = reinterpret_tensor(buf388, (1568, 512), (512, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (1568, 512), (512, 1), 0), reinterpret_tensor(arg233_1, (512, 512), (1, 512), 0), out=buf390)
        del arg233_1
        buf394 = reinterpret_tensor(buf389, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf389  # reuse
        # Source Nodes: [x_256, x_257], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf375, buf390, arg234_1, arg69_1, arg70_1, buf394, 1568, 512, grid=grid(1568), stream=stream0)
        del arg69_1
        del arg70_1
        buf395 = reinterpret_tensor(buf373, (1568, 2048), (2048, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (1568, 512), (512, 1), 0), reinterpret_tensor(arg235_1, (512, 2048), (1, 512), 0), out=buf395)
        del arg235_1
        buf396 = reinterpret_tensor(buf395, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf395  # reuse
        # Source Nodes: [x_259], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf396, arg236_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg236_1
        buf397 = reinterpret_tensor(buf394, (1568, 512), (512, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf396, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg237_1, (2048, 512), (1, 2048), 0), out=buf397)
        del arg237_1
        buf398 = reinterpret_tensor(buf397, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf397  # reuse
        buf402 = reinterpret_tensor(buf352, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf352  # reuse
        # Source Nodes: [x_256, x_264, y_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf398, buf375, buf390, arg234_1, arg238_1, arg71_1, arg72_1, buf402, 1568, 512, grid=grid(1568), stream=stream0)
        del arg234_1
        del arg238_1
        del arg71_1
        del arg72_1
        buf403 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf402, (1568, 512), (512, 1), 0), reinterpret_tensor(arg239_1, (512, 1536), (1, 512), 0), out=buf403)
        del arg239_1
        buf404 = reinterpret_tensor(buf402, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf402  # reuse
        # Source Nodes: [x_266], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf403, arg240_1, buf404, 802816, grid=grid(802816), stream=stream0)
        buf405 = reinterpret_tensor(buf390, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf390  # reuse
        # Source Nodes: [x_266], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf403, arg240_1, buf405, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf406 = reinterpret_tensor(buf386, (128, 196, 196), (38416, 196, 1), 0); del buf386  # reuse
        # Source Nodes: [x_266], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf404, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf405, (128, 32, 196), (6272, 196, 1), 0), out=buf406)
        buf409 = reinterpret_tensor(buf383, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf383  # reuse
        # Source Nodes: [x_266], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf406, buf409, 25088, 196, grid=grid(25088), stream=stream0)
        buf410 = reinterpret_tensor(buf405, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf405  # reuse
        # Source Nodes: [x_266], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf403, arg240_1, buf410, 802816, grid=grid(802816), stream=stream0)
        del arg240_1
        buf411 = reinterpret_tensor(buf404, (128, 196, 32), (6272, 32, 1), 0); del buf404  # reuse
        # Source Nodes: [x_266], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf410, (128, 196, 32), (6272, 32, 1), 0), out=buf411)
        buf412 = reinterpret_tensor(buf410, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf410  # reuse
        # Source Nodes: [x_267], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf411, buf412, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf413 = reinterpret_tensor(buf411, (1568, 512), (512, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (1568, 512), (512, 1), 0), reinterpret_tensor(arg241_1, (512, 512), (1, 512), 0), out=buf413)
        del arg241_1
        buf417 = reinterpret_tensor(buf412, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf412  # reuse
        # Source Nodes: [x_270, x_271], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf398, buf413, arg242_1, arg73_1, arg74_1, buf417, 1568, 512, grid=grid(1568), stream=stream0)
        del arg73_1
        del arg74_1
        buf418 = reinterpret_tensor(buf396, (1568, 2048), (2048, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf417, (1568, 512), (512, 1), 0), reinterpret_tensor(arg243_1, (512, 2048), (1, 512), 0), out=buf418)
        del arg243_1
        buf419 = reinterpret_tensor(buf418, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf418  # reuse
        # Source Nodes: [x_273], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf419, arg244_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg244_1
        buf420 = reinterpret_tensor(buf417, (1568, 512), (512, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf419, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg245_1, (2048, 512), (1, 2048), 0), out=buf420)
        del arg245_1
        buf421 = reinterpret_tensor(buf420, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf420  # reuse
        buf425 = reinterpret_tensor(buf375, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf375  # reuse
        # Source Nodes: [x_270, x_278, y_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf421, buf398, buf413, arg242_1, arg246_1, arg75_1, arg76_1, buf425, 1568, 512, grid=grid(1568), stream=stream0)
        del arg242_1
        del arg246_1
        del arg75_1
        del arg76_1
        buf426 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf425, (1568, 512), (512, 1), 0), reinterpret_tensor(arg247_1, (512, 1536), (1, 512), 0), out=buf426)
        del arg247_1
        buf427 = reinterpret_tensor(buf425, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf425  # reuse
        # Source Nodes: [x_280], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf426, arg248_1, buf427, 802816, grid=grid(802816), stream=stream0)
        buf428 = reinterpret_tensor(buf413, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf413  # reuse
        # Source Nodes: [x_280], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf426, arg248_1, buf428, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf429 = reinterpret_tensor(buf409, (128, 196, 196), (38416, 196, 1), 0); del buf409  # reuse
        # Source Nodes: [x_280], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf427, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf428, (128, 32, 196), (6272, 196, 1), 0), out=buf429)
        buf432 = reinterpret_tensor(buf406, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf406  # reuse
        # Source Nodes: [x_280], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf429, buf432, 25088, 196, grid=grid(25088), stream=stream0)
        buf433 = reinterpret_tensor(buf428, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf428  # reuse
        # Source Nodes: [x_280], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf426, arg248_1, buf433, 802816, grid=grid(802816), stream=stream0)
        del arg248_1
        buf434 = reinterpret_tensor(buf427, (128, 196, 32), (6272, 32, 1), 0); del buf427  # reuse
        # Source Nodes: [x_280], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf433, (128, 196, 32), (6272, 32, 1), 0), out=buf434)
        buf435 = reinterpret_tensor(buf433, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf433  # reuse
        # Source Nodes: [x_281], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf434, buf435, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf436 = reinterpret_tensor(buf434, (1568, 512), (512, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (1568, 512), (512, 1), 0), reinterpret_tensor(arg249_1, (512, 512), (1, 512), 0), out=buf436)
        del arg249_1
        buf440 = reinterpret_tensor(buf435, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf435  # reuse
        # Source Nodes: [x_284, x_285], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf421, buf436, arg250_1, arg77_1, arg78_1, buf440, 1568, 512, grid=grid(1568), stream=stream0)
        del arg77_1
        del arg78_1
        buf441 = reinterpret_tensor(buf419, (1568, 2048), (2048, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf440, (1568, 512), (512, 1), 0), reinterpret_tensor(arg251_1, (512, 2048), (1, 512), 0), out=buf441)
        del arg251_1
        buf442 = reinterpret_tensor(buf441, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf441  # reuse
        # Source Nodes: [x_287], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf442, arg252_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg252_1
        buf443 = reinterpret_tensor(buf440, (1568, 512), (512, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf442, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg253_1, (2048, 512), (1, 2048), 0), out=buf443)
        del arg253_1
        buf444 = reinterpret_tensor(buf443, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf443  # reuse
        buf448 = reinterpret_tensor(buf398, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf398  # reuse
        # Source Nodes: [x_284, x_292, y_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf444, buf421, buf436, arg250_1, arg254_1, arg79_1, arg80_1, buf448, 1568, 512, grid=grid(1568), stream=stream0)
        del arg250_1
        del arg254_1
        del arg79_1
        del arg80_1
        buf449 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf448, (1568, 512), (512, 1), 0), reinterpret_tensor(arg255_1, (512, 1536), (1, 512), 0), out=buf449)
        del arg255_1
        buf450 = reinterpret_tensor(buf448, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf448  # reuse
        # Source Nodes: [x_294], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf449, arg256_1, buf450, 802816, grid=grid(802816), stream=stream0)
        buf451 = reinterpret_tensor(buf436, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf436  # reuse
        # Source Nodes: [x_294], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf449, arg256_1, buf451, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf452 = reinterpret_tensor(buf432, (128, 196, 196), (38416, 196, 1), 0); del buf432  # reuse
        # Source Nodes: [x_294], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf451, (128, 32, 196), (6272, 196, 1), 0), out=buf452)
        buf455 = reinterpret_tensor(buf429, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf429  # reuse
        # Source Nodes: [x_294], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf452, buf455, 25088, 196, grid=grid(25088), stream=stream0)
        buf456 = reinterpret_tensor(buf451, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf451  # reuse
        # Source Nodes: [x_294], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf449, arg256_1, buf456, 802816, grid=grid(802816), stream=stream0)
        del arg256_1
        buf457 = reinterpret_tensor(buf450, (128, 196, 32), (6272, 32, 1), 0); del buf450  # reuse
        # Source Nodes: [x_294], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf455, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf456, (128, 196, 32), (6272, 32, 1), 0), out=buf457)
        buf458 = reinterpret_tensor(buf456, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf456  # reuse
        # Source Nodes: [x_295], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf457, buf458, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf459 = reinterpret_tensor(buf457, (1568, 512), (512, 1), 0); del buf457  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (1568, 512), (512, 1), 0), reinterpret_tensor(arg257_1, (512, 512), (1, 512), 0), out=buf459)
        del arg257_1
        buf463 = reinterpret_tensor(buf458, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf458  # reuse
        # Source Nodes: [x_298, x_299], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf444, buf459, arg258_1, arg81_1, arg82_1, buf463, 1568, 512, grid=grid(1568), stream=stream0)
        del arg81_1
        del arg82_1
        buf464 = reinterpret_tensor(buf442, (1568, 2048), (2048, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf463, (1568, 512), (512, 1), 0), reinterpret_tensor(arg259_1, (512, 2048), (1, 512), 0), out=buf464)
        del arg259_1
        buf465 = reinterpret_tensor(buf464, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf464  # reuse
        # Source Nodes: [x_301], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf465, arg260_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg260_1
        buf466 = reinterpret_tensor(buf463, (1568, 512), (512, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg261_1, (2048, 512), (1, 2048), 0), out=buf466)
        del arg261_1
        buf467 = reinterpret_tensor(buf466, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf466  # reuse
        buf471 = reinterpret_tensor(buf421, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf421  # reuse
        # Source Nodes: [x_298, x_306, y_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf467, buf444, buf459, arg258_1, arg262_1, arg83_1, arg84_1, buf471, 1568, 512, grid=grid(1568), stream=stream0)
        del arg258_1
        del arg262_1
        del arg83_1
        del arg84_1
        buf472 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (1568, 512), (512, 1), 0), reinterpret_tensor(arg263_1, (512, 1536), (1, 512), 0), out=buf472)
        del arg263_1
        buf473 = reinterpret_tensor(buf471, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf471  # reuse
        # Source Nodes: [x_308], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf472, arg264_1, buf473, 802816, grid=grid(802816), stream=stream0)
        buf474 = reinterpret_tensor(buf459, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf459  # reuse
        # Source Nodes: [x_308], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf472, arg264_1, buf474, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf475 = reinterpret_tensor(buf455, (128, 196, 196), (38416, 196, 1), 0); del buf455  # reuse
        # Source Nodes: [x_308], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf473, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf474, (128, 32, 196), (6272, 196, 1), 0), out=buf475)
        buf478 = reinterpret_tensor(buf452, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf452  # reuse
        # Source Nodes: [x_308], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf475, buf478, 25088, 196, grid=grid(25088), stream=stream0)
        buf479 = reinterpret_tensor(buf474, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf474  # reuse
        # Source Nodes: [x_308], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf472, arg264_1, buf479, 802816, grid=grid(802816), stream=stream0)
        del arg264_1
        buf480 = reinterpret_tensor(buf473, (128, 196, 32), (6272, 32, 1), 0); del buf473  # reuse
        # Source Nodes: [x_308], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf478, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf479, (128, 196, 32), (6272, 32, 1), 0), out=buf480)
        buf481 = reinterpret_tensor(buf479, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf479  # reuse
        # Source Nodes: [x_309], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf480, buf481, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf482 = reinterpret_tensor(buf480, (1568, 512), (512, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (1568, 512), (512, 1), 0), reinterpret_tensor(arg265_1, (512, 512), (1, 512), 0), out=buf482)
        del arg265_1
        buf486 = reinterpret_tensor(buf481, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf481  # reuse
        # Source Nodes: [x_312, x_313], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf467, buf482, arg266_1, arg85_1, arg86_1, buf486, 1568, 512, grid=grid(1568), stream=stream0)
        del arg85_1
        del arg86_1
        buf487 = reinterpret_tensor(buf465, (1568, 2048), (2048, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (1568, 512), (512, 1), 0), reinterpret_tensor(arg267_1, (512, 2048), (1, 512), 0), out=buf487)
        del arg267_1
        buf488 = reinterpret_tensor(buf487, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf487  # reuse
        # Source Nodes: [x_315], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf488, arg268_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg268_1
        buf489 = reinterpret_tensor(buf486, (1568, 512), (512, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg269_1, (2048, 512), (1, 2048), 0), out=buf489)
        del arg269_1
        buf490 = reinterpret_tensor(buf489, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf489  # reuse
        buf494 = reinterpret_tensor(buf444, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf444  # reuse
        # Source Nodes: [x_312, x_320, y_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf490, buf467, buf482, arg266_1, arg270_1, arg87_1, arg88_1, buf494, 1568, 512, grid=grid(1568), stream=stream0)
        del arg266_1
        del arg270_1
        del arg87_1
        del arg88_1
        buf495 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf494, (1568, 512), (512, 1), 0), reinterpret_tensor(arg271_1, (512, 1536), (1, 512), 0), out=buf495)
        del arg271_1
        buf496 = reinterpret_tensor(buf494, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf494  # reuse
        # Source Nodes: [x_322], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf495, arg272_1, buf496, 802816, grid=grid(802816), stream=stream0)
        buf497 = reinterpret_tensor(buf482, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf482  # reuse
        # Source Nodes: [x_322], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf495, arg272_1, buf497, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf498 = reinterpret_tensor(buf478, (128, 196, 196), (38416, 196, 1), 0); del buf478  # reuse
        # Source Nodes: [x_322], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf496, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf497, (128, 32, 196), (6272, 196, 1), 0), out=buf498)
        buf501 = reinterpret_tensor(buf475, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf475  # reuse
        # Source Nodes: [x_322], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf498, buf501, 25088, 196, grid=grid(25088), stream=stream0)
        buf502 = reinterpret_tensor(buf497, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf497  # reuse
        # Source Nodes: [x_322], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf495, arg272_1, buf502, 802816, grid=grid(802816), stream=stream0)
        del arg272_1
        buf503 = reinterpret_tensor(buf496, (128, 196, 32), (6272, 32, 1), 0); del buf496  # reuse
        # Source Nodes: [x_322], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf501, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf502, (128, 196, 32), (6272, 32, 1), 0), out=buf503)
        buf504 = reinterpret_tensor(buf502, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf502  # reuse
        # Source Nodes: [x_323], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf503, buf504, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf505 = reinterpret_tensor(buf503, (1568, 512), (512, 1), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf504, (1568, 512), (512, 1), 0), reinterpret_tensor(arg273_1, (512, 512), (1, 512), 0), out=buf505)
        del arg273_1
        buf509 = reinterpret_tensor(buf504, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf504  # reuse
        # Source Nodes: [x_326, x_327], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf490, buf505, arg274_1, arg89_1, arg90_1, buf509, 1568, 512, grid=grid(1568), stream=stream0)
        del arg89_1
        del arg90_1
        buf510 = reinterpret_tensor(buf488, (1568, 2048), (2048, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf509, (1568, 512), (512, 1), 0), reinterpret_tensor(arg275_1, (512, 2048), (1, 512), 0), out=buf510)
        del arg275_1
        buf511 = reinterpret_tensor(buf510, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf510  # reuse
        # Source Nodes: [x_329], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf511, arg276_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg276_1
        buf512 = reinterpret_tensor(buf509, (1568, 512), (512, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf511, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg277_1, (2048, 512), (1, 2048), 0), out=buf512)
        del arg277_1
        buf513 = reinterpret_tensor(buf512, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf512  # reuse
        buf517 = reinterpret_tensor(buf467, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf467  # reuse
        # Source Nodes: [x_326, x_334, y_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf513, buf490, buf505, arg274_1, arg278_1, arg91_1, arg92_1, buf517, 1568, 512, grid=grid(1568), stream=stream0)
        del arg274_1
        del arg278_1
        del arg91_1
        del arg92_1
        buf518 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf517, (1568, 512), (512, 1), 0), reinterpret_tensor(arg279_1, (512, 1536), (1, 512), 0), out=buf518)
        del arg279_1
        buf519 = reinterpret_tensor(buf517, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf517  # reuse
        # Source Nodes: [x_336], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf518, arg280_1, buf519, 802816, grid=grid(802816), stream=stream0)
        buf520 = reinterpret_tensor(buf505, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf505  # reuse
        # Source Nodes: [x_336], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf518, arg280_1, buf520, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf521 = reinterpret_tensor(buf501, (128, 196, 196), (38416, 196, 1), 0); del buf501  # reuse
        # Source Nodes: [x_336], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf519, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf520, (128, 32, 196), (6272, 196, 1), 0), out=buf521)
        buf524 = reinterpret_tensor(buf498, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf498  # reuse
        # Source Nodes: [x_336], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf521, buf524, 25088, 196, grid=grid(25088), stream=stream0)
        buf525 = reinterpret_tensor(buf520, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf520  # reuse
        # Source Nodes: [x_336], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf518, arg280_1, buf525, 802816, grid=grid(802816), stream=stream0)
        del arg280_1
        buf526 = reinterpret_tensor(buf519, (128, 196, 32), (6272, 32, 1), 0); del buf519  # reuse
        # Source Nodes: [x_336], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf524, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf525, (128, 196, 32), (6272, 32, 1), 0), out=buf526)
        buf527 = reinterpret_tensor(buf525, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf525  # reuse
        # Source Nodes: [x_337], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf526, buf527, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf528 = reinterpret_tensor(buf526, (1568, 512), (512, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf527, (1568, 512), (512, 1), 0), reinterpret_tensor(arg281_1, (512, 512), (1, 512), 0), out=buf528)
        del arg281_1
        buf532 = reinterpret_tensor(buf527, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf527  # reuse
        # Source Nodes: [x_340, x_341], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf513, buf528, arg282_1, arg93_1, arg94_1, buf532, 1568, 512, grid=grid(1568), stream=stream0)
        del arg93_1
        del arg94_1
        buf533 = reinterpret_tensor(buf511, (1568, 2048), (2048, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf532, (1568, 512), (512, 1), 0), reinterpret_tensor(arg283_1, (512, 2048), (1, 512), 0), out=buf533)
        del arg283_1
        buf534 = reinterpret_tensor(buf533, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf533  # reuse
        # Source Nodes: [x_343], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf534, arg284_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg284_1
        buf535 = reinterpret_tensor(buf532, (1568, 512), (512, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf534, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg285_1, (2048, 512), (1, 2048), 0), out=buf535)
        del arg285_1
        buf536 = reinterpret_tensor(buf535, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf535  # reuse
        buf540 = reinterpret_tensor(buf490, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf490  # reuse
        # Source Nodes: [x_340, x_348, y_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf536, buf513, buf528, arg282_1, arg286_1, arg95_1, arg96_1, buf540, 1568, 512, grid=grid(1568), stream=stream0)
        del arg282_1
        del arg286_1
        del arg95_1
        del arg96_1
        buf541 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (1568, 512), (512, 1), 0), reinterpret_tensor(arg287_1, (512, 1536), (1, 512), 0), out=buf541)
        del arg287_1
        buf542 = reinterpret_tensor(buf540, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf540  # reuse
        # Source Nodes: [x_350], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf541, arg288_1, buf542, 802816, grid=grid(802816), stream=stream0)
        buf543 = reinterpret_tensor(buf528, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf528  # reuse
        # Source Nodes: [x_350], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf541, arg288_1, buf543, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf544 = reinterpret_tensor(buf524, (128, 196, 196), (38416, 196, 1), 0); del buf524  # reuse
        # Source Nodes: [x_350], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf542, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf543, (128, 32, 196), (6272, 196, 1), 0), out=buf544)
        buf547 = reinterpret_tensor(buf521, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf521  # reuse
        # Source Nodes: [x_350], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf544, buf547, 25088, 196, grid=grid(25088), stream=stream0)
        buf548 = reinterpret_tensor(buf543, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf543  # reuse
        # Source Nodes: [x_350], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf541, arg288_1, buf548, 802816, grid=grid(802816), stream=stream0)
        del arg288_1
        buf549 = reinterpret_tensor(buf542, (128, 196, 32), (6272, 32, 1), 0); del buf542  # reuse
        # Source Nodes: [x_350], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf547, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf548, (128, 196, 32), (6272, 32, 1), 0), out=buf549)
        buf550 = reinterpret_tensor(buf548, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf548  # reuse
        # Source Nodes: [x_351], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf549, buf550, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf551 = reinterpret_tensor(buf549, (1568, 512), (512, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf550, (1568, 512), (512, 1), 0), reinterpret_tensor(arg289_1, (512, 512), (1, 512), 0), out=buf551)
        del arg289_1
        buf555 = reinterpret_tensor(buf550, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf550  # reuse
        # Source Nodes: [x_354, x_355], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf536, buf551, arg290_1, arg97_1, arg98_1, buf555, 1568, 512, grid=grid(1568), stream=stream0)
        del arg97_1
        del arg98_1
        buf556 = reinterpret_tensor(buf534, (1568, 2048), (2048, 1), 0); del buf534  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (1568, 512), (512, 1), 0), reinterpret_tensor(arg291_1, (512, 2048), (1, 512), 0), out=buf556)
        del arg291_1
        buf557 = reinterpret_tensor(buf556, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf556  # reuse
        # Source Nodes: [x_357], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf557, arg292_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg292_1
        buf558 = reinterpret_tensor(buf555, (1568, 512), (512, 1), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg293_1, (2048, 512), (1, 2048), 0), out=buf558)
        del arg293_1
        buf559 = reinterpret_tensor(buf558, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf558  # reuse
        buf563 = reinterpret_tensor(buf513, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf513  # reuse
        # Source Nodes: [x_354, x_362, y_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf559, buf536, buf551, arg290_1, arg294_1, arg99_1, arg100_1, buf563, 1568, 512, grid=grid(1568), stream=stream0)
        del arg100_1
        del arg290_1
        del arg294_1
        del arg99_1
        del buf536
        buf564 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf563, (1568, 512), (512, 1), 0), reinterpret_tensor(arg295_1, (512, 1536), (1, 512), 0), out=buf564)
        del arg295_1
        buf565 = reinterpret_tensor(buf563, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf563  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_37.run(buf564, arg296_1, buf565, 802816, grid=grid(802816), stream=stream0)
        buf566 = reinterpret_tensor(buf551, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf551  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_38.run(buf564, arg296_1, buf566, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf567 = reinterpret_tensor(buf547, (128, 196, 196), (38416, 196, 1), 0); del buf547  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf565, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf566, (128, 32, 196), (6272, 196, 1), 0), out=buf567)
        buf570 = reinterpret_tensor(buf544, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf544  # reuse
        # Source Nodes: [x_364], Original ATen: [aten._softmax]
        triton_per_fused__softmax_39.run(buf567, buf570, 25088, 196, grid=grid(25088), stream=stream0)
        del buf567
        buf571 = reinterpret_tensor(buf566, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf566  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf564, arg296_1, buf571, 802816, grid=grid(802816), stream=stream0)
        del arg296_1
        del buf564
        buf572 = reinterpret_tensor(buf565, (128, 196, 32), (6272, 32, 1), 0); del buf565  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf570, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf571, (128, 196, 32), (6272, 32, 1), 0), out=buf572)
        del buf570
        buf573 = reinterpret_tensor(buf571, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf571  # reuse
        # Source Nodes: [x_365], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf572, buf573, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf574 = reinterpret_tensor(buf572, (1568, 512), (512, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf573, (1568, 512), (512, 1), 0), reinterpret_tensor(arg297_1, (512, 512), (1, 512), 0), out=buf574)
        del arg297_1
        buf578 = reinterpret_tensor(buf573, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf573  # reuse
        # Source Nodes: [x_368, x_369], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf559, buf574, arg298_1, arg101_1, arg102_1, buf578, 1568, 512, grid=grid(1568), stream=stream0)
        del arg101_1
        del arg102_1
        buf579 = reinterpret_tensor(buf557, (1568, 2048), (2048, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (1568, 512), (512, 1), 0), reinterpret_tensor(arg299_1, (512, 2048), (1, 512), 0), out=buf579)
        del arg299_1
        buf580 = reinterpret_tensor(buf579, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf579  # reuse
        # Source Nodes: [x_371], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf580, arg300_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg300_1
        buf581 = reinterpret_tensor(buf578, (1568, 512), (512, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg301_1, (2048, 512), (1, 2048), 0), out=buf581)
        del arg301_1
        del buf580
        buf582 = reinterpret_tensor(buf581, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf581  # reuse
        buf583 = reinterpret_tensor(buf139, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf139  # reuse
        buf584 = reinterpret_tensor(buf138, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf138  # reuse
        # Source Nodes: [x_368, x_377, x_382], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_48.run(buf582, buf559, buf574, arg298_1, arg302_1, buf583, buf584, 1568, 512, grid=grid(1568), stream=stream0)
        del arg298_1
        del arg302_1
        del buf559
        del buf574
        buf586 = empty_strided((8, 512, 1, 1, 2), (1024, 1, 8192, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_385], Original ATen: [aten.mean]
        triton_red_fused_mean_49.run(buf582, buf583, buf584, arg103_1, arg104_1, buf586, 8192, 98, grid=grid(8192), stream=stream0)
        del arg103_1
        del arg104_1
        del buf582
        del buf583
        del buf584
        buf587 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf588 = reinterpret_tensor(buf587, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf587  # reuse
        # Source Nodes: [x_385], Original ATen: [aten.mean]
        triton_per_fused_mean_50.run(buf588, buf586, 4096, 2, grid=grid(4096), stream=stream0)
        del buf586
        buf589 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_389], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg304_1, reinterpret_tensor(buf588, (8, 512), (512, 1), 0), reinterpret_tensor(arg303_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf589)
        del arg303_1
        del arg304_1
        return (buf589, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 16, 196, 128), (401408, 25088, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1, 4, 196, 256), (200704, 50176, 256, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('jx_nest_base', benchmark_compiled_module)
