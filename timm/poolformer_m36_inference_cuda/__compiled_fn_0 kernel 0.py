
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3mdeneejappqqe5fk6zuhypf6spx7mpv4obazedrrc25iia6ca.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bdvhkxfvfrsql3bu5nxnsi3gqefncfcxmy433okmime2thguph.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2v22tglxne6utfkq4222ooif2shit26fr2e3wxfeehtp3r3agex.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean
triton_red_fused_native_group_norm_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 296
    rnumel = 8137
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 37
    x1 = (xindex // 37)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (8137*x0)
        tmp1 = tl.full([1, 1], 301056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((301056*x1) + ((r2 + (8137*x0)) % 301056)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (((r2 + (8137*x0)) // 3136) % 96), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
    tl.store(out_ptr2 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vw/cvw74cpmxvkzq4qhjc3zkq4vy6pk7dqrkomr5exdunonvl2rfmi5.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => var_mean
triton_per_fused_native_group_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 37
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (37*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (37*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (37*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqhvoytfnysmwovwial3dnxbjk5etfontydrhlk6zxtakgpks4u.py
# Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# group_norm => add_1, mul_1
triton_poi_fused_native_group_norm_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 96
    x2 = (xindex // 301056)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 301056.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjo4mtrrfcam5yq3evqqjunl55bxp545mk4esuz7et2sxpuxldea.py
# Source Nodes: [mul, sub, x, x_4, y], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.mul, aten.sub]
# mul => mul_2
# sub => sub_1
# x => convolution
# x_4 => add_2
# y => avg_pool2d
triton_poi_fused_add_avg_pool2d_convolution_mul_sub_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mul_sub_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x4 = xindex
    x6 = (xindex // 3136) % 96
    tmp90 = tl.load(in_ptr1 + (x4), None)
    tmp91 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (x4), None)
    tmp95 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-57) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-56) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-55) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (55 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (56 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (57 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp92 = tmp90 + tmp91
    tmp94 = tmp89 - tmp93
    tmp96 = tmp94 * tmp95
    tmp97 = tmp92 + tmp96
    tl.store(in_out_ptr0 + (x4), tmp97, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch26k2tbhwpqfc2pqvn4ualb7zxnfu7sc67bv2lgsvtyvzywx7dg.py
# Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
# group_norm_1 => var_mean_1
triton_red_fused_native_group_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 296
    rnumel = 8137
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 37
    x1 = (xindex // 37)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (8137*x0)
        tmp1 = tl.full([1, 1], 301056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((301056*x1) + ((r2 + (8137*x0)) % 301056)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cacs5m47shkvy4ta4szaghvgbvkndsqke2zhifuixl4j72nxww5h.py
# Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
# group_norm_1 => add_4, mul_4
triton_poi_fused_native_group_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 96)
    y0 = yindex % 96
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 301056.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chpaoy5lyilxnfyadlywe3arfpqkulilsruyt5cyamhqx5hg2hos.py
# Source Nodes: [group_norm_1, x_5, x_6], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
# group_norm_1 => add_4, mul_4
# x_5 => convolution_1
# x_6 => add_5, erf, mul_5, mul_6, mul_7
triton_poi_fused_convolution_gelu_native_group_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_native_group_norm_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + (y0 + (384*x2) + (1204224*y1)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ua/cualbeicgj3cdgjh3igjaf3cpcpgpzhp4aeeomltyy4zabt25zto.py
# Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# group_norm_2 => var_mean_2
triton_red_fused_native_group_norm_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 296
    rnumel = 8137
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 37
    x1 = (xindex // 37)
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (8137*x0)
        tmp1 = tl.full([1, 1], 301056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((301056*x1) + ((r2 + (8137*x0)) % 301056)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((301056*x1) + ((r2 + (8137*x0)) % 301056)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (((r2 + (8137*x0)) // 3136) % 96), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (((r2 + (8137*x0)) // 3136) % 96), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 + tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = 1.0
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp19 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_combine(
            tmp21_mean, tmp21_m2, tmp21_weight,
            tmp18, tmp19, tmp20
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
    tl.store(out_ptr2 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v7/cv7mf254z6itbarobwbpzonqdpqwnequ7bzlg3sodzh5mwccxyki.py
# Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# group_norm_2 => add_8, mul_10
triton_poi_fused_native_group_norm_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 96
    x2 = (xindex // 301056)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 301056.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mh/cmhrbt2oih4czsfrngmjfa2nv2awkjrfrbmzwt5555jcocffpg6h.py
# Source Nodes: [group_norm_1, mul_1, mul_2, sub_1, x_11, x_12, x_5, x_6, x_9, y_1], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
# group_norm_1 => add_4, mul_4
# mul_1 => mul_8
# mul_2 => mul_11
# sub_1 => sub_4
# x_11 => add_6
# x_12 => add_9
# x_5 => convolution_1
# x_6 => add_5, erf, mul_5, mul_6, mul_7
# x_9 => convolution_2
# y_1 => avg_pool2d_1
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x4 = xindex
    x6 = (xindex // 3136) % 96
    tmp90 = tl.load(in_out_ptr0 + (x4), None)
    tmp91 = tl.load(in_ptr1 + (x4), None)
    tmp92 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr0 + (x4), None)
    tmp99 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-57) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-56) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-55) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (55 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (56 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (57 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tmp90 + tmp95
    tmp98 = tmp89 - tmp97
    tmp100 = tmp98 * tmp99
    tmp101 = tmp96 + tmp100
    tl.store(in_out_ptr0 + (x4), tmp101, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/churbq67pycrp336tpekxakcm4avc5s7pkaf4rcwmc63ghx7dhpq.py
# Source Nodes: [group_norm_9, mul_10, mul_9, sub_5, x_37, x_38, x_41, x_43, x_44, y_5], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
# group_norm_9 => add_32, mul_40
# mul_10 => mul_47
# mul_9 => mul_44
# sub_5 => sub_16
# x_37 => convolution_9
# x_38 => add_33, erf_4, mul_41, mul_42, mul_43
# x_41 => convolution_10
# x_43 => add_34
# x_44 => add_37
# y_5 => avg_pool2d_5
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x4 = xindex
    x6 = (xindex // 3136) % 96
    tmp90 = tl.load(in_ptr1 + (x4), None)
    tmp91 = tl.load(in_ptr2 + (x4), None)
    tmp92 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr0 + (x4), None)
    tmp99 = tl.load(in_ptr5 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-57) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-56) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-55) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (55 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (56 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (57 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tmp90 + tmp95
    tmp98 = tmp89 - tmp97
    tmp100 = tmp98 * tmp99
    tmp101 = tmp96 + tmp100
    tl.store(in_out_ptr0 + (x4), tmp101, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/me/cmetzlx6sgecy2nw2vecqtjlwuqfh37j5xdgfmwtnb276ufxkwhu.py
# Source Nodes: [group_norm_11, mul_11, x_45, x_46, x_49, x_52], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
# group_norm_11 => add_39, mul_49
# mul_11 => mul_53
# x_45 => convolution_11
# x_46 => add_40, erf_5, mul_50, mul_51, mul_52
# x_49 => convolution_12
# x_52 => add_41
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cfl7rehuk2lyi5ve4ofyl3rpfssbohk2rwyo4r3urxopn4pv2bef.py
# Source Nodes: [group_norm_11, mul_11, x_45, x_46, x_49, x_52, x_55], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
# group_norm_11 => add_39, mul_49
# mul_11 => mul_53
# x_45 => convolution_11
# x_46 => add_40, erf_5, mul_50, mul_51, mul_52
# x_49 => convolution_12
# x_52 => add_41
# x_55 => convolution_13
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (864*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3tccpqccsjoibb7hwqnbux75iflnfyuufeot6ahlzitxkztyfo.py
# Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
# group_norm_12 => var_mean_12
triton_red_fused_native_group_norm_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 7923
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7923*x0)
        tmp1 = tl.full([1, 1], 150528, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((150528*x1) + ((r2 + (7923*x0)) % 150528)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (((r2 + (7923*x0)) // 784) % 192), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
    tl.store(out_ptr2 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxmefrxsdyd4ybfof7n6764b4nwb3aju2x6d3exayqq6jiek2ah.py
# Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
# group_norm_12 => var_mean_12
triton_per_fused_native_group_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (19*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxt7zyh7u5iytjuwn335w55lqbsfnfesenujx5jdty7purwbtmp.py
# Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
# group_norm_12 => add_43, mul_55
triton_poi_fused_native_group_norm_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_17', 'mutated_arg_names': []},
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
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 150528.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ud/cudwsr42vpkxd5lnawitogir5gehk2ayvex2qmaobmxjghb5yrta.py
# Source Nodes: [group_norm_11, mul_11, mul_12, sub_6, x_45, x_46, x_49, x_52, x_55, x_56, y_6], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
# group_norm_11 => add_39, mul_49
# mul_11 => mul_53
# mul_12 => mul_56
# sub_6 => sub_19
# x_45 => convolution_11
# x_46 => add_40, erf_5, mul_50, mul_51, mul_52
# x_49 => convolution_12
# x_52 => add_41
# x_55 => convolution_13
# x_56 => add_44
# y_6 => avg_pool2d_6
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x4 = xindex
    x6 = (xindex // 784) % 192
    tmp90 = tl.load(in_out_ptr0 + (x4), None)
    tmp91 = tl.load(in_ptr1 + (x6), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (x4), None)
    tmp95 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-29) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-28) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-27) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (27 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (28 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (29 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp92 = tmp90 + tmp91
    tmp94 = tmp89 - tmp93
    tmp96 = tmp94 * tmp95
    tmp97 = tmp92 + tmp96
    tl.store(in_out_ptr0 + (x4), tmp97, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/cto5j3ek73mshp7vgar3phydz24h4bvl2zliodlc5ghtkkfzv2qi.py
# Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
# group_norm_13 => var_mean_13
triton_red_fused_native_group_norm_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 7923
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7923*x0)
        tmp1 = tl.full([1, 1], 150528, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((150528*x1) + ((r2 + (7923*x0)) % 150528)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3j/c3j7c4fgkc7hjbztgqh2kyluqobhv3ry6eym3gotcek7dh5y3rom.py
# Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
# group_norm_13 => add_46, mul_58
triton_poi_fused_native_group_norm_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 192)
    y0 = yindex % 192
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 150528.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca24vex5iuosh2kglhscrfczjkengxrplil6b6fooazi4gdufxfi.py
# Source Nodes: [group_norm_13, x_57, x_58], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
# group_norm_13 => add_46, mul_58
# x_57 => convolution_14
# x_58 => add_47, erf_6, mul_59, mul_60, mul_61
triton_poi_fused_convolution_gelu_native_group_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_native_group_norm_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + (y0 + (768*x2) + (602112*y1)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5vzpsseyyjbyv333wcph3g7a6gbph7r6anlf45nsydpjft4627.py
# Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
# group_norm_14 => var_mean_14
triton_red_fused_native_group_norm_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 7923
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7923*x0)
        tmp1 = tl.full([1, 1], 150528, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((150528*x1) + ((r2 + (7923*x0)) % 150528)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((150528*x1) + ((r2 + (7923*x0)) % 150528)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (((r2 + (7923*x0)) // 784) % 192), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (((r2 + (7923*x0)) // 784) % 192), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 + tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = 1.0
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp19 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_combine(
            tmp21_mean, tmp21_m2, tmp21_weight,
            tmp18, tmp19, tmp20
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
    tl.store(out_ptr2 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3tqfqhbp7vb55s76ytxep7menugrb2qpwgvnfbg6a4nbl2k2cc.py
# Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
# group_norm_14 => add_50, mul_64
triton_poi_fused_native_group_norm_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_23', 'mutated_arg_names': []},
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
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 150528.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2qy6phw3t6bkof3h6hf3incypqvpzjgwukbrklg7wmbesvvi77.py
# Source Nodes: [group_norm_13, mul_13, mul_14, sub_7, x_57, x_58, x_61, x_63, x_64, y_7], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
# group_norm_13 => add_46, mul_58
# mul_13 => mul_62
# mul_14 => mul_65
# sub_7 => sub_22
# x_57 => convolution_14
# x_58 => add_47, erf_6, mul_59, mul_60, mul_61
# x_61 => convolution_15
# x_63 => add_48
# x_64 => add_51
# y_7 => avg_pool2d_7
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x4 = xindex
    x6 = (xindex // 784) % 192
    tmp90 = tl.load(in_out_ptr0 + (x4), None)
    tmp91 = tl.load(in_ptr1 + (x4), None)
    tmp92 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr0 + (x4), None)
    tmp99 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-29) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-28) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-27) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (27 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (28 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (29 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tmp90 + tmp95
    tmp98 = tmp89 - tmp97
    tmp100 = tmp98 * tmp99
    tmp101 = tmp96 + tmp100
    tl.store(in_out_ptr0 + (x4), tmp101, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctkjhdoajbabj5r3ss36vxjwxpev2o2rt6vq7bv32ltts76hwvrc.py
# Source Nodes: [group_norm_23, mul_23, x_101, x_104, x_97, x_98], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
# group_norm_23 => add_81, mul_103
# mul_23 => mul_107
# x_101 => convolution_25
# x_104 => add_83
# x_97 => convolution_24
# x_98 => add_82, erf_11, mul_104, mul_105, mul_106
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rbiagzz63cmvh34iefjmpwl4vcmfbc7xk2hlnjkpk3jwd6rf5e.py
# Source Nodes: [group_norm_23, mul_23, x_101, x_104, x_107, x_97, x_98], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
# group_norm_23 => add_81, mul_103
# mul_23 => mul_107
# x_101 => convolution_25
# x_104 => add_83
# x_107 => convolution_26
# x_97 => convolution_24
# x_98 => add_82, erf_11, mul_104, mul_105, mul_106
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbbhdagg5fnkqctonnygfihsonf43td2vh67sqh44n2lf55enam.py
# Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
# group_norm_24 => var_mean_24
triton_red_fused_native_group_norm_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 10
    x1 = (xindex // 10)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 75264, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((75264*x1) + ((r2 + (7527*x0)) % 75264)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (((r2 + (7527*x0)) // 196) % 384), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
    tl.store(out_ptr2 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehfvcyh3xfd52nsic2utjf2dbc4whqpasbqg72zwqtzu5bcldrw.py
# Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
# group_norm_24 => var_mean_24
triton_per_fused_native_group_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (10*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (10*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7wfmfajepil4jreygujqhbyk4zftgyaoufly5i76t7upryvyc4.py
# Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
# group_norm_24 => add_85, mul_109
triton_poi_fused_native_group_norm_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_29', 'mutated_arg_names': []},
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
    x2 = (xindex // 75264)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 75264.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xyb7k5lz3bnovzc3r7vie53nxdnw2c5j5iqom4z7bzalexthsy.py
# Source Nodes: [group_norm_23, mul_23, mul_24, sub_12, x_101, x_104, x_107, x_108, x_97, x_98, y_12], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
# group_norm_23 => add_81, mul_103
# mul_23 => mul_107
# mul_24 => mul_110
# sub_12 => sub_37
# x_101 => convolution_25
# x_104 => add_83
# x_107 => convolution_26
# x_108 => add_86
# x_97 => convolution_24
# x_98 => add_82, erf_11, mul_104, mul_105, mul_106
# y_12 => avg_pool2d_12
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x4 = xindex
    x6 = (xindex // 196) % 384
    tmp90 = tl.load(in_out_ptr0 + (x4), None)
    tmp91 = tl.load(in_ptr1 + (x6), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (x4), None)
    tmp95 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-15) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-14) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-13) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (13 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (14 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (15 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp92 = tmp90 + tmp91
    tmp94 = tmp89 - tmp93
    tmp96 = tmp94 * tmp95
    tmp97 = tmp92 + tmp96
    tl.store(in_out_ptr0 + (x4), tmp97, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/cssdgjlcw47k4g3st5xxipi3aswcz3neta7ldgygmbvnzz4ywhqg.py
# Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
# group_norm_25 => var_mean_25
triton_red_fused_native_group_norm_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 10
    x1 = (xindex // 10)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 75264, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((75264*x1) + ((r2 + (7527*x0)) % 75264)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kjr76a3zny2lfqwfckqmzyye3ahj26nqkzbso6lgehldhd34pe.py
# Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
# group_norm_25 => add_88, mul_112
triton_poi_fused_native_group_norm_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 384)
    y0 = yindex % 384
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 75264.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zzkeghzvvp2uluxowvqfbio63xww43w7yqacqeklggk6wphinm.py
# Source Nodes: [group_norm_25, x_109, x_110], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
# group_norm_25 => add_88, mul_112
# x_109 => convolution_27
# x_110 => add_89, erf_12, mul_113, mul_114, mul_115
triton_poi_fused_convolution_gelu_native_group_norm_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_native_group_norm_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + (y0 + (1536*x2) + (301056*y1)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cdual3vtcb67ae4j3xtsqrrpi3bp5znc4hkcdevsmpwa6sm4k5xt.py
# Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
# group_norm_26 => var_mean_26
triton_red_fused_native_group_norm_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 10
    x1 = (xindex // 10)
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 75264, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((75264*x1) + ((r2 + (7527*x0)) % 75264)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((75264*x1) + ((r2 + (7527*x0)) % 75264)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (((r2 + (7527*x0)) // 196) % 384), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (((r2 + (7527*x0)) // 196) % 384), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 + tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = 1.0
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp19 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_combine(
            tmp21_mean, tmp21_m2, tmp21_weight,
            tmp18, tmp19, tmp20
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
    tl.store(out_ptr2 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65gcr5i4aiqp2jy6joy6vtumj7idb4zat4pl3qcq7yzwp5ujdow.py
# Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
# group_norm_26 => add_92, mul_118
triton_poi_fused_native_group_norm_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_35', 'mutated_arg_names': []},
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
    x2 = (xindex // 75264)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 75264.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrhaog6c5h3ppfosd5uoobh2sm56ysu75il77tu4aafensb5j77.py
# Source Nodes: [group_norm_25, mul_25, mul_26, sub_13, x_109, x_110, x_113, x_115, x_116, y_13], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
# group_norm_25 => add_88, mul_112
# mul_25 => mul_116
# mul_26 => mul_119
# sub_13 => sub_40
# x_109 => convolution_27
# x_110 => add_89, erf_12, mul_113, mul_114, mul_115
# x_113 => convolution_28
# x_115 => add_90
# x_116 => add_93
# y_13 => avg_pool2d_13
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x4 = xindex
    x6 = (xindex // 196) % 384
    tmp90 = tl.load(in_out_ptr0 + (x4), None)
    tmp91 = tl.load(in_ptr1 + (x4), None)
    tmp92 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr0 + (x4), None)
    tmp99 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-15) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-14) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-13) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (13 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (14 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (15 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tmp90 + tmp95
    tmp98 = tmp89 - tmp97
    tmp100 = tmp98 * tmp99
    tmp101 = tmp96 + tmp100
    tl.store(in_out_ptr0 + (x4), tmp101, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cld2atnrxtpeqla2uhrhjkzzxnzmd3ulqcdoty5kjaqmwydr2uw4.py
# Source Nodes: [group_norm_59, mul_59, x_245, x_246, x_249, x_252], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
# group_norm_59 => add_207, mul_265
# mul_59 => mul_269
# x_245 => convolution_61
# x_246 => add_208, erf_29, mul_266, mul_267, mul_268
# x_249 => convolution_62
# x_252 => add_209
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwnokszr74qiho3la6rmv3yz2ivq5ldxddnfyuj6f56cgovuhcxu.py
# Source Nodes: [group_norm_59, mul_59, x_245, x_246, x_249, x_252, x_255], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
# group_norm_59 => add_207, mul_265
# mul_59 => mul_269
# x_245 => convolution_61
# x_246 => add_208, erf_29, mul_266, mul_267, mul_268
# x_249 => convolution_62
# x_252 => add_209
# x_255 => convolution_63
triton_poi_fused_add_convolution_gelu_mul_native_group_norm_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_native_group_norm_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (3456*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmmjglss5shspbfbc35z4sdiolmfmnewtbgynb2sdzj4ied5qap.py
# Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
# group_norm_60 => var_mean_60
triton_red_fused_native_group_norm_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 37632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (((r2 + (7527*x0)) // 49) % 768), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
    tl.store(out_ptr2 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mr/cmra4ugz7yroepcmsxfvepaocof7kwkbgeoig25d4qxbsfpt7mev.py
# Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
# group_norm_60 => var_mean_60
triton_per_fused_native_group_norm_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (5*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (5*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3ydxqn6o64vf4k7fkpqoeuimdmbwjmzpbgrn4fzccesvzd45tg.py
# Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
# group_norm_60 => add_211, mul_271
triton_poi_fused_native_group_norm_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_41', 'mutated_arg_names': []},
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
    x2 = (xindex // 37632)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 37632.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cm/ccmh4ngzqopsxt2j5eekzgqx7gtauimqvprjgugon22ryhbhufut.py
# Source Nodes: [group_norm_59, mul_59, mul_60, sub_30, x_245, x_246, x_249, x_252, x_255, x_256, y_30], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
# group_norm_59 => add_207, mul_265
# mul_59 => mul_269
# mul_60 => mul_272
# sub_30 => sub_91
# x_245 => convolution_61
# x_246 => add_208, erf_29, mul_266, mul_267, mul_268
# x_249 => convolution_62
# x_252 => add_209
# x_255 => convolution_63
# x_256 => add_212
# y_30 => avg_pool2d_30
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x4 = xindex
    x6 = (xindex // 49) % 768
    tmp90 = tl.load(in_out_ptr0 + (x4), None)
    tmp91 = tl.load(in_ptr1 + (x6), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (x4), None)
    tmp95 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 7, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-8) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-7) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-6) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (6 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (7 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (8 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp92 = tmp90 + tmp91
    tmp94 = tmp89 - tmp93
    tmp96 = tmp94 * tmp95
    tmp97 = tmp92 + tmp96
    tl.store(in_out_ptr0 + (x4), tmp97, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwboslp6ifrotxihnkynv5wr6gerv6s4a32o5fj4fe5svyzeizsv.py
# Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
# group_norm_61 => var_mean_61
triton_red_fused_native_group_norm_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 37632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cyw34nkwvznlnuoc62cn2erneuakqkh7lkdc4wp3rjxw3bunuitf.py
# Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
# group_norm_61 => add_214, mul_274
triton_poi_fused_native_group_norm_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 768)
    y0 = yindex % 768
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 37632.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (768*x2) + (37632*y1)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvdyspa7oibwnc25b6xzbci6oab6jhuonp3avctbrxkklt4l6y4.py
# Source Nodes: [group_norm_61, x_257, x_258], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
# group_norm_61 => add_214, mul_274
# x_257 => convolution_64
# x_258 => add_215, erf_30, mul_275, mul_276, mul_277
triton_poi_fused_convolution_gelu_native_group_norm_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_native_group_norm_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3072
    y1 = (yindex // 3072)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + (y0 + (3072*x2) + (150528*y1)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5ysposz2f5lrysqwfx4mi4xeed7zyrca7osjqhzcn5oqt63ggsh.py
# Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
# group_norm_62 => var_mean_62
triton_red_fused_native_group_norm_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 37632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (((r2 + (7527*x0)) // 49) % 768), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (((r2 + (7527*x0)) // 49) % 768), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 + tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = 1.0
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp19 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_combine(
            tmp21_mean, tmp21_m2, tmp21_weight,
            tmp18, tmp19, tmp20
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
    tl.store(out_ptr2 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdry3koxqxi44ukyfzb4x5dagfswfrrhqvdu53m7djugutq6qn3h.py
# Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
# group_norm_62 => add_218, mul_280
triton_poi_fused_native_group_norm_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_47', 'mutated_arg_names': []},
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
    x2 = (xindex // 37632)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 37632.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch2diziqfzy5lpz5rys3rje7v6yf6fdiutf4lye5dvxagmhircje.py
# Source Nodes: [group_norm_61, mul_61, mul_62, sub_31, x_257, x_258, x_261, x_263, x_264, y_31], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
# group_norm_61 => add_214, mul_274
# mul_61 => mul_278
# mul_62 => mul_281
# sub_31 => sub_94
# x_257 => convolution_64
# x_258 => add_215, erf_30, mul_275, mul_276, mul_277
# x_261 => convolution_65
# x_263 => add_216
# x_264 => add_219
# y_31 => avg_pool2d_31
triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x4 = xindex
    x6 = (xindex // 49) % 768
    tmp90 = tl.load(in_out_ptr0 + (x4), None)
    tmp91 = tl.load(in_ptr1 + (x4), None)
    tmp92 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr0 + (x4), None)
    tmp99 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 7, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-8) + x4), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-7) + x4), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-6) + x4), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + x4), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x4), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (6 + x4), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (7 + x4), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (8 + x4), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = 1.0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp10, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp70, tmp71)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp27, tmp70, tmp71)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp36, tmp70, tmp71)
    tmp78 = tmp77 + tmp76
    tmp79 = tl.where(tmp41, tmp70, tmp71)
    tmp80 = tmp79 + tmp78
    tmp81 = tl.where(tmp46, tmp70, tmp71)
    tmp82 = tmp81 + tmp80
    tmp83 = tl.where(tmp55, tmp70, tmp71)
    tmp84 = tmp83 + tmp82
    tmp85 = tl.where(tmp60, tmp70, tmp71)
    tmp86 = tmp85 + tmp84
    tmp87 = tl.where(tmp65, tmp70, tmp71)
    tmp88 = tmp87 + tmp86
    tmp89 = tmp69 / tmp88
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tmp90 + tmp95
    tmp98 = tmp89 - tmp97
    tmp100 = tmp98 * tmp99
    tmp101 = tmp96 + tmp100
    tl.store(in_out_ptr0 + (x4), tmp101, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ny/cnyazzv44p6o4vuofkhwisk6rxhi46g4p4ka7fsdtekcsmhumvcw.py
# Source Nodes: [group_norm_71, mul_71, x_297, x_298, x_301, x_306, x_307], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.native_group_norm]
# group_norm_71 => add_249, mul_319
# mul_71 => mul_323
# x_297 => convolution_74
# x_298 => add_250, erf_35, mul_320, mul_321, mul_322
# x_301 => convolution_75
# x_306 => add_251
# x_307 => mean
triton_per_fused_add_convolution_gelu_mean_mul_native_group_norm_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_gelu_mean_mul_native_group_norm_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (49*x3)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cyp3dd57e2pqpxr6ifcnmjfkpvcpcrjdw5wk6x4khhywmeoljels.py
# Source Nodes: [x_311], Original ATen: [aten.native_layer_norm]
# x_311 => add_252, add_253, mul_324, mul_325, rsqrt_72, sub_108, var_mean_72
triton_per_fused_native_layer_norm_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 768.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1 = args
    args.clear()
    assert_size_stride(arg0_1, (96, ), (1, ))
    assert_size_stride(arg1_1, (96, ), (1, ))
    assert_size_stride(arg2_1, (96, ), (1, ))
    assert_size_stride(arg3_1, (96, ), (1, ))
    assert_size_stride(arg4_1, (96, ), (1, ))
    assert_size_stride(arg5_1, (96, ), (1, ))
    assert_size_stride(arg6_1, (96, ), (1, ))
    assert_size_stride(arg7_1, (96, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (96, ), (1, ))
    assert_size_stride(arg12_1, (96, ), (1, ))
    assert_size_stride(arg13_1, (96, ), (1, ))
    assert_size_stride(arg14_1, (96, ), (1, ))
    assert_size_stride(arg15_1, (96, ), (1, ))
    assert_size_stride(arg16_1, (96, ), (1, ))
    assert_size_stride(arg17_1, (96, ), (1, ))
    assert_size_stride(arg18_1, (96, ), (1, ))
    assert_size_stride(arg19_1, (96, ), (1, ))
    assert_size_stride(arg20_1, (96, ), (1, ))
    assert_size_stride(arg21_1, (96, ), (1, ))
    assert_size_stride(arg22_1, (96, ), (1, ))
    assert_size_stride(arg23_1, (96, ), (1, ))
    assert_size_stride(arg24_1, (96, ), (1, ))
    assert_size_stride(arg25_1, (96, ), (1, ))
    assert_size_stride(arg26_1, (96, ), (1, ))
    assert_size_stride(arg27_1, (96, ), (1, ))
    assert_size_stride(arg28_1, (96, ), (1, ))
    assert_size_stride(arg29_1, (96, ), (1, ))
    assert_size_stride(arg30_1, (96, ), (1, ))
    assert_size_stride(arg31_1, (96, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (96, ), (1, ))
    assert_size_stride(arg36_1, (192, ), (1, ))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (192, ), (1, ))
    assert_size_stride(arg41_1, (192, ), (1, ))
    assert_size_stride(arg42_1, (192, ), (1, ))
    assert_size_stride(arg43_1, (192, ), (1, ))
    assert_size_stride(arg44_1, (192, ), (1, ))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (192, ), (1, ))
    assert_size_stride(arg47_1, (192, ), (1, ))
    assert_size_stride(arg48_1, (192, ), (1, ))
    assert_size_stride(arg49_1, (192, ), (1, ))
    assert_size_stride(arg50_1, (192, ), (1, ))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (192, ), (1, ))
    assert_size_stride(arg53_1, (192, ), (1, ))
    assert_size_stride(arg54_1, (192, ), (1, ))
    assert_size_stride(arg55_1, (192, ), (1, ))
    assert_size_stride(arg56_1, (192, ), (1, ))
    assert_size_stride(arg57_1, (192, ), (1, ))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (192, ), (1, ))
    assert_size_stride(arg61_1, (192, ), (1, ))
    assert_size_stride(arg62_1, (192, ), (1, ))
    assert_size_stride(arg63_1, (192, ), (1, ))
    assert_size_stride(arg64_1, (192, ), (1, ))
    assert_size_stride(arg65_1, (192, ), (1, ))
    assert_size_stride(arg66_1, (192, ), (1, ))
    assert_size_stride(arg67_1, (192, ), (1, ))
    assert_size_stride(arg68_1, (192, ), (1, ))
    assert_size_stride(arg69_1, (192, ), (1, ))
    assert_size_stride(arg70_1, (192, ), (1, ))
    assert_size_stride(arg71_1, (192, ), (1, ))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, ), (1, ))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (384, ), (1, ))
    assert_size_stride(arg80_1, (384, ), (1, ))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (384, ), (1, ))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (384, ), (1, ))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (384, ), (1, ))
    assert_size_stride(arg97_1, (384, ), (1, ))
    assert_size_stride(arg98_1, (384, ), (1, ))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (384, ), (1, ))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (384, ), (1, ))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (384, ), (1, ))
    assert_size_stride(arg115_1, (384, ), (1, ))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (384, ), (1, ))
    assert_size_stride(arg119_1, (384, ), (1, ))
    assert_size_stride(arg120_1, (384, ), (1, ))
    assert_size_stride(arg121_1, (384, ), (1, ))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (384, ), (1, ))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (384, ), (1, ))
    assert_size_stride(arg131_1, (384, ), (1, ))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (384, ), (1, ))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (384, ), (1, ))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (384, ), (1, ))
    assert_size_stride(arg143_1, (384, ), (1, ))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, ), (1, ))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (384, ), (1, ))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (384, ), (1, ))
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (384, ), (1, ))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (384, ), (1, ))
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (384, ), (1, ))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (384, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (768, ), (1, ))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (768, ), (1, ))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (96, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg219_1, (96, ), (1, ))
    assert_size_stride(arg220_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg223_1, (96, ), (1, ))
    assert_size_stride(arg224_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg227_1, (96, ), (1, ))
    assert_size_stride(arg228_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg231_1, (96, ), (1, ))
    assert_size_stride(arg232_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg235_1, (96, ), (1, ))
    assert_size_stride(arg236_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg237_1, (384, ), (1, ))
    assert_size_stride(arg238_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg239_1, (96, ), (1, ))
    assert_size_stride(arg240_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg241_1, (384, ), (1, ))
    assert_size_stride(arg242_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg243_1, (96, ), (1, ))
    assert_size_stride(arg244_1, (192, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg245_1, (192, ), (1, ))
    assert_size_stride(arg246_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg249_1, (192, ), (1, ))
    assert_size_stride(arg250_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg253_1, (192, ), (1, ))
    assert_size_stride(arg254_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg255_1, (768, ), (1, ))
    assert_size_stride(arg256_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg257_1, (192, ), (1, ))
    assert_size_stride(arg258_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg261_1, (192, ), (1, ))
    assert_size_stride(arg262_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg263_1, (768, ), (1, ))
    assert_size_stride(arg264_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg265_1, (192, ), (1, ))
    assert_size_stride(arg266_1, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg269_1, (192, ), (1, ))
    assert_size_stride(arg270_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg273_1, (1536, ), (1, ))
    assert_size_stride(arg274_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg275_1, (384, ), (1, ))
    assert_size_stride(arg276_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg277_1, (1536, ), (1, ))
    assert_size_stride(arg278_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg281_1, (1536, ), (1, ))
    assert_size_stride(arg282_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg283_1, (384, ), (1, ))
    assert_size_stride(arg284_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg285_1, (1536, ), (1, ))
    assert_size_stride(arg286_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg287_1, (384, ), (1, ))
    assert_size_stride(arg288_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg289_1, (1536, ), (1, ))
    assert_size_stride(arg290_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg291_1, (384, ), (1, ))
    assert_size_stride(arg292_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg293_1, (1536, ), (1, ))
    assert_size_stride(arg294_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg295_1, (384, ), (1, ))
    assert_size_stride(arg296_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg297_1, (1536, ), (1, ))
    assert_size_stride(arg298_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg299_1, (384, ), (1, ))
    assert_size_stride(arg300_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg301_1, (1536, ), (1, ))
    assert_size_stride(arg302_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg303_1, (384, ), (1, ))
    assert_size_stride(arg304_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg305_1, (1536, ), (1, ))
    assert_size_stride(arg306_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg307_1, (384, ), (1, ))
    assert_size_stride(arg308_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg309_1, (1536, ), (1, ))
    assert_size_stride(arg310_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg311_1, (384, ), (1, ))
    assert_size_stride(arg312_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg313_1, (1536, ), (1, ))
    assert_size_stride(arg314_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg315_1, (384, ), (1, ))
    assert_size_stride(arg316_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg317_1, (1536, ), (1, ))
    assert_size_stride(arg318_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg319_1, (384, ), (1, ))
    assert_size_stride(arg320_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg321_1, (1536, ), (1, ))
    assert_size_stride(arg322_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg323_1, (384, ), (1, ))
    assert_size_stride(arg324_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg325_1, (1536, ), (1, ))
    assert_size_stride(arg326_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg327_1, (384, ), (1, ))
    assert_size_stride(arg328_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg329_1, (1536, ), (1, ))
    assert_size_stride(arg330_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg331_1, (384, ), (1, ))
    assert_size_stride(arg332_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg333_1, (1536, ), (1, ))
    assert_size_stride(arg334_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg335_1, (384, ), (1, ))
    assert_size_stride(arg336_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg337_1, (1536, ), (1, ))
    assert_size_stride(arg338_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg339_1, (384, ), (1, ))
    assert_size_stride(arg340_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg341_1, (1536, ), (1, ))
    assert_size_stride(arg342_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg343_1, (384, ), (1, ))
    assert_size_stride(arg344_1, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg345_1, (768, ), (1, ))
    assert_size_stride(arg346_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg347_1, (3072, ), (1, ))
    assert_size_stride(arg348_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg349_1, (768, ), (1, ))
    assert_size_stride(arg350_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg351_1, (3072, ), (1, ))
    assert_size_stride(arg352_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg353_1, (768, ), (1, ))
    assert_size_stride(arg354_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg355_1, (3072, ), (1, ))
    assert_size_stride(arg356_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg357_1, (768, ), (1, ))
    assert_size_stride(arg358_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg359_1, (3072, ), (1, ))
    assert_size_stride(arg360_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg361_1, (768, ), (1, ))
    assert_size_stride(arg362_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg363_1, (3072, ), (1, ))
    assert_size_stride(arg364_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg365_1, (768, ), (1, ))
    assert_size_stride(arg366_1, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg367_1, (3072, ), (1, ))
    assert_size_stride(arg368_1, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(arg369_1, (768, ), (1, ))
    assert_size_stride(arg370_1, (1000, 768), (768, 1))
    assert_size_stride(arg371_1, (1000, ), (1, ))
    assert_size_stride(arg372_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg372_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg372_1
        buf1 = empty_strided((96, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg218_1, buf1, 288, 49, grid=grid(288, 49), stream=stream0)
        del arg218_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del buf1
        buf3 = empty_strided((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 1, 1, 1, 37), (37, 296, 296, 296, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_2.run(buf2, arg219_1, buf3, buf4, buf5, 296, 8137, grid=grid(296), stream=stream0)
        buf6 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf3, buf4, buf5, buf6, buf7, 8, 37, grid=grid(8), stream=stream0)
        buf9 = empty((8, 96, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_4.run(buf2, arg219_1, buf6, buf7, arg0_1, arg1_1, buf9, 2408448, grid=grid(2408448), stream=stream0)
        del arg0_1
        del arg1_1
        buf10 = empty((8, 96, 56, 56), device='cuda', dtype=torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [mul, sub, x, x_4, y], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.mul, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_mul_sub_5.run(buf11, buf9, buf2, arg219_1, arg2_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg219_1
        del arg2_1
        del buf2
        buf12 = buf5; del buf5  # reuse
        buf13 = buf4; del buf4  # reuse
        buf14 = buf3; del buf3  # reuse
        # Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_6.run(buf11, buf12, buf13, buf14, 296, 8137, grid=grid(296), stream=stream0)
        buf15 = buf7; del buf7  # reuse
        buf16 = buf6; del buf6  # reuse
        # Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf12, buf13, buf14, buf15, buf16, 8, 37, grid=grid(8), stream=stream0)
        buf18 = reinterpret_tensor(buf9, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf9  # reuse
        # Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_7.run(buf11, buf15, buf16, arg3_1, arg4_1, buf18, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg3_1
        del arg4_1
        # Source Nodes: [group_norm_1, x_5], Original ATen: [aten.convolution, aten.native_group_norm]
        buf19 = extern_kernels.convolution(buf18, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        del arg220_1
        buf20 = empty_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_1, x_5, x_6], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_8.run(buf19, arg221_1, buf20, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del arg221_1
        del buf19
        # Source Nodes: [group_norm_1, x_5, x_6, x_9], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf21 = extern_kernels.convolution(buf20, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg222_1
        buf22 = buf14; del buf14  # reuse
        buf23 = buf13; del buf13  # reuse
        buf24 = buf12; del buf12  # reuse
        # Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_9.run(buf11, buf21, arg223_1, arg5_1, buf22, buf23, buf24, 296, 8137, grid=grid(296), stream=stream0)
        buf25 = buf16; del buf16  # reuse
        buf26 = buf15; del buf15  # reuse
        # Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf22, buf23, buf24, buf25, buf26, 8, 37, grid=grid(8), stream=stream0)
        buf28 = reinterpret_tensor(buf18, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf18  # reuse
        # Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_10.run(buf11, buf21, arg223_1, arg5_1, buf25, buf26, arg6_1, arg7_1, buf28, 2408448, grid=grid(2408448), stream=stream0)
        del arg6_1
        del arg7_1
        buf30 = buf11; del buf11  # reuse
        # Source Nodes: [group_norm_1, mul_1, mul_2, sub_1, x_11, x_12, x_5, x_6, x_9, y_1], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_11.run(buf30, buf28, buf21, arg223_1, arg5_1, arg8_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg223_1
        del arg5_1
        del arg8_1
        del buf21
        buf31 = buf24; del buf24  # reuse
        buf32 = buf23; del buf23  # reuse
        buf33 = buf22; del buf22  # reuse
        # Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_6.run(buf30, buf31, buf32, buf33, 296, 8137, grid=grid(296), stream=stream0)
        buf34 = buf26; del buf26  # reuse
        buf35 = buf25; del buf25  # reuse
        # Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf31, buf32, buf33, buf34, buf35, 8, 37, grid=grid(8), stream=stream0)
        buf37 = reinterpret_tensor(buf28, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf28  # reuse
        # Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_7.run(buf30, buf34, buf35, arg9_1, arg10_1, buf37, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg10_1
        del arg9_1
        # Source Nodes: [group_norm_3, x_13], Original ATen: [aten.convolution, aten.native_group_norm]
        buf38 = extern_kernels.convolution(buf37, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        del arg224_1
        buf39 = buf20; del buf20  # reuse
        # Source Nodes: [group_norm_3, x_13, x_14], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_8.run(buf38, arg225_1, buf39, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del arg225_1
        del buf38
        # Source Nodes: [group_norm_3, x_13, x_14, x_17], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf40 = extern_kernels.convolution(buf39, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg226_1
        buf41 = buf33; del buf33  # reuse
        buf42 = buf32; del buf32  # reuse
        buf43 = buf31; del buf31  # reuse
        # Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_9.run(buf30, buf40, arg227_1, arg11_1, buf41, buf42, buf43, 296, 8137, grid=grid(296), stream=stream0)
        buf44 = buf35; del buf35  # reuse
        buf45 = buf34; del buf34  # reuse
        # Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf41, buf42, buf43, buf44, buf45, 8, 37, grid=grid(8), stream=stream0)
        buf47 = reinterpret_tensor(buf37, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf37  # reuse
        # Source Nodes: [group_norm_4], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_10.run(buf30, buf40, arg227_1, arg11_1, buf44, buf45, arg12_1, arg13_1, buf47, 2408448, grid=grid(2408448), stream=stream0)
        del arg12_1
        del arg13_1
        buf49 = buf30; del buf30  # reuse
        # Source Nodes: [group_norm_3, mul_3, mul_4, sub_2, x_13, x_14, x_17, x_19, x_20, y_2], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_11.run(buf49, buf47, buf40, arg227_1, arg11_1, arg14_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg11_1
        del arg14_1
        del arg227_1
        del buf40
        buf50 = buf43; del buf43  # reuse
        buf51 = buf42; del buf42  # reuse
        buf52 = buf41; del buf41  # reuse
        # Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_6.run(buf49, buf50, buf51, buf52, 296, 8137, grid=grid(296), stream=stream0)
        buf53 = buf45; del buf45  # reuse
        buf54 = buf44; del buf44  # reuse
        # Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf50, buf51, buf52, buf53, buf54, 8, 37, grid=grid(8), stream=stream0)
        buf56 = reinterpret_tensor(buf47, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf47  # reuse
        # Source Nodes: [group_norm_5], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_7.run(buf49, buf53, buf54, arg15_1, arg16_1, buf56, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg15_1
        del arg16_1
        # Source Nodes: [group_norm_5, x_21], Original ATen: [aten.convolution, aten.native_group_norm]
        buf57 = extern_kernels.convolution(buf56, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        del arg228_1
        buf58 = buf39; del buf39  # reuse
        # Source Nodes: [group_norm_5, x_21, x_22], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_8.run(buf57, arg229_1, buf58, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del arg229_1
        del buf57
        # Source Nodes: [group_norm_5, x_21, x_22, x_25], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf59 = extern_kernels.convolution(buf58, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg230_1
        buf60 = buf52; del buf52  # reuse
        buf61 = buf51; del buf51  # reuse
        buf62 = buf50; del buf50  # reuse
        # Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_9.run(buf49, buf59, arg231_1, arg17_1, buf60, buf61, buf62, 296, 8137, grid=grid(296), stream=stream0)
        buf63 = buf54; del buf54  # reuse
        buf64 = buf53; del buf53  # reuse
        # Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf60, buf61, buf62, buf63, buf64, 8, 37, grid=grid(8), stream=stream0)
        buf66 = reinterpret_tensor(buf56, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf56  # reuse
        # Source Nodes: [group_norm_6], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_10.run(buf49, buf59, arg231_1, arg17_1, buf63, buf64, arg18_1, arg19_1, buf66, 2408448, grid=grid(2408448), stream=stream0)
        del arg18_1
        del arg19_1
        buf68 = buf49; del buf49  # reuse
        # Source Nodes: [group_norm_5, mul_5, mul_6, sub_3, x_21, x_22, x_25, x_27, x_28, y_3], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_11.run(buf68, buf66, buf59, arg231_1, arg17_1, arg20_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg17_1
        del arg20_1
        del arg231_1
        del buf59
        buf69 = buf62; del buf62  # reuse
        buf70 = buf61; del buf61  # reuse
        buf71 = buf60; del buf60  # reuse
        # Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_6.run(buf68, buf69, buf70, buf71, 296, 8137, grid=grid(296), stream=stream0)
        buf72 = buf64; del buf64  # reuse
        buf73 = buf63; del buf63  # reuse
        # Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf69, buf70, buf71, buf72, buf73, 8, 37, grid=grid(8), stream=stream0)
        buf75 = reinterpret_tensor(buf66, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf66  # reuse
        # Source Nodes: [group_norm_7], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_7.run(buf68, buf72, buf73, arg21_1, arg22_1, buf75, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg21_1
        del arg22_1
        # Source Nodes: [group_norm_7, x_29], Original ATen: [aten.convolution, aten.native_group_norm]
        buf76 = extern_kernels.convolution(buf75, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        del arg232_1
        buf77 = buf58; del buf58  # reuse
        # Source Nodes: [group_norm_7, x_29, x_30], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_8.run(buf76, arg233_1, buf77, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del arg233_1
        del buf76
        # Source Nodes: [group_norm_7, x_29, x_30, x_33], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf78 = extern_kernels.convolution(buf77, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg234_1
        buf79 = buf71; del buf71  # reuse
        buf80 = buf70; del buf70  # reuse
        buf81 = buf69; del buf69  # reuse
        # Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_9.run(buf68, buf78, arg235_1, arg23_1, buf79, buf80, buf81, 296, 8137, grid=grid(296), stream=stream0)
        buf82 = buf73; del buf73  # reuse
        buf83 = buf72; del buf72  # reuse
        # Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf79, buf80, buf81, buf82, buf83, 8, 37, grid=grid(8), stream=stream0)
        buf85 = reinterpret_tensor(buf75, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf75  # reuse
        # Source Nodes: [group_norm_8], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_10.run(buf68, buf78, arg235_1, arg23_1, buf82, buf83, arg24_1, arg25_1, buf85, 2408448, grid=grid(2408448), stream=stream0)
        del arg24_1
        del arg25_1
        buf87 = buf68; del buf68  # reuse
        # Source Nodes: [group_norm_7, mul_7, mul_8, sub_4, x_29, x_30, x_33, x_35, x_36, y_4], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_11.run(buf87, buf85, buf78, arg235_1, arg23_1, arg26_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg235_1
        del arg23_1
        del arg26_1
        buf88 = buf81; del buf81  # reuse
        buf89 = buf80; del buf80  # reuse
        buf90 = buf79; del buf79  # reuse
        # Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_6.run(buf87, buf88, buf89, buf90, 296, 8137, grid=grid(296), stream=stream0)
        buf91 = buf83; del buf83  # reuse
        buf92 = buf82; del buf82  # reuse
        # Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf88, buf89, buf90, buf91, buf92, 8, 37, grid=grid(8), stream=stream0)
        buf94 = reinterpret_tensor(buf85, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf85  # reuse
        # Source Nodes: [group_norm_9], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_7.run(buf87, buf91, buf92, arg27_1, arg28_1, buf94, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg27_1
        del arg28_1
        # Source Nodes: [group_norm_9, x_37], Original ATen: [aten.convolution, aten.native_group_norm]
        buf95 = extern_kernels.convolution(buf94, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        del arg236_1
        buf96 = buf77; del buf77  # reuse
        # Source Nodes: [group_norm_9, x_37, x_38], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_8.run(buf95, arg237_1, buf96, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del arg237_1
        del buf95
        # Source Nodes: [group_norm_9, x_37, x_38, x_41], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf97 = extern_kernels.convolution(buf96, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg238_1
        buf98 = buf90; del buf90  # reuse
        buf99 = buf89; del buf89  # reuse
        buf100 = buf88; del buf88  # reuse
        # Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_9.run(buf87, buf97, arg239_1, arg29_1, buf98, buf99, buf100, 296, 8137, grid=grid(296), stream=stream0)
        buf101 = buf92; del buf92  # reuse
        buf102 = buf91; del buf91  # reuse
        # Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf98, buf99, buf100, buf101, buf102, 8, 37, grid=grid(8), stream=stream0)
        buf104 = reinterpret_tensor(buf94, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf94  # reuse
        # Source Nodes: [group_norm_10], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_10.run(buf87, buf97, arg239_1, arg29_1, buf101, buf102, arg30_1, arg31_1, buf104, 2408448, grid=grid(2408448), stream=stream0)
        del arg30_1
        del arg31_1
        buf105 = buf78; del buf78  # reuse
        buf106 = buf105; del buf105  # reuse
        # Source Nodes: [group_norm_9, mul_10, mul_9, sub_5, x_37, x_38, x_41, x_43, x_44, y_5], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_12.run(buf106, buf104, buf87, buf97, arg239_1, arg29_1, arg32_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg239_1
        del arg29_1
        del arg32_1
        del buf104
        del buf87
        buf107 = buf99; del buf99  # reuse
        buf108 = buf98; del buf98  # reuse
        buf109 = buf100; del buf100  # reuse
        # Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_6.run(buf106, buf107, buf108, buf109, 296, 8137, grid=grid(296), stream=stream0)
        buf110 = buf102; del buf102  # reuse
        buf111 = buf101; del buf101  # reuse
        # Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_3.run(buf107, buf108, buf109, buf110, buf111, 8, 37, grid=grid(8), stream=stream0)
        del buf107
        del buf108
        del buf109
        buf113 = reinterpret_tensor(buf97, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf97  # reuse
        # Source Nodes: [group_norm_11], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_7.run(buf106, buf110, buf111, arg33_1, arg34_1, buf113, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg33_1
        del arg34_1
        # Source Nodes: [group_norm_11, x_45], Original ATen: [aten.convolution, aten.native_group_norm]
        buf114 = extern_kernels.convolution(buf113, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 384, 56, 56), (1204224, 3136, 56, 1))
        del arg240_1
        buf115 = buf96; del buf96  # reuse
        # Source Nodes: [group_norm_11, x_45, x_46], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_8.run(buf114, arg241_1, buf115, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del arg241_1
        del buf114
        # Source Nodes: [group_norm_11, x_45, x_46, x_49], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf116 = extern_kernels.convolution(buf115, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg242_1
        del buf115
        buf117 = buf113; del buf113  # reuse
        # Source Nodes: [group_norm_11, mul_11, x_45, x_46, x_49, x_52], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_13.run(buf106, buf116, arg243_1, arg35_1, buf117, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg243_1
        del arg35_1
        del buf106
        del buf116
        buf118 = empty_strided((192, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_11, mul_11, x_45, x_46, x_49, x_52, x_55], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_14.run(arg244_1, buf118, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg244_1
        # Source Nodes: [group_norm_11, mul_11, x_45, x_46, x_49, x_52, x_55], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        buf119 = extern_kernels.convolution(buf117, buf118, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 192, 28, 28), (150528, 784, 28, 1))
        del buf118
        buf120 = empty_strided((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), device='cuda', dtype=torch.float32)
        buf121 = empty_strided((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), device='cuda', dtype=torch.float32)
        buf122 = empty_strided((8, 1, 1, 1, 19), (19, 152, 152, 152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_15.run(buf119, arg245_1, buf120, buf121, buf122, 152, 7923, grid=grid(152), stream=stream0)
        buf123 = buf111; del buf111  # reuse
        buf124 = buf110; del buf110  # reuse
        # Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf120, buf121, buf122, buf123, buf124, 8, 19, grid=grid(8), stream=stream0)
        buf126 = reinterpret_tensor(buf0, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf0  # reuse
        # Source Nodes: [group_norm_12], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_17.run(buf119, arg245_1, buf123, buf124, arg36_1, arg37_1, buf126, 1204224, grid=grid(1204224), stream=stream0)
        del arg36_1
        del arg37_1
        buf128 = buf119; del buf119  # reuse
        # Source Nodes: [group_norm_11, mul_11, mul_12, sub_6, x_45, x_46, x_49, x_52, x_55, x_56, y_6], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_18.run(buf128, buf126, arg245_1, arg38_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg245_1
        del arg38_1
        buf129 = buf122; del buf122  # reuse
        buf130 = buf121; del buf121  # reuse
        buf131 = buf120; del buf120  # reuse
        # Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_19.run(buf128, buf129, buf130, buf131, 152, 7923, grid=grid(152), stream=stream0)
        buf132 = buf124; del buf124  # reuse
        buf133 = buf123; del buf123  # reuse
        # Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf129, buf130, buf131, buf132, buf133, 8, 19, grid=grid(8), stream=stream0)
        buf135 = reinterpret_tensor(buf126, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf126  # reuse
        # Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_20.run(buf128, buf132, buf133, arg39_1, arg40_1, buf135, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg39_1
        del arg40_1
        # Source Nodes: [group_norm_13, x_57], Original ATen: [aten.convolution, aten.native_group_norm]
        buf136 = extern_kernels.convolution(buf135, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg246_1
        buf137 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_13, x_57, x_58], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_21.run(buf136, arg247_1, buf137, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg247_1
        del buf136
        # Source Nodes: [group_norm_13, x_57, x_58, x_61], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf138 = extern_kernels.convolution(buf137, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg248_1
        buf139 = buf131; del buf131  # reuse
        buf140 = buf130; del buf130  # reuse
        buf141 = buf129; del buf129  # reuse
        # Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_22.run(buf128, buf138, arg249_1, arg41_1, buf139, buf140, buf141, 152, 7923, grid=grid(152), stream=stream0)
        buf142 = buf133; del buf133  # reuse
        buf143 = buf132; del buf132  # reuse
        # Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf139, buf140, buf141, buf142, buf143, 8, 19, grid=grid(8), stream=stream0)
        buf145 = reinterpret_tensor(buf135, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf135  # reuse
        # Source Nodes: [group_norm_14], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_23.run(buf128, buf138, arg249_1, arg41_1, buf142, buf143, arg42_1, arg43_1, buf145, 1204224, grid=grid(1204224), stream=stream0)
        del arg42_1
        del arg43_1
        buf147 = buf128; del buf128  # reuse
        # Source Nodes: [group_norm_13, mul_13, mul_14, sub_7, x_57, x_58, x_61, x_63, x_64, y_7], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_24.run(buf147, buf145, buf138, arg249_1, arg41_1, arg44_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg249_1
        del arg41_1
        del arg44_1
        del buf138
        buf148 = buf141; del buf141  # reuse
        buf149 = buf140; del buf140  # reuse
        buf150 = buf139; del buf139  # reuse
        # Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_19.run(buf147, buf148, buf149, buf150, 152, 7923, grid=grid(152), stream=stream0)
        buf151 = buf143; del buf143  # reuse
        buf152 = buf142; del buf142  # reuse
        # Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf148, buf149, buf150, buf151, buf152, 8, 19, grid=grid(8), stream=stream0)
        buf154 = reinterpret_tensor(buf145, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf145  # reuse
        # Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_20.run(buf147, buf151, buf152, arg45_1, arg46_1, buf154, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg45_1
        del arg46_1
        # Source Nodes: [group_norm_15, x_65], Original ATen: [aten.convolution, aten.native_group_norm]
        buf155 = extern_kernels.convolution(buf154, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg250_1
        buf156 = buf137; del buf137  # reuse
        # Source Nodes: [group_norm_15, x_65, x_66], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_21.run(buf155, arg251_1, buf156, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg251_1
        del buf155
        # Source Nodes: [group_norm_15, x_65, x_66, x_69], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf157 = extern_kernels.convolution(buf156, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg252_1
        buf158 = buf150; del buf150  # reuse
        buf159 = buf149; del buf149  # reuse
        buf160 = buf148; del buf148  # reuse
        # Source Nodes: [group_norm_16], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_22.run(buf147, buf157, arg253_1, arg47_1, buf158, buf159, buf160, 152, 7923, grid=grid(152), stream=stream0)
        buf161 = buf152; del buf152  # reuse
        buf162 = buf151; del buf151  # reuse
        # Source Nodes: [group_norm_16], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf158, buf159, buf160, buf161, buf162, 8, 19, grid=grid(8), stream=stream0)
        buf164 = reinterpret_tensor(buf154, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf154  # reuse
        # Source Nodes: [group_norm_16], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_23.run(buf147, buf157, arg253_1, arg47_1, buf161, buf162, arg48_1, arg49_1, buf164, 1204224, grid=grid(1204224), stream=stream0)
        del arg48_1
        del arg49_1
        buf166 = buf147; del buf147  # reuse
        # Source Nodes: [group_norm_15, mul_15, mul_16, sub_8, x_65, x_66, x_69, x_71, x_72, y_8], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_24.run(buf166, buf164, buf157, arg253_1, arg47_1, arg50_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg253_1
        del arg47_1
        del arg50_1
        del buf157
        buf167 = buf160; del buf160  # reuse
        buf168 = buf159; del buf159  # reuse
        buf169 = buf158; del buf158  # reuse
        # Source Nodes: [group_norm_17], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_19.run(buf166, buf167, buf168, buf169, 152, 7923, grid=grid(152), stream=stream0)
        buf170 = buf162; del buf162  # reuse
        buf171 = buf161; del buf161  # reuse
        # Source Nodes: [group_norm_17], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf167, buf168, buf169, buf170, buf171, 8, 19, grid=grid(8), stream=stream0)
        buf173 = reinterpret_tensor(buf164, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf164  # reuse
        # Source Nodes: [group_norm_17], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_20.run(buf166, buf170, buf171, arg51_1, arg52_1, buf173, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg51_1
        del arg52_1
        # Source Nodes: [group_norm_17, x_73], Original ATen: [aten.convolution, aten.native_group_norm]
        buf174 = extern_kernels.convolution(buf173, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg254_1
        buf175 = buf156; del buf156  # reuse
        # Source Nodes: [group_norm_17, x_73, x_74], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_21.run(buf174, arg255_1, buf175, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg255_1
        del buf174
        # Source Nodes: [group_norm_17, x_73, x_74, x_77], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf176 = extern_kernels.convolution(buf175, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg256_1
        buf177 = buf169; del buf169  # reuse
        buf178 = buf168; del buf168  # reuse
        buf179 = buf167; del buf167  # reuse
        # Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_22.run(buf166, buf176, arg257_1, arg53_1, buf177, buf178, buf179, 152, 7923, grid=grid(152), stream=stream0)
        buf180 = buf171; del buf171  # reuse
        buf181 = buf170; del buf170  # reuse
        # Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf177, buf178, buf179, buf180, buf181, 8, 19, grid=grid(8), stream=stream0)
        buf183 = reinterpret_tensor(buf173, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf173  # reuse
        # Source Nodes: [group_norm_18], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_23.run(buf166, buf176, arg257_1, arg53_1, buf180, buf181, arg54_1, arg55_1, buf183, 1204224, grid=grid(1204224), stream=stream0)
        del arg54_1
        del arg55_1
        buf185 = buf166; del buf166  # reuse
        # Source Nodes: [group_norm_17, mul_17, mul_18, sub_9, x_73, x_74, x_77, x_79, x_80, y_9], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_24.run(buf185, buf183, buf176, arg257_1, arg53_1, arg56_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg257_1
        del arg53_1
        del arg56_1
        del buf176
        buf186 = buf179; del buf179  # reuse
        buf187 = buf178; del buf178  # reuse
        buf188 = buf177; del buf177  # reuse
        # Source Nodes: [group_norm_19], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_19.run(buf185, buf186, buf187, buf188, 152, 7923, grid=grid(152), stream=stream0)
        buf189 = buf181; del buf181  # reuse
        buf190 = buf180; del buf180  # reuse
        # Source Nodes: [group_norm_19], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf186, buf187, buf188, buf189, buf190, 8, 19, grid=grid(8), stream=stream0)
        buf192 = reinterpret_tensor(buf183, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf183  # reuse
        # Source Nodes: [group_norm_19], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_20.run(buf185, buf189, buf190, arg57_1, arg58_1, buf192, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg57_1
        del arg58_1
        # Source Nodes: [group_norm_19, x_81], Original ATen: [aten.convolution, aten.native_group_norm]
        buf193 = extern_kernels.convolution(buf192, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg258_1
        buf194 = buf175; del buf175  # reuse
        # Source Nodes: [group_norm_19, x_81, x_82], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_21.run(buf193, arg259_1, buf194, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg259_1
        del buf193
        # Source Nodes: [group_norm_19, x_81, x_82, x_85], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf195 = extern_kernels.convolution(buf194, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg260_1
        buf196 = buf188; del buf188  # reuse
        buf197 = buf187; del buf187  # reuse
        buf198 = buf186; del buf186  # reuse
        # Source Nodes: [group_norm_20], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_22.run(buf185, buf195, arg261_1, arg59_1, buf196, buf197, buf198, 152, 7923, grid=grid(152), stream=stream0)
        buf199 = buf190; del buf190  # reuse
        buf200 = buf189; del buf189  # reuse
        # Source Nodes: [group_norm_20], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf196, buf197, buf198, buf199, buf200, 8, 19, grid=grid(8), stream=stream0)
        buf202 = reinterpret_tensor(buf192, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf192  # reuse
        # Source Nodes: [group_norm_20], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_23.run(buf185, buf195, arg261_1, arg59_1, buf199, buf200, arg60_1, arg61_1, buf202, 1204224, grid=grid(1204224), stream=stream0)
        del arg60_1
        del arg61_1
        buf204 = buf185; del buf185  # reuse
        # Source Nodes: [group_norm_19, mul_19, mul_20, sub_10, x_81, x_82, x_85, x_87, x_88, y_10], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_24.run(buf204, buf202, buf195, arg261_1, arg59_1, arg62_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg261_1
        del arg59_1
        del arg62_1
        del buf195
        buf205 = buf198; del buf198  # reuse
        buf206 = buf197; del buf197  # reuse
        buf207 = buf196; del buf196  # reuse
        # Source Nodes: [group_norm_21], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_19.run(buf204, buf205, buf206, buf207, 152, 7923, grid=grid(152), stream=stream0)
        buf208 = buf200; del buf200  # reuse
        buf209 = buf199; del buf199  # reuse
        # Source Nodes: [group_norm_21], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf205, buf206, buf207, buf208, buf209, 8, 19, grid=grid(8), stream=stream0)
        buf211 = reinterpret_tensor(buf202, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf202  # reuse
        # Source Nodes: [group_norm_21], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_20.run(buf204, buf208, buf209, arg63_1, arg64_1, buf211, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg63_1
        del arg64_1
        # Source Nodes: [group_norm_21, x_89], Original ATen: [aten.convolution, aten.native_group_norm]
        buf212 = extern_kernels.convolution(buf211, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg262_1
        buf213 = buf194; del buf194  # reuse
        # Source Nodes: [group_norm_21, x_89, x_90], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_21.run(buf212, arg263_1, buf213, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg263_1
        del buf212
        # Source Nodes: [group_norm_21, x_89, x_90, x_93], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf214 = extern_kernels.convolution(buf213, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg264_1
        buf215 = buf207; del buf207  # reuse
        buf216 = buf206; del buf206  # reuse
        buf217 = buf205; del buf205  # reuse
        # Source Nodes: [group_norm_22], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_22.run(buf204, buf214, arg265_1, arg65_1, buf215, buf216, buf217, 152, 7923, grid=grid(152), stream=stream0)
        buf218 = buf209; del buf209  # reuse
        buf219 = buf208; del buf208  # reuse
        # Source Nodes: [group_norm_22], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf215, buf216, buf217, buf218, buf219, 8, 19, grid=grid(8), stream=stream0)
        buf221 = reinterpret_tensor(buf211, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf211  # reuse
        # Source Nodes: [group_norm_22], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_23.run(buf204, buf214, arg265_1, arg65_1, buf218, buf219, arg66_1, arg67_1, buf221, 1204224, grid=grid(1204224), stream=stream0)
        del arg66_1
        del arg67_1
        buf223 = buf204; del buf204  # reuse
        # Source Nodes: [group_norm_21, mul_21, mul_22, sub_11, x_89, x_90, x_93, x_95, x_96, y_11], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_24.run(buf223, buf221, buf214, arg265_1, arg65_1, arg68_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg265_1
        del arg65_1
        del arg68_1
        del buf214
        buf224 = buf217; del buf217  # reuse
        buf225 = buf216; del buf216  # reuse
        buf226 = buf215; del buf215  # reuse
        # Source Nodes: [group_norm_23], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_19.run(buf223, buf224, buf225, buf226, 152, 7923, grid=grid(152), stream=stream0)
        buf227 = buf219; del buf219  # reuse
        buf228 = buf218; del buf218  # reuse
        # Source Nodes: [group_norm_23], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_16.run(buf224, buf225, buf226, buf227, buf228, 8, 19, grid=grid(8), stream=stream0)
        del buf224
        del buf225
        del buf226
        buf230 = reinterpret_tensor(buf221, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf221  # reuse
        # Source Nodes: [group_norm_23], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_20.run(buf223, buf227, buf228, arg69_1, arg70_1, buf230, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg69_1
        del arg70_1
        # Source Nodes: [group_norm_23, x_97], Original ATen: [aten.convolution, aten.native_group_norm]
        buf231 = extern_kernels.convolution(buf230, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 768, 28, 28), (602112, 784, 28, 1))
        del arg266_1
        buf232 = buf213; del buf213  # reuse
        # Source Nodes: [group_norm_23, x_97, x_98], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_21.run(buf231, arg267_1, buf232, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg267_1
        del buf231
        # Source Nodes: [group_norm_23, x_101, x_97, x_98], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf233 = extern_kernels.convolution(buf232, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg268_1
        del buf232
        buf234 = buf230; del buf230  # reuse
        # Source Nodes: [group_norm_23, mul_23, x_101, x_104, x_97, x_98], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_25.run(buf223, buf233, arg269_1, arg71_1, buf234, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg269_1
        del arg71_1
        del buf223
        del buf233
        buf235 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_23, mul_23, x_101, x_104, x_107, x_97, x_98], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_26.run(arg270_1, buf235, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del arg270_1
        # Source Nodes: [group_norm_23, mul_23, x_101, x_104, x_107, x_97, x_98], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        buf236 = extern_kernels.convolution(buf234, buf235, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 384, 14, 14), (75264, 196, 14, 1))
        del buf235
        buf237 = empty_strided((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), device='cuda', dtype=torch.float32)
        buf238 = empty_strided((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), device='cuda', dtype=torch.float32)
        buf239 = empty_strided((8, 1, 1, 1, 10), (10, 80, 80, 80, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_27.run(buf236, arg271_1, buf237, buf238, buf239, 80, 7527, grid=grid(80), stream=stream0)
        buf240 = buf228; del buf228  # reuse
        buf241 = buf227; del buf227  # reuse
        # Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf237, buf238, buf239, buf240, buf241, 8, 10, grid=grid(8), stream=stream0)
        buf243 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_29.run(buf236, arg271_1, buf240, buf241, arg72_1, arg73_1, buf243, 602112, grid=grid(602112), stream=stream0)
        del arg72_1
        del arg73_1
        buf245 = buf236; del buf236  # reuse
        # Source Nodes: [group_norm_23, mul_23, mul_24, sub_12, x_101, x_104, x_107, x_108, x_97, x_98, y_12], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_30.run(buf245, buf243, arg271_1, arg74_1, 602112, grid=grid(602112), stream=stream0)
        del arg271_1
        del arg74_1
        buf246 = buf239; del buf239  # reuse
        buf247 = buf238; del buf238  # reuse
        buf248 = buf237; del buf237  # reuse
        # Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf245, buf246, buf247, buf248, 80, 7527, grid=grid(80), stream=stream0)
        buf249 = buf241; del buf241  # reuse
        buf250 = buf240; del buf240  # reuse
        # Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf246, buf247, buf248, buf249, buf250, 8, 10, grid=grid(8), stream=stream0)
        buf252 = reinterpret_tensor(buf243, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf243  # reuse
        # Source Nodes: [group_norm_25], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf245, buf249, buf250, arg75_1, arg76_1, buf252, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg75_1
        del arg76_1
        # Source Nodes: [group_norm_25, x_109], Original ATen: [aten.convolution, aten.native_group_norm]
        buf253 = extern_kernels.convolution(buf252, arg272_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg272_1
        buf254 = reinterpret_tensor(buf117, (8, 1536, 14, 14), (301056, 1, 21504, 1536), 0); del buf117  # reuse
        # Source Nodes: [group_norm_25, x_109, x_110], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf253, arg273_1, buf254, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg273_1
        del buf253
        # Source Nodes: [group_norm_25, x_109, x_110, x_113], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf255 = extern_kernels.convolution(buf254, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg274_1
        buf256 = buf248; del buf248  # reuse
        buf257 = buf247; del buf247  # reuse
        buf258 = buf246; del buf246  # reuse
        # Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf245, buf255, arg275_1, arg77_1, buf256, buf257, buf258, 80, 7527, grid=grid(80), stream=stream0)
        buf259 = buf250; del buf250  # reuse
        buf260 = buf249; del buf249  # reuse
        # Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf256, buf257, buf258, buf259, buf260, 8, 10, grid=grid(8), stream=stream0)
        buf262 = reinterpret_tensor(buf252, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf252  # reuse
        # Source Nodes: [group_norm_26], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf245, buf255, arg275_1, arg77_1, buf259, buf260, arg78_1, arg79_1, buf262, 602112, grid=grid(602112), stream=stream0)
        del arg78_1
        del arg79_1
        buf264 = buf245; del buf245  # reuse
        # Source Nodes: [group_norm_25, mul_25, mul_26, sub_13, x_109, x_110, x_113, x_115, x_116, y_13], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf264, buf262, buf255, arg275_1, arg77_1, arg80_1, 602112, grid=grid(602112), stream=stream0)
        del arg275_1
        del arg77_1
        del arg80_1
        del buf255
        buf265 = buf258; del buf258  # reuse
        buf266 = buf257; del buf257  # reuse
        buf267 = buf256; del buf256  # reuse
        # Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf264, buf265, buf266, buf267, 80, 7527, grid=grid(80), stream=stream0)
        buf268 = buf260; del buf260  # reuse
        buf269 = buf259; del buf259  # reuse
        # Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf265, buf266, buf267, buf268, buf269, 8, 10, grid=grid(8), stream=stream0)
        buf271 = reinterpret_tensor(buf262, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf262  # reuse
        # Source Nodes: [group_norm_27], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf264, buf268, buf269, arg81_1, arg82_1, buf271, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg81_1
        del arg82_1
        # Source Nodes: [group_norm_27, x_117], Original ATen: [aten.convolution, aten.native_group_norm]
        buf272 = extern_kernels.convolution(buf271, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg276_1
        buf273 = buf254; del buf254  # reuse
        # Source Nodes: [group_norm_27, x_117, x_118], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf272, arg277_1, buf273, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg277_1
        del buf272
        # Source Nodes: [group_norm_27, x_117, x_118, x_121], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf274 = extern_kernels.convolution(buf273, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg278_1
        buf275 = buf267; del buf267  # reuse
        buf276 = buf266; del buf266  # reuse
        buf277 = buf265; del buf265  # reuse
        # Source Nodes: [group_norm_28], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf264, buf274, arg279_1, arg83_1, buf275, buf276, buf277, 80, 7527, grid=grid(80), stream=stream0)
        buf278 = buf269; del buf269  # reuse
        buf279 = buf268; del buf268  # reuse
        # Source Nodes: [group_norm_28], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf275, buf276, buf277, buf278, buf279, 8, 10, grid=grid(8), stream=stream0)
        buf281 = reinterpret_tensor(buf271, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf271  # reuse
        # Source Nodes: [group_norm_28], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf264, buf274, arg279_1, arg83_1, buf278, buf279, arg84_1, arg85_1, buf281, 602112, grid=grid(602112), stream=stream0)
        del arg84_1
        del arg85_1
        buf283 = buf264; del buf264  # reuse
        # Source Nodes: [group_norm_27, mul_27, mul_28, sub_14, x_117, x_118, x_121, x_123, x_124, y_14], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf283, buf281, buf274, arg279_1, arg83_1, arg86_1, 602112, grid=grid(602112), stream=stream0)
        del arg279_1
        del arg83_1
        del arg86_1
        del buf274
        buf284 = buf277; del buf277  # reuse
        buf285 = buf276; del buf276  # reuse
        buf286 = buf275; del buf275  # reuse
        # Source Nodes: [group_norm_29], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf283, buf284, buf285, buf286, 80, 7527, grid=grid(80), stream=stream0)
        buf287 = buf279; del buf279  # reuse
        buf288 = buf278; del buf278  # reuse
        # Source Nodes: [group_norm_29], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf284, buf285, buf286, buf287, buf288, 8, 10, grid=grid(8), stream=stream0)
        buf290 = reinterpret_tensor(buf281, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf281  # reuse
        # Source Nodes: [group_norm_29], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf283, buf287, buf288, arg87_1, arg88_1, buf290, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg87_1
        del arg88_1
        # Source Nodes: [group_norm_29, x_125], Original ATen: [aten.convolution, aten.native_group_norm]
        buf291 = extern_kernels.convolution(buf290, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg280_1
        buf292 = buf273; del buf273  # reuse
        # Source Nodes: [group_norm_29, x_125, x_126], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf291, arg281_1, buf292, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg281_1
        del buf291
        # Source Nodes: [group_norm_29, x_125, x_126, x_129], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf293 = extern_kernels.convolution(buf292, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg282_1
        buf294 = buf286; del buf286  # reuse
        buf295 = buf285; del buf285  # reuse
        buf296 = buf284; del buf284  # reuse
        # Source Nodes: [group_norm_30], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf283, buf293, arg283_1, arg89_1, buf294, buf295, buf296, 80, 7527, grid=grid(80), stream=stream0)
        buf297 = buf288; del buf288  # reuse
        buf298 = buf287; del buf287  # reuse
        # Source Nodes: [group_norm_30], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf294, buf295, buf296, buf297, buf298, 8, 10, grid=grid(8), stream=stream0)
        buf300 = reinterpret_tensor(buf290, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf290  # reuse
        # Source Nodes: [group_norm_30], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf283, buf293, arg283_1, arg89_1, buf297, buf298, arg90_1, arg91_1, buf300, 602112, grid=grid(602112), stream=stream0)
        del arg90_1
        del arg91_1
        buf302 = buf283; del buf283  # reuse
        # Source Nodes: [group_norm_29, mul_29, mul_30, sub_15, x_125, x_126, x_129, x_131, x_132, y_15], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf302, buf300, buf293, arg283_1, arg89_1, arg92_1, 602112, grid=grid(602112), stream=stream0)
        del arg283_1
        del arg89_1
        del arg92_1
        del buf293
        buf303 = buf296; del buf296  # reuse
        buf304 = buf295; del buf295  # reuse
        buf305 = buf294; del buf294  # reuse
        # Source Nodes: [group_norm_31], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf302, buf303, buf304, buf305, 80, 7527, grid=grid(80), stream=stream0)
        buf306 = buf298; del buf298  # reuse
        buf307 = buf297; del buf297  # reuse
        # Source Nodes: [group_norm_31], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf303, buf304, buf305, buf306, buf307, 8, 10, grid=grid(8), stream=stream0)
        buf309 = reinterpret_tensor(buf300, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf300  # reuse
        # Source Nodes: [group_norm_31], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf302, buf306, buf307, arg93_1, arg94_1, buf309, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg93_1
        del arg94_1
        # Source Nodes: [group_norm_31, x_133], Original ATen: [aten.convolution, aten.native_group_norm]
        buf310 = extern_kernels.convolution(buf309, arg284_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg284_1
        buf311 = buf292; del buf292  # reuse
        # Source Nodes: [group_norm_31, x_133, x_134], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf310, arg285_1, buf311, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg285_1
        del buf310
        # Source Nodes: [group_norm_31, x_133, x_134, x_137], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf312 = extern_kernels.convolution(buf311, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg286_1
        buf313 = buf305; del buf305  # reuse
        buf314 = buf304; del buf304  # reuse
        buf315 = buf303; del buf303  # reuse
        # Source Nodes: [group_norm_32], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf302, buf312, arg287_1, arg95_1, buf313, buf314, buf315, 80, 7527, grid=grid(80), stream=stream0)
        buf316 = buf307; del buf307  # reuse
        buf317 = buf306; del buf306  # reuse
        # Source Nodes: [group_norm_32], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf313, buf314, buf315, buf316, buf317, 8, 10, grid=grid(8), stream=stream0)
        buf319 = reinterpret_tensor(buf309, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf309  # reuse
        # Source Nodes: [group_norm_32], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf302, buf312, arg287_1, arg95_1, buf316, buf317, arg96_1, arg97_1, buf319, 602112, grid=grid(602112), stream=stream0)
        del arg96_1
        del arg97_1
        buf321 = buf302; del buf302  # reuse
        # Source Nodes: [group_norm_31, mul_31, mul_32, sub_16, x_133, x_134, x_137, x_139, x_140, y_16], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf321, buf319, buf312, arg287_1, arg95_1, arg98_1, 602112, grid=grid(602112), stream=stream0)
        del arg287_1
        del arg95_1
        del arg98_1
        del buf312
        buf322 = buf315; del buf315  # reuse
        buf323 = buf314; del buf314  # reuse
        buf324 = buf313; del buf313  # reuse
        # Source Nodes: [group_norm_33], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf321, buf322, buf323, buf324, 80, 7527, grid=grid(80), stream=stream0)
        buf325 = buf317; del buf317  # reuse
        buf326 = buf316; del buf316  # reuse
        # Source Nodes: [group_norm_33], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf322, buf323, buf324, buf325, buf326, 8, 10, grid=grid(8), stream=stream0)
        buf328 = reinterpret_tensor(buf319, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf319  # reuse
        # Source Nodes: [group_norm_33], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf321, buf325, buf326, arg99_1, arg100_1, buf328, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg100_1
        del arg99_1
        # Source Nodes: [group_norm_33, x_141], Original ATen: [aten.convolution, aten.native_group_norm]
        buf329 = extern_kernels.convolution(buf328, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg288_1
        buf330 = buf311; del buf311  # reuse
        # Source Nodes: [group_norm_33, x_141, x_142], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf329, arg289_1, buf330, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg289_1
        del buf329
        # Source Nodes: [group_norm_33, x_141, x_142, x_145], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf331 = extern_kernels.convolution(buf330, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg290_1
        buf332 = buf324; del buf324  # reuse
        buf333 = buf323; del buf323  # reuse
        buf334 = buf322; del buf322  # reuse
        # Source Nodes: [group_norm_34], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf321, buf331, arg291_1, arg101_1, buf332, buf333, buf334, 80, 7527, grid=grid(80), stream=stream0)
        buf335 = buf326; del buf326  # reuse
        buf336 = buf325; del buf325  # reuse
        # Source Nodes: [group_norm_34], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf332, buf333, buf334, buf335, buf336, 8, 10, grid=grid(8), stream=stream0)
        buf338 = reinterpret_tensor(buf328, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf328  # reuse
        # Source Nodes: [group_norm_34], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf321, buf331, arg291_1, arg101_1, buf335, buf336, arg102_1, arg103_1, buf338, 602112, grid=grid(602112), stream=stream0)
        del arg102_1
        del arg103_1
        buf340 = buf321; del buf321  # reuse
        # Source Nodes: [group_norm_33, mul_33, mul_34, sub_17, x_141, x_142, x_145, x_147, x_148, y_17], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf340, buf338, buf331, arg291_1, arg101_1, arg104_1, 602112, grid=grid(602112), stream=stream0)
        del arg101_1
        del arg104_1
        del arg291_1
        del buf331
        buf341 = buf334; del buf334  # reuse
        buf342 = buf333; del buf333  # reuse
        buf343 = buf332; del buf332  # reuse
        # Source Nodes: [group_norm_35], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf340, buf341, buf342, buf343, 80, 7527, grid=grid(80), stream=stream0)
        buf344 = buf336; del buf336  # reuse
        buf345 = buf335; del buf335  # reuse
        # Source Nodes: [group_norm_35], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf341, buf342, buf343, buf344, buf345, 8, 10, grid=grid(8), stream=stream0)
        buf347 = reinterpret_tensor(buf338, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf338  # reuse
        # Source Nodes: [group_norm_35], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf340, buf344, buf345, arg105_1, arg106_1, buf347, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg105_1
        del arg106_1
        # Source Nodes: [group_norm_35, x_149], Original ATen: [aten.convolution, aten.native_group_norm]
        buf348 = extern_kernels.convolution(buf347, arg292_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg292_1
        buf349 = buf330; del buf330  # reuse
        # Source Nodes: [group_norm_35, x_149, x_150], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf348, arg293_1, buf349, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg293_1
        del buf348
        # Source Nodes: [group_norm_35, x_149, x_150, x_153], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf350 = extern_kernels.convolution(buf349, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg294_1
        buf351 = buf343; del buf343  # reuse
        buf352 = buf342; del buf342  # reuse
        buf353 = buf341; del buf341  # reuse
        # Source Nodes: [group_norm_36], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf340, buf350, arg295_1, arg107_1, buf351, buf352, buf353, 80, 7527, grid=grid(80), stream=stream0)
        buf354 = buf345; del buf345  # reuse
        buf355 = buf344; del buf344  # reuse
        # Source Nodes: [group_norm_36], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf351, buf352, buf353, buf354, buf355, 8, 10, grid=grid(8), stream=stream0)
        buf357 = reinterpret_tensor(buf347, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf347  # reuse
        # Source Nodes: [group_norm_36], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf340, buf350, arg295_1, arg107_1, buf354, buf355, arg108_1, arg109_1, buf357, 602112, grid=grid(602112), stream=stream0)
        del arg108_1
        del arg109_1
        buf359 = buf340; del buf340  # reuse
        # Source Nodes: [group_norm_35, mul_35, mul_36, sub_18, x_149, x_150, x_153, x_155, x_156, y_18], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf359, buf357, buf350, arg295_1, arg107_1, arg110_1, 602112, grid=grid(602112), stream=stream0)
        del arg107_1
        del arg110_1
        del arg295_1
        del buf350
        buf360 = buf353; del buf353  # reuse
        buf361 = buf352; del buf352  # reuse
        buf362 = buf351; del buf351  # reuse
        # Source Nodes: [group_norm_37], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf359, buf360, buf361, buf362, 80, 7527, grid=grid(80), stream=stream0)
        buf363 = buf355; del buf355  # reuse
        buf364 = buf354; del buf354  # reuse
        # Source Nodes: [group_norm_37], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf360, buf361, buf362, buf363, buf364, 8, 10, grid=grid(8), stream=stream0)
        buf366 = reinterpret_tensor(buf357, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf357  # reuse
        # Source Nodes: [group_norm_37], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf359, buf363, buf364, arg111_1, arg112_1, buf366, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg111_1
        del arg112_1
        # Source Nodes: [group_norm_37, x_157], Original ATen: [aten.convolution, aten.native_group_norm]
        buf367 = extern_kernels.convolution(buf366, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg296_1
        buf368 = buf349; del buf349  # reuse
        # Source Nodes: [group_norm_37, x_157, x_158], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf367, arg297_1, buf368, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg297_1
        del buf367
        # Source Nodes: [group_norm_37, x_157, x_158, x_161], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf369 = extern_kernels.convolution(buf368, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg298_1
        buf370 = buf362; del buf362  # reuse
        buf371 = buf361; del buf361  # reuse
        buf372 = buf360; del buf360  # reuse
        # Source Nodes: [group_norm_38], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf359, buf369, arg299_1, arg113_1, buf370, buf371, buf372, 80, 7527, grid=grid(80), stream=stream0)
        buf373 = buf364; del buf364  # reuse
        buf374 = buf363; del buf363  # reuse
        # Source Nodes: [group_norm_38], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf370, buf371, buf372, buf373, buf374, 8, 10, grid=grid(8), stream=stream0)
        buf376 = reinterpret_tensor(buf366, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf366  # reuse
        # Source Nodes: [group_norm_38], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf359, buf369, arg299_1, arg113_1, buf373, buf374, arg114_1, arg115_1, buf376, 602112, grid=grid(602112), stream=stream0)
        del arg114_1
        del arg115_1
        buf378 = buf359; del buf359  # reuse
        # Source Nodes: [group_norm_37, mul_37, mul_38, sub_19, x_157, x_158, x_161, x_163, x_164, y_19], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf378, buf376, buf369, arg299_1, arg113_1, arg116_1, 602112, grid=grid(602112), stream=stream0)
        del arg113_1
        del arg116_1
        del arg299_1
        del buf369
        buf379 = buf372; del buf372  # reuse
        buf380 = buf371; del buf371  # reuse
        buf381 = buf370; del buf370  # reuse
        # Source Nodes: [group_norm_39], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf378, buf379, buf380, buf381, 80, 7527, grid=grid(80), stream=stream0)
        buf382 = buf374; del buf374  # reuse
        buf383 = buf373; del buf373  # reuse
        # Source Nodes: [group_norm_39], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf379, buf380, buf381, buf382, buf383, 8, 10, grid=grid(8), stream=stream0)
        buf385 = reinterpret_tensor(buf376, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf376  # reuse
        # Source Nodes: [group_norm_39], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf378, buf382, buf383, arg117_1, arg118_1, buf385, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg117_1
        del arg118_1
        # Source Nodes: [group_norm_39, x_165], Original ATen: [aten.convolution, aten.native_group_norm]
        buf386 = extern_kernels.convolution(buf385, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg300_1
        buf387 = buf368; del buf368  # reuse
        # Source Nodes: [group_norm_39, x_165, x_166], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf386, arg301_1, buf387, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg301_1
        del buf386
        # Source Nodes: [group_norm_39, x_165, x_166, x_169], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf388 = extern_kernels.convolution(buf387, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg302_1
        buf389 = buf381; del buf381  # reuse
        buf390 = buf380; del buf380  # reuse
        buf391 = buf379; del buf379  # reuse
        # Source Nodes: [group_norm_40], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf378, buf388, arg303_1, arg119_1, buf389, buf390, buf391, 80, 7527, grid=grid(80), stream=stream0)
        buf392 = buf383; del buf383  # reuse
        buf393 = buf382; del buf382  # reuse
        # Source Nodes: [group_norm_40], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf389, buf390, buf391, buf392, buf393, 8, 10, grid=grid(8), stream=stream0)
        buf395 = reinterpret_tensor(buf385, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf385  # reuse
        # Source Nodes: [group_norm_40], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf378, buf388, arg303_1, arg119_1, buf392, buf393, arg120_1, arg121_1, buf395, 602112, grid=grid(602112), stream=stream0)
        del arg120_1
        del arg121_1
        buf397 = buf378; del buf378  # reuse
        # Source Nodes: [group_norm_39, mul_39, mul_40, sub_20, x_165, x_166, x_169, x_171, x_172, y_20], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf397, buf395, buf388, arg303_1, arg119_1, arg122_1, 602112, grid=grid(602112), stream=stream0)
        del arg119_1
        del arg122_1
        del arg303_1
        del buf388
        buf398 = buf391; del buf391  # reuse
        buf399 = buf390; del buf390  # reuse
        buf400 = buf389; del buf389  # reuse
        # Source Nodes: [group_norm_41], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf397, buf398, buf399, buf400, 80, 7527, grid=grid(80), stream=stream0)
        buf401 = buf393; del buf393  # reuse
        buf402 = buf392; del buf392  # reuse
        # Source Nodes: [group_norm_41], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf398, buf399, buf400, buf401, buf402, 8, 10, grid=grid(8), stream=stream0)
        buf404 = reinterpret_tensor(buf395, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf395  # reuse
        # Source Nodes: [group_norm_41], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf397, buf401, buf402, arg123_1, arg124_1, buf404, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg123_1
        del arg124_1
        # Source Nodes: [group_norm_41, x_173], Original ATen: [aten.convolution, aten.native_group_norm]
        buf405 = extern_kernels.convolution(buf404, arg304_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg304_1
        buf406 = buf387; del buf387  # reuse
        # Source Nodes: [group_norm_41, x_173, x_174], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf405, arg305_1, buf406, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg305_1
        del buf405
        # Source Nodes: [group_norm_41, x_173, x_174, x_177], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf407 = extern_kernels.convolution(buf406, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg306_1
        buf408 = buf400; del buf400  # reuse
        buf409 = buf399; del buf399  # reuse
        buf410 = buf398; del buf398  # reuse
        # Source Nodes: [group_norm_42], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf397, buf407, arg307_1, arg125_1, buf408, buf409, buf410, 80, 7527, grid=grid(80), stream=stream0)
        buf411 = buf402; del buf402  # reuse
        buf412 = buf401; del buf401  # reuse
        # Source Nodes: [group_norm_42], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf408, buf409, buf410, buf411, buf412, 8, 10, grid=grid(8), stream=stream0)
        buf414 = reinterpret_tensor(buf404, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf404  # reuse
        # Source Nodes: [group_norm_42], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf397, buf407, arg307_1, arg125_1, buf411, buf412, arg126_1, arg127_1, buf414, 602112, grid=grid(602112), stream=stream0)
        del arg126_1
        del arg127_1
        buf416 = buf397; del buf397  # reuse
        # Source Nodes: [group_norm_41, mul_41, mul_42, sub_21, x_173, x_174, x_177, x_179, x_180, y_21], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf416, buf414, buf407, arg307_1, arg125_1, arg128_1, 602112, grid=grid(602112), stream=stream0)
        del arg125_1
        del arg128_1
        del arg307_1
        del buf407
        buf417 = buf410; del buf410  # reuse
        buf418 = buf409; del buf409  # reuse
        buf419 = buf408; del buf408  # reuse
        # Source Nodes: [group_norm_43], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf416, buf417, buf418, buf419, 80, 7527, grid=grid(80), stream=stream0)
        buf420 = buf412; del buf412  # reuse
        buf421 = buf411; del buf411  # reuse
        # Source Nodes: [group_norm_43], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf417, buf418, buf419, buf420, buf421, 8, 10, grid=grid(8), stream=stream0)
        buf423 = reinterpret_tensor(buf414, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf414  # reuse
        # Source Nodes: [group_norm_43], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf416, buf420, buf421, arg129_1, arg130_1, buf423, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg129_1
        del arg130_1
        # Source Nodes: [group_norm_43, x_181], Original ATen: [aten.convolution, aten.native_group_norm]
        buf424 = extern_kernels.convolution(buf423, arg308_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg308_1
        buf425 = buf406; del buf406  # reuse
        # Source Nodes: [group_norm_43, x_181, x_182], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf424, arg309_1, buf425, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg309_1
        del buf424
        # Source Nodes: [group_norm_43, x_181, x_182, x_185], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf426 = extern_kernels.convolution(buf425, arg310_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg310_1
        buf427 = buf419; del buf419  # reuse
        buf428 = buf418; del buf418  # reuse
        buf429 = buf417; del buf417  # reuse
        # Source Nodes: [group_norm_44], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf416, buf426, arg311_1, arg131_1, buf427, buf428, buf429, 80, 7527, grid=grid(80), stream=stream0)
        buf430 = buf421; del buf421  # reuse
        buf431 = buf420; del buf420  # reuse
        # Source Nodes: [group_norm_44], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf427, buf428, buf429, buf430, buf431, 8, 10, grid=grid(8), stream=stream0)
        buf433 = reinterpret_tensor(buf423, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf423  # reuse
        # Source Nodes: [group_norm_44], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf416, buf426, arg311_1, arg131_1, buf430, buf431, arg132_1, arg133_1, buf433, 602112, grid=grid(602112), stream=stream0)
        del arg132_1
        del arg133_1
        buf435 = buf416; del buf416  # reuse
        # Source Nodes: [group_norm_43, mul_43, mul_44, sub_22, x_181, x_182, x_185, x_187, x_188, y_22], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf435, buf433, buf426, arg311_1, arg131_1, arg134_1, 602112, grid=grid(602112), stream=stream0)
        del arg131_1
        del arg134_1
        del arg311_1
        del buf426
        buf436 = buf429; del buf429  # reuse
        buf437 = buf428; del buf428  # reuse
        buf438 = buf427; del buf427  # reuse
        # Source Nodes: [group_norm_45], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf435, buf436, buf437, buf438, 80, 7527, grid=grid(80), stream=stream0)
        buf439 = buf431; del buf431  # reuse
        buf440 = buf430; del buf430  # reuse
        # Source Nodes: [group_norm_45], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf436, buf437, buf438, buf439, buf440, 8, 10, grid=grid(8), stream=stream0)
        buf442 = reinterpret_tensor(buf433, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf433  # reuse
        # Source Nodes: [group_norm_45], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf435, buf439, buf440, arg135_1, arg136_1, buf442, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg135_1
        del arg136_1
        # Source Nodes: [group_norm_45, x_189], Original ATen: [aten.convolution, aten.native_group_norm]
        buf443 = extern_kernels.convolution(buf442, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg312_1
        buf444 = buf425; del buf425  # reuse
        # Source Nodes: [group_norm_45, x_189, x_190], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf443, arg313_1, buf444, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg313_1
        del buf443
        # Source Nodes: [group_norm_45, x_189, x_190, x_193], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf445 = extern_kernels.convolution(buf444, arg314_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg314_1
        buf446 = buf438; del buf438  # reuse
        buf447 = buf437; del buf437  # reuse
        buf448 = buf436; del buf436  # reuse
        # Source Nodes: [group_norm_46], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf435, buf445, arg315_1, arg137_1, buf446, buf447, buf448, 80, 7527, grid=grid(80), stream=stream0)
        buf449 = buf440; del buf440  # reuse
        buf450 = buf439; del buf439  # reuse
        # Source Nodes: [group_norm_46], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf446, buf447, buf448, buf449, buf450, 8, 10, grid=grid(8), stream=stream0)
        buf452 = reinterpret_tensor(buf442, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf442  # reuse
        # Source Nodes: [group_norm_46], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf435, buf445, arg315_1, arg137_1, buf449, buf450, arg138_1, arg139_1, buf452, 602112, grid=grid(602112), stream=stream0)
        del arg138_1
        del arg139_1
        buf454 = buf435; del buf435  # reuse
        # Source Nodes: [group_norm_45, mul_45, mul_46, sub_23, x_189, x_190, x_193, x_195, x_196, y_23], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf454, buf452, buf445, arg315_1, arg137_1, arg140_1, 602112, grid=grid(602112), stream=stream0)
        del arg137_1
        del arg140_1
        del arg315_1
        del buf445
        buf455 = buf448; del buf448  # reuse
        buf456 = buf447; del buf447  # reuse
        buf457 = buf446; del buf446  # reuse
        # Source Nodes: [group_norm_47], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf454, buf455, buf456, buf457, 80, 7527, grid=grid(80), stream=stream0)
        buf458 = buf450; del buf450  # reuse
        buf459 = buf449; del buf449  # reuse
        # Source Nodes: [group_norm_47], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf455, buf456, buf457, buf458, buf459, 8, 10, grid=grid(8), stream=stream0)
        buf461 = reinterpret_tensor(buf452, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf452  # reuse
        # Source Nodes: [group_norm_47], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf454, buf458, buf459, arg141_1, arg142_1, buf461, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg141_1
        del arg142_1
        # Source Nodes: [group_norm_47, x_197], Original ATen: [aten.convolution, aten.native_group_norm]
        buf462 = extern_kernels.convolution(buf461, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg316_1
        buf463 = buf444; del buf444  # reuse
        # Source Nodes: [group_norm_47, x_197, x_198], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf462, arg317_1, buf463, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg317_1
        del buf462
        # Source Nodes: [group_norm_47, x_197, x_198, x_201], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf464 = extern_kernels.convolution(buf463, arg318_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg318_1
        buf465 = buf457; del buf457  # reuse
        buf466 = buf456; del buf456  # reuse
        buf467 = buf455; del buf455  # reuse
        # Source Nodes: [group_norm_48], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf454, buf464, arg319_1, arg143_1, buf465, buf466, buf467, 80, 7527, grid=grid(80), stream=stream0)
        buf468 = buf459; del buf459  # reuse
        buf469 = buf458; del buf458  # reuse
        # Source Nodes: [group_norm_48], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf465, buf466, buf467, buf468, buf469, 8, 10, grid=grid(8), stream=stream0)
        buf471 = reinterpret_tensor(buf461, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf461  # reuse
        # Source Nodes: [group_norm_48], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf454, buf464, arg319_1, arg143_1, buf468, buf469, arg144_1, arg145_1, buf471, 602112, grid=grid(602112), stream=stream0)
        del arg144_1
        del arg145_1
        buf473 = buf454; del buf454  # reuse
        # Source Nodes: [group_norm_47, mul_47, mul_48, sub_24, x_197, x_198, x_201, x_203, x_204, y_24], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf473, buf471, buf464, arg319_1, arg143_1, arg146_1, 602112, grid=grid(602112), stream=stream0)
        del arg143_1
        del arg146_1
        del arg319_1
        del buf464
        buf474 = buf467; del buf467  # reuse
        buf475 = buf466; del buf466  # reuse
        buf476 = buf465; del buf465  # reuse
        # Source Nodes: [group_norm_49], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf473, buf474, buf475, buf476, 80, 7527, grid=grid(80), stream=stream0)
        buf477 = buf469; del buf469  # reuse
        buf478 = buf468; del buf468  # reuse
        # Source Nodes: [group_norm_49], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf474, buf475, buf476, buf477, buf478, 8, 10, grid=grid(8), stream=stream0)
        buf480 = reinterpret_tensor(buf471, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf471  # reuse
        # Source Nodes: [group_norm_49], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf473, buf477, buf478, arg147_1, arg148_1, buf480, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg147_1
        del arg148_1
        # Source Nodes: [group_norm_49, x_205], Original ATen: [aten.convolution, aten.native_group_norm]
        buf481 = extern_kernels.convolution(buf480, arg320_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg320_1
        buf482 = buf463; del buf463  # reuse
        # Source Nodes: [group_norm_49, x_205, x_206], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf481, arg321_1, buf482, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg321_1
        del buf481
        # Source Nodes: [group_norm_49, x_205, x_206, x_209], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf483 = extern_kernels.convolution(buf482, arg322_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg322_1
        buf484 = buf476; del buf476  # reuse
        buf485 = buf475; del buf475  # reuse
        buf486 = buf474; del buf474  # reuse
        # Source Nodes: [group_norm_50], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf473, buf483, arg323_1, arg149_1, buf484, buf485, buf486, 80, 7527, grid=grid(80), stream=stream0)
        buf487 = buf478; del buf478  # reuse
        buf488 = buf477; del buf477  # reuse
        # Source Nodes: [group_norm_50], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf484, buf485, buf486, buf487, buf488, 8, 10, grid=grid(8), stream=stream0)
        buf490 = reinterpret_tensor(buf480, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf480  # reuse
        # Source Nodes: [group_norm_50], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf473, buf483, arg323_1, arg149_1, buf487, buf488, arg150_1, arg151_1, buf490, 602112, grid=grid(602112), stream=stream0)
        del arg150_1
        del arg151_1
        buf492 = buf473; del buf473  # reuse
        # Source Nodes: [group_norm_49, mul_49, mul_50, sub_25, x_205, x_206, x_209, x_211, x_212, y_25], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf492, buf490, buf483, arg323_1, arg149_1, arg152_1, 602112, grid=grid(602112), stream=stream0)
        del arg149_1
        del arg152_1
        del arg323_1
        del buf483
        buf493 = buf486; del buf486  # reuse
        buf494 = buf485; del buf485  # reuse
        buf495 = buf484; del buf484  # reuse
        # Source Nodes: [group_norm_51], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf492, buf493, buf494, buf495, 80, 7527, grid=grid(80), stream=stream0)
        buf496 = buf488; del buf488  # reuse
        buf497 = buf487; del buf487  # reuse
        # Source Nodes: [group_norm_51], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf493, buf494, buf495, buf496, buf497, 8, 10, grid=grid(8), stream=stream0)
        buf499 = reinterpret_tensor(buf490, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf490  # reuse
        # Source Nodes: [group_norm_51], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf492, buf496, buf497, arg153_1, arg154_1, buf499, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg153_1
        del arg154_1
        # Source Nodes: [group_norm_51, x_213], Original ATen: [aten.convolution, aten.native_group_norm]
        buf500 = extern_kernels.convolution(buf499, arg324_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg324_1
        buf501 = buf482; del buf482  # reuse
        # Source Nodes: [group_norm_51, x_213, x_214], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf500, arg325_1, buf501, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg325_1
        del buf500
        # Source Nodes: [group_norm_51, x_213, x_214, x_217], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf502 = extern_kernels.convolution(buf501, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg326_1
        buf503 = buf495; del buf495  # reuse
        buf504 = buf494; del buf494  # reuse
        buf505 = buf493; del buf493  # reuse
        # Source Nodes: [group_norm_52], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf492, buf502, arg327_1, arg155_1, buf503, buf504, buf505, 80, 7527, grid=grid(80), stream=stream0)
        buf506 = buf497; del buf497  # reuse
        buf507 = buf496; del buf496  # reuse
        # Source Nodes: [group_norm_52], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf503, buf504, buf505, buf506, buf507, 8, 10, grid=grid(8), stream=stream0)
        buf509 = reinterpret_tensor(buf499, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf499  # reuse
        # Source Nodes: [group_norm_52], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf492, buf502, arg327_1, arg155_1, buf506, buf507, arg156_1, arg157_1, buf509, 602112, grid=grid(602112), stream=stream0)
        del arg156_1
        del arg157_1
        buf511 = buf492; del buf492  # reuse
        # Source Nodes: [group_norm_51, mul_51, mul_52, sub_26, x_213, x_214, x_217, x_219, x_220, y_26], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf511, buf509, buf502, arg327_1, arg155_1, arg158_1, 602112, grid=grid(602112), stream=stream0)
        del arg155_1
        del arg158_1
        del arg327_1
        del buf502
        buf512 = buf505; del buf505  # reuse
        buf513 = buf504; del buf504  # reuse
        buf514 = buf503; del buf503  # reuse
        # Source Nodes: [group_norm_53], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf511, buf512, buf513, buf514, 80, 7527, grid=grid(80), stream=stream0)
        buf515 = buf507; del buf507  # reuse
        buf516 = buf506; del buf506  # reuse
        # Source Nodes: [group_norm_53], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf512, buf513, buf514, buf515, buf516, 8, 10, grid=grid(8), stream=stream0)
        buf518 = reinterpret_tensor(buf509, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf509  # reuse
        # Source Nodes: [group_norm_53], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf511, buf515, buf516, arg159_1, arg160_1, buf518, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg159_1
        del arg160_1
        # Source Nodes: [group_norm_53, x_221], Original ATen: [aten.convolution, aten.native_group_norm]
        buf519 = extern_kernels.convolution(buf518, arg328_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg328_1
        buf520 = buf501; del buf501  # reuse
        # Source Nodes: [group_norm_53, x_221, x_222], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf519, arg329_1, buf520, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg329_1
        del buf519
        # Source Nodes: [group_norm_53, x_221, x_222, x_225], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf521 = extern_kernels.convolution(buf520, arg330_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf521, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg330_1
        buf522 = buf514; del buf514  # reuse
        buf523 = buf513; del buf513  # reuse
        buf524 = buf512; del buf512  # reuse
        # Source Nodes: [group_norm_54], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf511, buf521, arg331_1, arg161_1, buf522, buf523, buf524, 80, 7527, grid=grid(80), stream=stream0)
        buf525 = buf516; del buf516  # reuse
        buf526 = buf515; del buf515  # reuse
        # Source Nodes: [group_norm_54], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf522, buf523, buf524, buf525, buf526, 8, 10, grid=grid(8), stream=stream0)
        buf528 = reinterpret_tensor(buf518, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf518  # reuse
        # Source Nodes: [group_norm_54], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf511, buf521, arg331_1, arg161_1, buf525, buf526, arg162_1, arg163_1, buf528, 602112, grid=grid(602112), stream=stream0)
        del arg162_1
        del arg163_1
        buf530 = buf511; del buf511  # reuse
        # Source Nodes: [group_norm_53, mul_53, mul_54, sub_27, x_221, x_222, x_225, x_227, x_228, y_27], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf530, buf528, buf521, arg331_1, arg161_1, arg164_1, 602112, grid=grid(602112), stream=stream0)
        del arg161_1
        del arg164_1
        del arg331_1
        del buf521
        buf531 = buf524; del buf524  # reuse
        buf532 = buf523; del buf523  # reuse
        buf533 = buf522; del buf522  # reuse
        # Source Nodes: [group_norm_55], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf530, buf531, buf532, buf533, 80, 7527, grid=grid(80), stream=stream0)
        buf534 = buf526; del buf526  # reuse
        buf535 = buf525; del buf525  # reuse
        # Source Nodes: [group_norm_55], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf531, buf532, buf533, buf534, buf535, 8, 10, grid=grid(8), stream=stream0)
        buf537 = reinterpret_tensor(buf528, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf528  # reuse
        # Source Nodes: [group_norm_55], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf530, buf534, buf535, arg165_1, arg166_1, buf537, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg165_1
        del arg166_1
        # Source Nodes: [group_norm_55, x_229], Original ATen: [aten.convolution, aten.native_group_norm]
        buf538 = extern_kernels.convolution(buf537, arg332_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg332_1
        buf539 = buf520; del buf520  # reuse
        # Source Nodes: [group_norm_55, x_229, x_230], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf538, arg333_1, buf539, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg333_1
        del buf538
        # Source Nodes: [group_norm_55, x_229, x_230, x_233], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf540 = extern_kernels.convolution(buf539, arg334_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg334_1
        buf541 = buf533; del buf533  # reuse
        buf542 = buf532; del buf532  # reuse
        buf543 = buf531; del buf531  # reuse
        # Source Nodes: [group_norm_56], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf530, buf540, arg335_1, arg167_1, buf541, buf542, buf543, 80, 7527, grid=grid(80), stream=stream0)
        buf544 = buf535; del buf535  # reuse
        buf545 = buf534; del buf534  # reuse
        # Source Nodes: [group_norm_56], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf541, buf542, buf543, buf544, buf545, 8, 10, grid=grid(8), stream=stream0)
        buf547 = reinterpret_tensor(buf537, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf537  # reuse
        # Source Nodes: [group_norm_56], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf530, buf540, arg335_1, arg167_1, buf544, buf545, arg168_1, arg169_1, buf547, 602112, grid=grid(602112), stream=stream0)
        del arg168_1
        del arg169_1
        buf549 = buf530; del buf530  # reuse
        # Source Nodes: [group_norm_55, mul_55, mul_56, sub_28, x_229, x_230, x_233, x_235, x_236, y_28], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf549, buf547, buf540, arg335_1, arg167_1, arg170_1, 602112, grid=grid(602112), stream=stream0)
        del arg167_1
        del arg170_1
        del arg335_1
        del buf540
        buf550 = buf543; del buf543  # reuse
        buf551 = buf542; del buf542  # reuse
        buf552 = buf541; del buf541  # reuse
        # Source Nodes: [group_norm_57], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf549, buf550, buf551, buf552, 80, 7527, grid=grid(80), stream=stream0)
        buf553 = buf545; del buf545  # reuse
        buf554 = buf544; del buf544  # reuse
        # Source Nodes: [group_norm_57], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf550, buf551, buf552, buf553, buf554, 8, 10, grid=grid(8), stream=stream0)
        buf556 = reinterpret_tensor(buf547, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf547  # reuse
        # Source Nodes: [group_norm_57], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf549, buf553, buf554, arg171_1, arg172_1, buf556, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg171_1
        del arg172_1
        # Source Nodes: [group_norm_57, x_237], Original ATen: [aten.convolution, aten.native_group_norm]
        buf557 = extern_kernels.convolution(buf556, arg336_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg336_1
        buf558 = buf539; del buf539  # reuse
        # Source Nodes: [group_norm_57, x_237, x_238], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf557, arg337_1, buf558, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg337_1
        del buf557
        # Source Nodes: [group_norm_57, x_237, x_238, x_241], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf559 = extern_kernels.convolution(buf558, arg338_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf559, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg338_1
        buf560 = buf552; del buf552  # reuse
        buf561 = buf551; del buf551  # reuse
        buf562 = buf550; del buf550  # reuse
        # Source Nodes: [group_norm_58], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_34.run(buf549, buf559, arg339_1, arg173_1, buf560, buf561, buf562, 80, 7527, grid=grid(80), stream=stream0)
        buf563 = buf554; del buf554  # reuse
        buf564 = buf553; del buf553  # reuse
        # Source Nodes: [group_norm_58], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf560, buf561, buf562, buf563, buf564, 8, 10, grid=grid(8), stream=stream0)
        buf566 = reinterpret_tensor(buf556, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf556  # reuse
        # Source Nodes: [group_norm_58], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_35.run(buf549, buf559, arg339_1, arg173_1, buf563, buf564, arg174_1, arg175_1, buf566, 602112, grid=grid(602112), stream=stream0)
        del arg174_1
        del arg175_1
        buf568 = buf549; del buf549  # reuse
        # Source Nodes: [group_norm_57, mul_57, mul_58, sub_29, x_237, x_238, x_241, x_243, x_244, y_29], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_36.run(buf568, buf566, buf559, arg339_1, arg173_1, arg176_1, 602112, grid=grid(602112), stream=stream0)
        del arg173_1
        del arg176_1
        del arg339_1
        del buf559
        buf569 = buf562; del buf562  # reuse
        buf570 = buf561; del buf561  # reuse
        buf571 = buf560; del buf560  # reuse
        # Source Nodes: [group_norm_59], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_31.run(buf568, buf569, buf570, buf571, 80, 7527, grid=grid(80), stream=stream0)
        buf572 = buf564; del buf564  # reuse
        buf573 = buf563; del buf563  # reuse
        # Source Nodes: [group_norm_59], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_28.run(buf569, buf570, buf571, buf572, buf573, 8, 10, grid=grid(8), stream=stream0)
        del buf569
        del buf570
        del buf571
        buf575 = reinterpret_tensor(buf566, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf566  # reuse
        # Source Nodes: [group_norm_59], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_32.run(buf568, buf572, buf573, arg177_1, arg178_1, buf575, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg177_1
        del arg178_1
        # Source Nodes: [group_norm_59, x_245], Original ATen: [aten.convolution, aten.native_group_norm]
        buf576 = extern_kernels.convolution(buf575, arg340_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (8, 1536, 14, 14), (301056, 196, 14, 1))
        del arg340_1
        buf577 = buf558; del buf558  # reuse
        # Source Nodes: [group_norm_59, x_245, x_246], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_33.run(buf576, arg341_1, buf577, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del arg341_1
        del buf576
        # Source Nodes: [group_norm_59, x_245, x_246, x_249], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf578 = extern_kernels.convolution(buf577, arg342_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg342_1
        del buf577
        buf579 = buf575; del buf575  # reuse
        # Source Nodes: [group_norm_59, mul_59, x_245, x_246, x_249, x_252], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_37.run(buf568, buf578, arg343_1, arg179_1, buf579, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg179_1
        del arg343_1
        del buf568
        del buf578
        buf580 = empty_strided((768, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_59, mul_59, x_245, x_246, x_249, x_252, x_255], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        triton_poi_fused_add_convolution_gelu_mul_native_group_norm_38.run(arg344_1, buf580, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del arg344_1
        # Source Nodes: [group_norm_59, mul_59, x_245, x_246, x_249, x_252, x_255], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm]
        buf581 = extern_kernels.convolution(buf579, buf580, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf581, (8, 768, 7, 7), (37632, 49, 7, 1))
        del buf579
        del buf580
        buf582 = empty_strided((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), device='cuda', dtype=torch.float32)
        buf583 = empty_strided((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), device='cuda', dtype=torch.float32)
        buf584 = empty_strided((8, 1, 1, 1, 5), (5, 40, 40, 40, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_39.run(buf581, arg345_1, buf582, buf583, buf584, 40, 7527, grid=grid(40), stream=stream0)
        buf585 = buf573; del buf573  # reuse
        buf586 = buf572; del buf572  # reuse
        # Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf582, buf583, buf584, buf585, buf586, 8, 5, grid=grid(8), stream=stream0)
        buf588 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_41.run(buf581, arg345_1, buf585, buf586, arg180_1, arg181_1, buf588, 301056, grid=grid(301056), stream=stream0)
        del arg180_1
        del arg181_1
        buf590 = buf581; del buf581  # reuse
        # Source Nodes: [group_norm_59, mul_59, mul_60, sub_30, x_245, x_246, x_249, x_252, x_255, x_256, y_30], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_42.run(buf590, buf588, arg345_1, arg182_1, 301056, grid=grid(301056), stream=stream0)
        del arg182_1
        del arg345_1
        buf591 = buf584; del buf584  # reuse
        buf592 = buf583; del buf583  # reuse
        buf593 = buf582; del buf582  # reuse
        # Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_43.run(buf590, buf591, buf592, buf593, 40, 7527, grid=grid(40), stream=stream0)
        buf594 = buf586; del buf586  # reuse
        buf595 = buf585; del buf585  # reuse
        # Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf591, buf592, buf593, buf594, buf595, 8, 5, grid=grid(8), stream=stream0)
        buf597 = reinterpret_tensor(buf588, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf588  # reuse
        # Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_44.run(buf590, buf594, buf595, arg183_1, arg184_1, buf597, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg183_1
        del arg184_1
        # Source Nodes: [group_norm_61, x_257], Original ATen: [aten.convolution, aten.native_group_norm]
        buf598 = extern_kernels.convolution(buf597, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg346_1
        buf599 = reinterpret_tensor(buf234, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf234  # reuse
        # Source Nodes: [group_norm_61, x_257, x_258], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_45.run(buf598, arg347_1, buf599, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del arg347_1
        del buf598
        # Source Nodes: [group_norm_61, x_257, x_258, x_261], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf600 = extern_kernels.convolution(buf599, arg348_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg348_1
        buf601 = buf593; del buf593  # reuse
        buf602 = buf592; del buf592  # reuse
        buf603 = buf591; del buf591  # reuse
        # Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_46.run(buf590, buf600, arg349_1, arg185_1, buf601, buf602, buf603, 40, 7527, grid=grid(40), stream=stream0)
        buf604 = buf595; del buf595  # reuse
        buf605 = buf594; del buf594  # reuse
        # Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf601, buf602, buf603, buf604, buf605, 8, 5, grid=grid(8), stream=stream0)
        buf607 = reinterpret_tensor(buf597, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf597  # reuse
        # Source Nodes: [group_norm_62], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_47.run(buf590, buf600, arg349_1, arg185_1, buf604, buf605, arg186_1, arg187_1, buf607, 301056, grid=grid(301056), stream=stream0)
        del arg186_1
        del arg187_1
        buf609 = buf590; del buf590  # reuse
        # Source Nodes: [group_norm_61, mul_61, mul_62, sub_31, x_257, x_258, x_261, x_263, x_264, y_31], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_48.run(buf609, buf607, buf600, arg349_1, arg185_1, arg188_1, 301056, grid=grid(301056), stream=stream0)
        del arg185_1
        del arg188_1
        del arg349_1
        del buf600
        buf610 = buf603; del buf603  # reuse
        buf611 = buf602; del buf602  # reuse
        buf612 = buf601; del buf601  # reuse
        # Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_43.run(buf609, buf610, buf611, buf612, 40, 7527, grid=grid(40), stream=stream0)
        buf613 = buf605; del buf605  # reuse
        buf614 = buf604; del buf604  # reuse
        # Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf610, buf611, buf612, buf613, buf614, 8, 5, grid=grid(8), stream=stream0)
        buf616 = reinterpret_tensor(buf607, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf607  # reuse
        # Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_44.run(buf609, buf613, buf614, arg189_1, arg190_1, buf616, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg189_1
        del arg190_1
        # Source Nodes: [group_norm_63, x_265], Original ATen: [aten.convolution, aten.native_group_norm]
        buf617 = extern_kernels.convolution(buf616, arg350_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg350_1
        buf618 = buf599; del buf599  # reuse
        # Source Nodes: [group_norm_63, x_265, x_266], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_45.run(buf617, arg351_1, buf618, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del arg351_1
        del buf617
        # Source Nodes: [group_norm_63, x_265, x_266, x_269], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf619 = extern_kernels.convolution(buf618, arg352_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf619, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg352_1
        buf620 = buf612; del buf612  # reuse
        buf621 = buf611; del buf611  # reuse
        buf622 = buf610; del buf610  # reuse
        # Source Nodes: [group_norm_64], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_46.run(buf609, buf619, arg353_1, arg191_1, buf620, buf621, buf622, 40, 7527, grid=grid(40), stream=stream0)
        buf623 = buf614; del buf614  # reuse
        buf624 = buf613; del buf613  # reuse
        # Source Nodes: [group_norm_64], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf620, buf621, buf622, buf623, buf624, 8, 5, grid=grid(8), stream=stream0)
        buf626 = reinterpret_tensor(buf616, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf616  # reuse
        # Source Nodes: [group_norm_64], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_47.run(buf609, buf619, arg353_1, arg191_1, buf623, buf624, arg192_1, arg193_1, buf626, 301056, grid=grid(301056), stream=stream0)
        del arg192_1
        del arg193_1
        buf628 = buf609; del buf609  # reuse
        # Source Nodes: [group_norm_63, mul_63, mul_64, sub_32, x_265, x_266, x_269, x_271, x_272, y_32], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_48.run(buf628, buf626, buf619, arg353_1, arg191_1, arg194_1, 301056, grid=grid(301056), stream=stream0)
        del arg191_1
        del arg194_1
        del arg353_1
        del buf619
        buf629 = buf622; del buf622  # reuse
        buf630 = buf621; del buf621  # reuse
        buf631 = buf620; del buf620  # reuse
        # Source Nodes: [group_norm_65], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_43.run(buf628, buf629, buf630, buf631, 40, 7527, grid=grid(40), stream=stream0)
        buf632 = buf624; del buf624  # reuse
        buf633 = buf623; del buf623  # reuse
        # Source Nodes: [group_norm_65], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf629, buf630, buf631, buf632, buf633, 8, 5, grid=grid(8), stream=stream0)
        buf635 = reinterpret_tensor(buf626, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf626  # reuse
        # Source Nodes: [group_norm_65], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_44.run(buf628, buf632, buf633, arg195_1, arg196_1, buf635, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg195_1
        del arg196_1
        # Source Nodes: [group_norm_65, x_273], Original ATen: [aten.convolution, aten.native_group_norm]
        buf636 = extern_kernels.convolution(buf635, arg354_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf636, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg354_1
        buf637 = buf618; del buf618  # reuse
        # Source Nodes: [group_norm_65, x_273, x_274], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_45.run(buf636, arg355_1, buf637, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del arg355_1
        del buf636
        # Source Nodes: [group_norm_65, x_273, x_274, x_277], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf638 = extern_kernels.convolution(buf637, arg356_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf638, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg356_1
        buf639 = buf631; del buf631  # reuse
        buf640 = buf630; del buf630  # reuse
        buf641 = buf629; del buf629  # reuse
        # Source Nodes: [group_norm_66], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_46.run(buf628, buf638, arg357_1, arg197_1, buf639, buf640, buf641, 40, 7527, grid=grid(40), stream=stream0)
        buf642 = buf633; del buf633  # reuse
        buf643 = buf632; del buf632  # reuse
        # Source Nodes: [group_norm_66], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf639, buf640, buf641, buf642, buf643, 8, 5, grid=grid(8), stream=stream0)
        buf645 = reinterpret_tensor(buf635, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf635  # reuse
        # Source Nodes: [group_norm_66], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_47.run(buf628, buf638, arg357_1, arg197_1, buf642, buf643, arg198_1, arg199_1, buf645, 301056, grid=grid(301056), stream=stream0)
        del arg198_1
        del arg199_1
        buf647 = buf628; del buf628  # reuse
        # Source Nodes: [group_norm_65, mul_65, mul_66, sub_33, x_273, x_274, x_277, x_279, x_280, y_33], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_48.run(buf647, buf645, buf638, arg357_1, arg197_1, arg200_1, 301056, grid=grid(301056), stream=stream0)
        del arg197_1
        del arg200_1
        del arg357_1
        del buf638
        buf648 = buf641; del buf641  # reuse
        buf649 = buf640; del buf640  # reuse
        buf650 = buf639; del buf639  # reuse
        # Source Nodes: [group_norm_67], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_43.run(buf647, buf648, buf649, buf650, 40, 7527, grid=grid(40), stream=stream0)
        buf651 = buf643; del buf643  # reuse
        buf652 = buf642; del buf642  # reuse
        # Source Nodes: [group_norm_67], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf648, buf649, buf650, buf651, buf652, 8, 5, grid=grid(8), stream=stream0)
        buf654 = reinterpret_tensor(buf645, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf645  # reuse
        # Source Nodes: [group_norm_67], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_44.run(buf647, buf651, buf652, arg201_1, arg202_1, buf654, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg201_1
        del arg202_1
        # Source Nodes: [group_norm_67, x_281], Original ATen: [aten.convolution, aten.native_group_norm]
        buf655 = extern_kernels.convolution(buf654, arg358_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf655, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg358_1
        buf656 = buf637; del buf637  # reuse
        # Source Nodes: [group_norm_67, x_281, x_282], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_45.run(buf655, arg359_1, buf656, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del arg359_1
        del buf655
        # Source Nodes: [group_norm_67, x_281, x_282, x_285], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf657 = extern_kernels.convolution(buf656, arg360_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg360_1
        buf658 = buf650; del buf650  # reuse
        buf659 = buf649; del buf649  # reuse
        buf660 = buf648; del buf648  # reuse
        # Source Nodes: [group_norm_68], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_46.run(buf647, buf657, arg361_1, arg203_1, buf658, buf659, buf660, 40, 7527, grid=grid(40), stream=stream0)
        buf661 = buf652; del buf652  # reuse
        buf662 = buf651; del buf651  # reuse
        # Source Nodes: [group_norm_68], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf658, buf659, buf660, buf661, buf662, 8, 5, grid=grid(8), stream=stream0)
        buf664 = reinterpret_tensor(buf654, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf654  # reuse
        # Source Nodes: [group_norm_68], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_47.run(buf647, buf657, arg361_1, arg203_1, buf661, buf662, arg204_1, arg205_1, buf664, 301056, grid=grid(301056), stream=stream0)
        del arg204_1
        del arg205_1
        buf666 = buf647; del buf647  # reuse
        # Source Nodes: [group_norm_67, mul_67, mul_68, sub_34, x_281, x_282, x_285, x_287, x_288, y_34], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_48.run(buf666, buf664, buf657, arg361_1, arg203_1, arg206_1, 301056, grid=grid(301056), stream=stream0)
        del arg203_1
        del arg206_1
        del arg361_1
        del buf657
        buf667 = buf660; del buf660  # reuse
        buf668 = buf659; del buf659  # reuse
        buf669 = buf658; del buf658  # reuse
        # Source Nodes: [group_norm_69], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_43.run(buf666, buf667, buf668, buf669, 40, 7527, grid=grid(40), stream=stream0)
        buf670 = buf662; del buf662  # reuse
        buf671 = buf661; del buf661  # reuse
        # Source Nodes: [group_norm_69], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf667, buf668, buf669, buf670, buf671, 8, 5, grid=grid(8), stream=stream0)
        buf673 = reinterpret_tensor(buf664, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf664  # reuse
        # Source Nodes: [group_norm_69], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_44.run(buf666, buf670, buf671, arg207_1, arg208_1, buf673, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg207_1
        del arg208_1
        # Source Nodes: [group_norm_69, x_289], Original ATen: [aten.convolution, aten.native_group_norm]
        buf674 = extern_kernels.convolution(buf673, arg362_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf674, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg362_1
        buf675 = buf656; del buf656  # reuse
        # Source Nodes: [group_norm_69, x_289, x_290], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_45.run(buf674, arg363_1, buf675, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del arg363_1
        del buf674
        # Source Nodes: [group_norm_69, x_289, x_290, x_293], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf676 = extern_kernels.convolution(buf675, arg364_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg364_1
        buf677 = buf669; del buf669  # reuse
        buf678 = buf668; del buf668  # reuse
        buf679 = buf667; del buf667  # reuse
        # Source Nodes: [group_norm_70], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_46.run(buf666, buf676, arg365_1, arg209_1, buf677, buf678, buf679, 40, 7527, grid=grid(40), stream=stream0)
        buf680 = buf671; del buf671  # reuse
        buf681 = buf670; del buf670  # reuse
        # Source Nodes: [group_norm_70], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf677, buf678, buf679, buf680, buf681, 8, 5, grid=grid(8), stream=stream0)
        buf683 = reinterpret_tensor(buf673, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf673  # reuse
        # Source Nodes: [group_norm_70], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_47.run(buf666, buf676, arg365_1, arg209_1, buf680, buf681, arg210_1, arg211_1, buf683, 301056, grid=grid(301056), stream=stream0)
        del arg210_1
        del arg211_1
        buf685 = buf666; del buf666  # reuse
        # Source Nodes: [group_norm_69, mul_69, mul_70, sub_35, x_289, x_290, x_293, x_295, x_296, y_35], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.gelu, aten.mul, aten.native_group_norm, aten.sub]
        triton_poi_fused_add_avg_pool2d_convolution_gelu_mul_native_group_norm_sub_48.run(buf685, buf683, buf676, arg365_1, arg209_1, arg212_1, 301056, grid=grid(301056), stream=stream0)
        del arg209_1
        del arg212_1
        del arg365_1
        del buf676
        buf686 = buf679; del buf679  # reuse
        buf687 = buf678; del buf678  # reuse
        buf688 = buf677; del buf677  # reuse
        # Source Nodes: [group_norm_71], Original ATen: [aten.native_group_norm]
        triton_red_fused_native_group_norm_43.run(buf685, buf686, buf687, buf688, 40, 7527, grid=grid(40), stream=stream0)
        buf689 = buf681; del buf681  # reuse
        buf690 = buf680; del buf680  # reuse
        # Source Nodes: [group_norm_71], Original ATen: [aten.native_group_norm]
        triton_per_fused_native_group_norm_40.run(buf686, buf687, buf688, buf689, buf690, 8, 5, grid=grid(8), stream=stream0)
        del buf686
        del buf687
        del buf688
        buf692 = reinterpret_tensor(buf683, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf683  # reuse
        # Source Nodes: [group_norm_71], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_44.run(buf685, buf689, buf690, arg213_1, arg214_1, buf692, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del arg213_1
        del arg214_1
        del buf689
        del buf690
        # Source Nodes: [group_norm_71, x_297], Original ATen: [aten.convolution, aten.native_group_norm]
        buf693 = extern_kernels.convolution(buf692, arg366_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf693, (8, 3072, 7, 7), (150528, 49, 7, 1))
        del arg366_1
        del buf692
        buf694 = buf675; del buf675  # reuse
        # Source Nodes: [group_norm_71, x_297, x_298], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        triton_poi_fused_convolution_gelu_native_group_norm_45.run(buf693, arg367_1, buf694, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del arg367_1
        del buf693
        # Source Nodes: [group_norm_71, x_297, x_298, x_301], Original ATen: [aten.convolution, aten.gelu, aten.native_group_norm]
        buf695 = extern_kernels.convolution(buf694, arg368_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf695, (8, 768, 7, 7), (37632, 49, 7, 1))
        del arg368_1
        del buf694
        buf696 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        # Source Nodes: [group_norm_71, mul_71, x_297, x_298, x_301, x_306, x_307], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.native_group_norm]
        triton_per_fused_add_convolution_gelu_mean_mul_native_group_norm_49.run(buf685, buf695, arg369_1, arg215_1, buf696, 6144, 49, grid=grid(6144), stream=stream0)
        del arg215_1
        del arg369_1
        del buf685
        del buf695
        buf700 = empty((8, 1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_311], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_50.run(buf696, arg216_1, arg217_1, buf700, 8, 768, grid=grid(8), stream=stream0)
        del arg216_1
        del arg217_1
        del buf696
        buf701 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_316], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg371_1, reinterpret_tensor(buf700, (8, 768), (768, 1), 0), reinterpret_tensor(arg370_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf701)
        del arg370_1
        del arg371_1
        return (buf701, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((96, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((192, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('poolformer_m36', benchmark_compiled_module)
