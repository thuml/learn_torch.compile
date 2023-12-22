
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


# kernel path: /tmp/torchinductor_youkaichao/yz/cyzvhz5eucdaoom3elduvalzszkzha2f7bxfm5echeonhg4bz5sr.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ko/ckowufij3paig7x67vavgowelk2fsgz6chglyo5lz3fpnubxaftt.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlp6yt5s5nx6zhopzy7wxojovniwlikwar7q4xyvkisizlevjv3.py
# Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
# l__mod___stem_0 => convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (768*x2) + (786432*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cdakm5xwgb6hnpyjezduyrojsa2vgqrs46xltnwhfni66t4luuy3.py
# Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___stem_1 => relu
# x => var_mean
triton_red_fused__native_batch_norm_legit_functional_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_relu_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = triton_helpers.maximum(0, tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight,
        )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7ab5yvatfvd2iss4wgve24qoh3kgj5sxoi42dkszdkhw6xurf3.py
# Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___stem_1 => relu
# x => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_relu_4', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 8192.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001220852154804
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefkcf2h27y7u4v2vug2z2h7zuiqm47b7cksr2ezmicjlitngxze.py
# Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___stem_1 => relu
# x => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp3 = tmp1 - tmp2
    tmp5 = 8192.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwmrtpzk2hknozsufjem7oin3dupfcuxxqc37qgisdrl4fp25id.py
# Source Nodes: [add, getattr_getattr_l__mod___blocks___0_____0___fn_1, getattr_getattr_l__mod___blocks___0_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# add => add_10
# getattr_getattr_l__mod___blocks___0_____0___fn_1 => relu_1
# getattr_getattr_l__mod___blocks___0_____0___fn_2 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_add_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), None)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp3 = tmp1 - tmp2
    tmp5 = 8192.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckozctshe6z4l6neenobpanhbibgdcclsylicznnr7dlemvc6vrd.py
# Source Nodes: [l__mod___blocks_31_2, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
# l__mod___blocks_31_2 => relu_64
# x_2 => add_353, add_356, mul_448, mul_454, rsqrt_64, sub_64, var_mean_64
# x_3 => mean
triton_red_fused__native_batch_norm_legit_functional_mean_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_mean_relu_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = triton_helpers.maximum(0, tmp0)
        tmp3 = tmp1 - tmp2
        tmp5 = 8192.0
        tmp6 = tmp4 / tmp5
        tmp7 = 1e-05
        tmp8 = tmp6 + tmp7
        tmp9 = tl.math.rsqrt(tmp8)
        tmp10 = tmp3 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/cantpmwupqlodu4q436sao744lcvyrao5vajuc64kaez4v6jvyye.py
# Source Nodes: [l__mod___blocks_31_2, x_2, x_3, x_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu, aten.view]
# l__mod___blocks_31_2 => relu_64
# x_2 => add_353, add_356, mul_448, mul_454, rsqrt_64, sub_64, var_mean_64
# x_3 => mean
# x_5 => view
triton_per_fused__native_batch_norm_legit_functional_mean_relu_view_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_relu_view_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (6144*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sldbznp5aigrxyswcgub6bw2ui27i5wxkhkd7ysj6c3v4zes5l.py
# Source Nodes: [x], Original ATen: [aten.add]
# x => add
triton_poi_fused_add_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458 = args
    args.clear()
    assert_size_stride(primals_1, (768, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (768, ), (1, ))
    assert_size_stride(primals_157, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_169, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_197, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_202, (768, ), (1, ))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_206, (768, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (768, ), (1, ))
    assert_size_stride(primals_212, (768, ), (1, ))
    assert_size_stride(primals_213, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_218, (768, ), (1, ))
    assert_size_stride(primals_219, (768, ), (1, ))
    assert_size_stride(primals_220, (768, ), (1, ))
    assert_size_stride(primals_221, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (768, ), (1, ))
    assert_size_stride(primals_224, (768, ), (1, ))
    assert_size_stride(primals_225, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_226, (768, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_228, (768, ), (1, ))
    assert_size_stride(primals_229, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_230, (768, ), (1, ))
    assert_size_stride(primals_231, (768, ), (1, ))
    assert_size_stride(primals_232, (768, ), (1, ))
    assert_size_stride(primals_233, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_234, (768, ), (1, ))
    assert_size_stride(primals_235, (768, ), (1, ))
    assert_size_stride(primals_236, (768, ), (1, ))
    assert_size_stride(primals_237, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_238, (768, ), (1, ))
    assert_size_stride(primals_239, (768, ), (1, ))
    assert_size_stride(primals_240, (768, ), (1, ))
    assert_size_stride(primals_241, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_242, (768, ), (1, ))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_244, (768, ), (1, ))
    assert_size_stride(primals_245, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_246, (768, ), (1, ))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_248, (768, ), (1, ))
    assert_size_stride(primals_249, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_250, (768, ), (1, ))
    assert_size_stride(primals_251, (768, ), (1, ))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_254, (768, ), (1, ))
    assert_size_stride(primals_255, (768, ), (1, ))
    assert_size_stride(primals_256, (768, ), (1, ))
    assert_size_stride(primals_257, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_260, (768, ), (1, ))
    assert_size_stride(primals_261, (1000, 768), (768, 1))
    assert_size_stride(primals_262, (1000, ), (1, ))
    assert_size_stride(primals_263, (768, ), (1, ))
    assert_size_stride(primals_264, (768, ), (1, ))
    assert_size_stride(primals_265, (), ())
    assert_size_stride(primals_266, (768, ), (1, ))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_268, (), ())
    assert_size_stride(primals_269, (768, ), (1, ))
    assert_size_stride(primals_270, (768, ), (1, ))
    assert_size_stride(primals_271, (), ())
    assert_size_stride(primals_272, (768, ), (1, ))
    assert_size_stride(primals_273, (768, ), (1, ))
    assert_size_stride(primals_274, (), ())
    assert_size_stride(primals_275, (768, ), (1, ))
    assert_size_stride(primals_276, (768, ), (1, ))
    assert_size_stride(primals_277, (), ())
    assert_size_stride(primals_278, (768, ), (1, ))
    assert_size_stride(primals_279, (768, ), (1, ))
    assert_size_stride(primals_280, (), ())
    assert_size_stride(primals_281, (768, ), (1, ))
    assert_size_stride(primals_282, (768, ), (1, ))
    assert_size_stride(primals_283, (), ())
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_285, (768, ), (1, ))
    assert_size_stride(primals_286, (), ())
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_288, (768, ), (1, ))
    assert_size_stride(primals_289, (), ())
    assert_size_stride(primals_290, (768, ), (1, ))
    assert_size_stride(primals_291, (768, ), (1, ))
    assert_size_stride(primals_292, (), ())
    assert_size_stride(primals_293, (768, ), (1, ))
    assert_size_stride(primals_294, (768, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (768, ), (1, ))
    assert_size_stride(primals_297, (768, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (768, ), (1, ))
    assert_size_stride(primals_300, (768, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (768, ), (1, ))
    assert_size_stride(primals_303, (768, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (768, ), (1, ))
    assert_size_stride(primals_306, (768, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (768, ), (1, ))
    assert_size_stride(primals_309, (768, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (768, ), (1, ))
    assert_size_stride(primals_312, (768, ), (1, ))
    assert_size_stride(primals_313, (), ())
    assert_size_stride(primals_314, (768, ), (1, ))
    assert_size_stride(primals_315, (768, ), (1, ))
    assert_size_stride(primals_316, (), ())
    assert_size_stride(primals_317, (768, ), (1, ))
    assert_size_stride(primals_318, (768, ), (1, ))
    assert_size_stride(primals_319, (), ())
    assert_size_stride(primals_320, (768, ), (1, ))
    assert_size_stride(primals_321, (768, ), (1, ))
    assert_size_stride(primals_322, (), ())
    assert_size_stride(primals_323, (768, ), (1, ))
    assert_size_stride(primals_324, (768, ), (1, ))
    assert_size_stride(primals_325, (), ())
    assert_size_stride(primals_326, (768, ), (1, ))
    assert_size_stride(primals_327, (768, ), (1, ))
    assert_size_stride(primals_328, (), ())
    assert_size_stride(primals_329, (768, ), (1, ))
    assert_size_stride(primals_330, (768, ), (1, ))
    assert_size_stride(primals_331, (), ())
    assert_size_stride(primals_332, (768, ), (1, ))
    assert_size_stride(primals_333, (768, ), (1, ))
    assert_size_stride(primals_334, (), ())
    assert_size_stride(primals_335, (768, ), (1, ))
    assert_size_stride(primals_336, (768, ), (1, ))
    assert_size_stride(primals_337, (), ())
    assert_size_stride(primals_338, (768, ), (1, ))
    assert_size_stride(primals_339, (768, ), (1, ))
    assert_size_stride(primals_340, (), ())
    assert_size_stride(primals_341, (768, ), (1, ))
    assert_size_stride(primals_342, (768, ), (1, ))
    assert_size_stride(primals_343, (), ())
    assert_size_stride(primals_344, (768, ), (1, ))
    assert_size_stride(primals_345, (768, ), (1, ))
    assert_size_stride(primals_346, (), ())
    assert_size_stride(primals_347, (768, ), (1, ))
    assert_size_stride(primals_348, (768, ), (1, ))
    assert_size_stride(primals_349, (), ())
    assert_size_stride(primals_350, (768, ), (1, ))
    assert_size_stride(primals_351, (768, ), (1, ))
    assert_size_stride(primals_352, (), ())
    assert_size_stride(primals_353, (768, ), (1, ))
    assert_size_stride(primals_354, (768, ), (1, ))
    assert_size_stride(primals_355, (), ())
    assert_size_stride(primals_356, (768, ), (1, ))
    assert_size_stride(primals_357, (768, ), (1, ))
    assert_size_stride(primals_358, (), ())
    assert_size_stride(primals_359, (768, ), (1, ))
    assert_size_stride(primals_360, (768, ), (1, ))
    assert_size_stride(primals_361, (), ())
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (768, ), (1, ))
    assert_size_stride(primals_364, (), ())
    assert_size_stride(primals_365, (768, ), (1, ))
    assert_size_stride(primals_366, (768, ), (1, ))
    assert_size_stride(primals_367, (), ())
    assert_size_stride(primals_368, (768, ), (1, ))
    assert_size_stride(primals_369, (768, ), (1, ))
    assert_size_stride(primals_370, (), ())
    assert_size_stride(primals_371, (768, ), (1, ))
    assert_size_stride(primals_372, (768, ), (1, ))
    assert_size_stride(primals_373, (), ())
    assert_size_stride(primals_374, (768, ), (1, ))
    assert_size_stride(primals_375, (768, ), (1, ))
    assert_size_stride(primals_376, (), ())
    assert_size_stride(primals_377, (768, ), (1, ))
    assert_size_stride(primals_378, (768, ), (1, ))
    assert_size_stride(primals_379, (), ())
    assert_size_stride(primals_380, (768, ), (1, ))
    assert_size_stride(primals_381, (768, ), (1, ))
    assert_size_stride(primals_382, (), ())
    assert_size_stride(primals_383, (768, ), (1, ))
    assert_size_stride(primals_384, (768, ), (1, ))
    assert_size_stride(primals_385, (), ())
    assert_size_stride(primals_386, (768, ), (1, ))
    assert_size_stride(primals_387, (768, ), (1, ))
    assert_size_stride(primals_388, (), ())
    assert_size_stride(primals_389, (768, ), (1, ))
    assert_size_stride(primals_390, (768, ), (1, ))
    assert_size_stride(primals_391, (), ())
    assert_size_stride(primals_392, (768, ), (1, ))
    assert_size_stride(primals_393, (768, ), (1, ))
    assert_size_stride(primals_394, (), ())
    assert_size_stride(primals_395, (768, ), (1, ))
    assert_size_stride(primals_396, (768, ), (1, ))
    assert_size_stride(primals_397, (), ())
    assert_size_stride(primals_398, (768, ), (1, ))
    assert_size_stride(primals_399, (768, ), (1, ))
    assert_size_stride(primals_400, (), ())
    assert_size_stride(primals_401, (768, ), (1, ))
    assert_size_stride(primals_402, (768, ), (1, ))
    assert_size_stride(primals_403, (), ())
    assert_size_stride(primals_404, (768, ), (1, ))
    assert_size_stride(primals_405, (768, ), (1, ))
    assert_size_stride(primals_406, (), ())
    assert_size_stride(primals_407, (768, ), (1, ))
    assert_size_stride(primals_408, (768, ), (1, ))
    assert_size_stride(primals_409, (), ())
    assert_size_stride(primals_410, (768, ), (1, ))
    assert_size_stride(primals_411, (768, ), (1, ))
    assert_size_stride(primals_412, (), ())
    assert_size_stride(primals_413, (768, ), (1, ))
    assert_size_stride(primals_414, (768, ), (1, ))
    assert_size_stride(primals_415, (), ())
    assert_size_stride(primals_416, (768, ), (1, ))
    assert_size_stride(primals_417, (768, ), (1, ))
    assert_size_stride(primals_418, (), ())
    assert_size_stride(primals_419, (768, ), (1, ))
    assert_size_stride(primals_420, (768, ), (1, ))
    assert_size_stride(primals_421, (), ())
    assert_size_stride(primals_422, (768, ), (1, ))
    assert_size_stride(primals_423, (768, ), (1, ))
    assert_size_stride(primals_424, (), ())
    assert_size_stride(primals_425, (768, ), (1, ))
    assert_size_stride(primals_426, (768, ), (1, ))
    assert_size_stride(primals_427, (), ())
    assert_size_stride(primals_428, (768, ), (1, ))
    assert_size_stride(primals_429, (768, ), (1, ))
    assert_size_stride(primals_430, (), ())
    assert_size_stride(primals_431, (768, ), (1, ))
    assert_size_stride(primals_432, (768, ), (1, ))
    assert_size_stride(primals_433, (), ())
    assert_size_stride(primals_434, (768, ), (1, ))
    assert_size_stride(primals_435, (768, ), (1, ))
    assert_size_stride(primals_436, (), ())
    assert_size_stride(primals_437, (768, ), (1, ))
    assert_size_stride(primals_438, (768, ), (1, ))
    assert_size_stride(primals_439, (), ())
    assert_size_stride(primals_440, (768, ), (1, ))
    assert_size_stride(primals_441, (768, ), (1, ))
    assert_size_stride(primals_442, (), ())
    assert_size_stride(primals_443, (768, ), (1, ))
    assert_size_stride(primals_444, (768, ), (1, ))
    assert_size_stride(primals_445, (), ())
    assert_size_stride(primals_446, (768, ), (1, ))
    assert_size_stride(primals_447, (768, ), (1, ))
    assert_size_stride(primals_448, (), ())
    assert_size_stride(primals_449, (768, ), (1, ))
    assert_size_stride(primals_450, (768, ), (1, ))
    assert_size_stride(primals_451, (), ())
    assert_size_stride(primals_452, (768, ), (1, ))
    assert_size_stride(primals_453, (768, ), (1, ))
    assert_size_stride(primals_454, (), ())
    assert_size_stride(primals_455, (768, ), (1, ))
    assert_size_stride(primals_456, (768, ), (1, ))
    assert_size_stride(primals_457, (), ())
    assert_size_stride(primals_458, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((768, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 2304, 49, grid=grid(2304, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_458, buf1, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_458
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf3 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, primals_2, buf3, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_2
        buf4 = empty_strided((1, 768, 1, 1, 64), (49152, 1, 49152, 49152, 768), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 768, 1, 1, 64), (49152, 1, 49152, 49152, 768), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((1, 768, 1, 1, 64), (49152, 1, 49152, 49152, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf3, buf4, buf5, buf6, 49152, 128, grid=grid(49152), stream=stream0)
        buf7 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf10 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf4, buf5, buf6, primals_263, primals_264, buf7, buf8, buf10, primals_263, primals_264, 768, 64, grid=grid(768), stream=stream0)
        del primals_263
        del primals_264
        buf11 = reinterpret_tensor(buf2, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf2  # reuse
        # Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf3, buf7, buf8, primals_3, primals_4, buf11, 6291456, grid=grid(6291456), stream=stream0)
        del primals_4
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_0], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_5, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf12, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf13 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf12, primals_6, buf13, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_6
        buf14 = buf6; del buf6  # reuse
        buf15 = buf5; del buf5  # reuse
        buf16 = buf4; del buf4  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_1, getattr_getattr_l__mod___blocks___0_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf13, buf14, buf15, buf16, 49152, 128, grid=grid(49152), stream=stream0)
        buf17 = buf8; del buf8  # reuse
        buf18 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf20 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_1, getattr_getattr_l__mod___blocks___0_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf14, buf15, buf16, primals_266, primals_267, buf17, buf18, buf20, primals_266, primals_267, 768, 64, grid=grid(768), stream=stream0)
        del primals_266
        del primals_267
        buf21 = reinterpret_tensor(buf12, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf12  # reuse
        # Source Nodes: [add, getattr_getattr_l__mod___blocks___0_____0___fn_1, getattr_getattr_l__mod___blocks___0_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf13, buf17, buf18, primals_7, primals_8, buf11, buf21, 6291456, grid=grid(6291456), stream=stream0)
        del primals_8
        # Source Nodes: [l__mod___blocks_0_1], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf23 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf22, primals_10, buf23, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_10
        buf24 = buf16; del buf16  # reuse
        buf25 = buf15; del buf15  # reuse
        buf26 = buf14; del buf14  # reuse
        # Source Nodes: [l__mod___blocks_0_2, l__mod___blocks_0_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf23, buf24, buf25, buf26, 49152, 128, grid=grid(49152), stream=stream0)
        buf27 = buf18; del buf18  # reuse
        buf28 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf30 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_2, l__mod___blocks_0_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf24, buf25, buf26, primals_269, primals_270, buf27, buf28, buf30, primals_269, primals_270, 768, 64, grid=grid(768), stream=stream0)
        del primals_269
        del primals_270
        buf31 = reinterpret_tensor(buf22, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf22  # reuse
        # Source Nodes: [l__mod___blocks_0_2, l__mod___blocks_0_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf23, buf27, buf28, primals_11, primals_12, buf31, 6291456, grid=grid(6291456), stream=stream0)
        del primals_12
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_0], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_13, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf32, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf33 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf32, primals_14, buf33, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_14
        buf34 = buf26; del buf26  # reuse
        buf35 = buf25; del buf25  # reuse
        buf36 = buf24; del buf24  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_1, getattr_getattr_l__mod___blocks___1_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf33, buf34, buf35, buf36, 49152, 128, grid=grid(49152), stream=stream0)
        buf37 = buf28; del buf28  # reuse
        buf38 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf40 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_1, getattr_getattr_l__mod___blocks___1_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf34, buf35, buf36, primals_272, primals_273, buf37, buf38, buf40, primals_272, primals_273, 768, 64, grid=grid(768), stream=stream0)
        del primals_272
        del primals_273
        buf41 = reinterpret_tensor(buf32, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf32  # reuse
        # Source Nodes: [add_1, getattr_getattr_l__mod___blocks___1_____0___fn_1, getattr_getattr_l__mod___blocks___1_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf33, buf37, buf38, primals_15, primals_16, buf31, buf41, 6291456, grid=grid(6291456), stream=stream0)
        del primals_16
        # Source Nodes: [l__mod___blocks_1_1], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf43 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf42, primals_18, buf43, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_18
        buf44 = buf36; del buf36  # reuse
        buf45 = buf35; del buf35  # reuse
        buf46 = buf34; del buf34  # reuse
        # Source Nodes: [l__mod___blocks_1_2, l__mod___blocks_1_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf43, buf44, buf45, buf46, 49152, 128, grid=grid(49152), stream=stream0)
        buf47 = buf38; del buf38  # reuse
        buf48 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf50 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_2, l__mod___blocks_1_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf44, buf45, buf46, primals_275, primals_276, buf47, buf48, buf50, primals_275, primals_276, 768, 64, grid=grid(768), stream=stream0)
        del primals_275
        del primals_276
        buf51 = reinterpret_tensor(buf42, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf42  # reuse
        # Source Nodes: [l__mod___blocks_1_2, l__mod___blocks_1_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf43, buf47, buf48, primals_19, primals_20, buf51, 6291456, grid=grid(6291456), stream=stream0)
        del primals_20
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_0], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_21, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf52, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf53 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf52, primals_22, buf53, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_22
        buf54 = buf46; del buf46  # reuse
        buf55 = buf45; del buf45  # reuse
        buf56 = buf44; del buf44  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_1, getattr_getattr_l__mod___blocks___2_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf53, buf54, buf55, buf56, 49152, 128, grid=grid(49152), stream=stream0)
        buf57 = buf48; del buf48  # reuse
        buf58 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf60 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_1, getattr_getattr_l__mod___blocks___2_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf54, buf55, buf56, primals_278, primals_279, buf57, buf58, buf60, primals_278, primals_279, 768, 64, grid=grid(768), stream=stream0)
        del primals_278
        del primals_279
        buf61 = reinterpret_tensor(buf52, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf52  # reuse
        # Source Nodes: [add_2, getattr_getattr_l__mod___blocks___2_____0___fn_1, getattr_getattr_l__mod___blocks___2_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf53, buf57, buf58, primals_23, primals_24, buf51, buf61, 6291456, grid=grid(6291456), stream=stream0)
        del primals_24
        # Source Nodes: [l__mod___blocks_2_1], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf63 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf62, primals_26, buf63, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_26
        buf64 = buf56; del buf56  # reuse
        buf65 = buf55; del buf55  # reuse
        buf66 = buf54; del buf54  # reuse
        # Source Nodes: [l__mod___blocks_2_2, l__mod___blocks_2_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf63, buf64, buf65, buf66, 49152, 128, grid=grid(49152), stream=stream0)
        buf67 = buf58; del buf58  # reuse
        buf68 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf70 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_2, l__mod___blocks_2_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf64, buf65, buf66, primals_281, primals_282, buf67, buf68, buf70, primals_281, primals_282, 768, 64, grid=grid(768), stream=stream0)
        del primals_281
        del primals_282
        buf71 = reinterpret_tensor(buf62, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf62  # reuse
        # Source Nodes: [l__mod___blocks_2_2, l__mod___blocks_2_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf63, buf67, buf68, primals_27, primals_28, buf71, 6291456, grid=grid(6291456), stream=stream0)
        del primals_28
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_0], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_29, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf72, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf73 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf72, primals_30, buf73, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_30
        buf74 = buf66; del buf66  # reuse
        buf75 = buf65; del buf65  # reuse
        buf76 = buf64; del buf64  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_1, getattr_getattr_l__mod___blocks___3_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf73, buf74, buf75, buf76, 49152, 128, grid=grid(49152), stream=stream0)
        buf77 = buf68; del buf68  # reuse
        buf78 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf80 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_1, getattr_getattr_l__mod___blocks___3_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf74, buf75, buf76, primals_284, primals_285, buf77, buf78, buf80, primals_284, primals_285, 768, 64, grid=grid(768), stream=stream0)
        del primals_284
        del primals_285
        buf81 = reinterpret_tensor(buf72, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf72  # reuse
        # Source Nodes: [add_3, getattr_getattr_l__mod___blocks___3_____0___fn_1, getattr_getattr_l__mod___blocks___3_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf73, buf77, buf78, primals_31, primals_32, buf71, buf81, 6291456, grid=grid(6291456), stream=stream0)
        del primals_32
        # Source Nodes: [l__mod___blocks_3_1], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf83 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf82, primals_34, buf83, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_34
        buf84 = buf76; del buf76  # reuse
        buf85 = buf75; del buf75  # reuse
        buf86 = buf74; del buf74  # reuse
        # Source Nodes: [l__mod___blocks_3_2, l__mod___blocks_3_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf83, buf84, buf85, buf86, 49152, 128, grid=grid(49152), stream=stream0)
        buf87 = buf78; del buf78  # reuse
        buf88 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf90 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_2, l__mod___blocks_3_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf84, buf85, buf86, primals_287, primals_288, buf87, buf88, buf90, primals_287, primals_288, 768, 64, grid=grid(768), stream=stream0)
        del primals_287
        del primals_288
        buf91 = reinterpret_tensor(buf82, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf82  # reuse
        # Source Nodes: [l__mod___blocks_3_2, l__mod___blocks_3_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf83, buf87, buf88, primals_35, primals_36, buf91, 6291456, grid=grid(6291456), stream=stream0)
        del primals_36
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_0], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_37, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf92, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf93 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf92, primals_38, buf93, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_38
        buf94 = buf86; del buf86  # reuse
        buf95 = buf85; del buf85  # reuse
        buf96 = buf84; del buf84  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_1, getattr_getattr_l__mod___blocks___4_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf93, buf94, buf95, buf96, 49152, 128, grid=grid(49152), stream=stream0)
        buf97 = buf88; del buf88  # reuse
        buf98 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf100 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_1, getattr_getattr_l__mod___blocks___4_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf94, buf95, buf96, primals_290, primals_291, buf97, buf98, buf100, primals_290, primals_291, 768, 64, grid=grid(768), stream=stream0)
        del primals_290
        del primals_291
        buf101 = reinterpret_tensor(buf92, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf92  # reuse
        # Source Nodes: [add_4, getattr_getattr_l__mod___blocks___4_____0___fn_1, getattr_getattr_l__mod___blocks___4_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf93, buf97, buf98, primals_39, primals_40, buf91, buf101, 6291456, grid=grid(6291456), stream=stream0)
        del primals_40
        # Source Nodes: [l__mod___blocks_4_1], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf103 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf102, primals_42, buf103, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_42
        buf104 = buf96; del buf96  # reuse
        buf105 = buf95; del buf95  # reuse
        buf106 = buf94; del buf94  # reuse
        # Source Nodes: [l__mod___blocks_4_2, l__mod___blocks_4_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf103, buf104, buf105, buf106, 49152, 128, grid=grid(49152), stream=stream0)
        buf107 = buf98; del buf98  # reuse
        buf108 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf110 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_2, l__mod___blocks_4_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf104, buf105, buf106, primals_293, primals_294, buf107, buf108, buf110, primals_293, primals_294, 768, 64, grid=grid(768), stream=stream0)
        del primals_293
        del primals_294
        buf111 = reinterpret_tensor(buf102, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf102  # reuse
        # Source Nodes: [l__mod___blocks_4_2, l__mod___blocks_4_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf103, buf107, buf108, primals_43, primals_44, buf111, 6291456, grid=grid(6291456), stream=stream0)
        del primals_44
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_0], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_45, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf112, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf113 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf112, primals_46, buf113, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_46
        buf114 = buf106; del buf106  # reuse
        buf115 = buf105; del buf105  # reuse
        buf116 = buf104; del buf104  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_1, getattr_getattr_l__mod___blocks___5_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf113, buf114, buf115, buf116, 49152, 128, grid=grid(49152), stream=stream0)
        buf117 = buf108; del buf108  # reuse
        buf118 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf120 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_1, getattr_getattr_l__mod___blocks___5_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf114, buf115, buf116, primals_296, primals_297, buf117, buf118, buf120, primals_296, primals_297, 768, 64, grid=grid(768), stream=stream0)
        del primals_296
        del primals_297
        buf121 = reinterpret_tensor(buf112, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf112  # reuse
        # Source Nodes: [add_5, getattr_getattr_l__mod___blocks___5_____0___fn_1, getattr_getattr_l__mod___blocks___5_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf113, buf117, buf118, primals_47, primals_48, buf111, buf121, 6291456, grid=grid(6291456), stream=stream0)
        del primals_48
        # Source Nodes: [l__mod___blocks_5_1], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf123 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf122, primals_50, buf123, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_50
        buf124 = buf116; del buf116  # reuse
        buf125 = buf115; del buf115  # reuse
        buf126 = buf114; del buf114  # reuse
        # Source Nodes: [l__mod___blocks_5_2, l__mod___blocks_5_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf123, buf124, buf125, buf126, 49152, 128, grid=grid(49152), stream=stream0)
        buf127 = buf118; del buf118  # reuse
        buf128 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf130 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_2, l__mod___blocks_5_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf124, buf125, buf126, primals_299, primals_300, buf127, buf128, buf130, primals_299, primals_300, 768, 64, grid=grid(768), stream=stream0)
        del primals_299
        del primals_300
        buf131 = reinterpret_tensor(buf122, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf122  # reuse
        # Source Nodes: [l__mod___blocks_5_2, l__mod___blocks_5_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf123, buf127, buf128, primals_51, primals_52, buf131, 6291456, grid=grid(6291456), stream=stream0)
        del primals_52
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_0], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_53, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf132, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf133 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf132, primals_54, buf133, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_54
        buf134 = buf126; del buf126  # reuse
        buf135 = buf125; del buf125  # reuse
        buf136 = buf124; del buf124  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_1, getattr_getattr_l__mod___blocks___6_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf133, buf134, buf135, buf136, 49152, 128, grid=grid(49152), stream=stream0)
        buf137 = buf128; del buf128  # reuse
        buf138 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf140 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_1, getattr_getattr_l__mod___blocks___6_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf134, buf135, buf136, primals_302, primals_303, buf137, buf138, buf140, primals_302, primals_303, 768, 64, grid=grid(768), stream=stream0)
        del primals_302
        del primals_303
        buf141 = reinterpret_tensor(buf132, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf132  # reuse
        # Source Nodes: [add_6, getattr_getattr_l__mod___blocks___6_____0___fn_1, getattr_getattr_l__mod___blocks___6_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf133, buf137, buf138, primals_55, primals_56, buf131, buf141, 6291456, grid=grid(6291456), stream=stream0)
        del primals_56
        # Source Nodes: [l__mod___blocks_6_1], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf143 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf142, primals_58, buf143, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_58
        buf144 = buf136; del buf136  # reuse
        buf145 = buf135; del buf135  # reuse
        buf146 = buf134; del buf134  # reuse
        # Source Nodes: [l__mod___blocks_6_2, l__mod___blocks_6_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf143, buf144, buf145, buf146, 49152, 128, grid=grid(49152), stream=stream0)
        buf147 = buf138; del buf138  # reuse
        buf148 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf150 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_2, l__mod___blocks_6_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf144, buf145, buf146, primals_305, primals_306, buf147, buf148, buf150, primals_305, primals_306, 768, 64, grid=grid(768), stream=stream0)
        del primals_305
        del primals_306
        buf151 = reinterpret_tensor(buf142, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf142  # reuse
        # Source Nodes: [l__mod___blocks_6_2, l__mod___blocks_6_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf143, buf147, buf148, primals_59, primals_60, buf151, 6291456, grid=grid(6291456), stream=stream0)
        del primals_60
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_0], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_61, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf152, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf153 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf152, primals_62, buf153, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_62
        buf154 = buf146; del buf146  # reuse
        buf155 = buf145; del buf145  # reuse
        buf156 = buf144; del buf144  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_1, getattr_getattr_l__mod___blocks___7_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf153, buf154, buf155, buf156, 49152, 128, grid=grid(49152), stream=stream0)
        buf157 = buf148; del buf148  # reuse
        buf158 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf160 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_1, getattr_getattr_l__mod___blocks___7_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf154, buf155, buf156, primals_308, primals_309, buf157, buf158, buf160, primals_308, primals_309, 768, 64, grid=grid(768), stream=stream0)
        del primals_308
        del primals_309
        buf161 = reinterpret_tensor(buf152, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf152  # reuse
        # Source Nodes: [add_7, getattr_getattr_l__mod___blocks___7_____0___fn_1, getattr_getattr_l__mod___blocks___7_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf153, buf157, buf158, primals_63, primals_64, buf151, buf161, 6291456, grid=grid(6291456), stream=stream0)
        del primals_64
        # Source Nodes: [l__mod___blocks_7_1], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_65, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf163 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf162, primals_66, buf163, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_66
        buf164 = buf156; del buf156  # reuse
        buf165 = buf155; del buf155  # reuse
        buf166 = buf154; del buf154  # reuse
        # Source Nodes: [l__mod___blocks_7_2, l__mod___blocks_7_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf163, buf164, buf165, buf166, 49152, 128, grid=grid(49152), stream=stream0)
        buf167 = buf158; del buf158  # reuse
        buf168 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf170 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_2, l__mod___blocks_7_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf164, buf165, buf166, primals_311, primals_312, buf167, buf168, buf170, primals_311, primals_312, 768, 64, grid=grid(768), stream=stream0)
        del primals_311
        del primals_312
        buf171 = reinterpret_tensor(buf162, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf162  # reuse
        # Source Nodes: [l__mod___blocks_7_2, l__mod___blocks_7_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf163, buf167, buf168, primals_67, primals_68, buf171, 6291456, grid=grid(6291456), stream=stream0)
        del primals_68
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_0], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_69, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf172, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf173 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf172, primals_70, buf173, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_70
        buf174 = buf166; del buf166  # reuse
        buf175 = buf165; del buf165  # reuse
        buf176 = buf164; del buf164  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_1, getattr_getattr_l__mod___blocks___8_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf173, buf174, buf175, buf176, 49152, 128, grid=grid(49152), stream=stream0)
        buf177 = buf168; del buf168  # reuse
        buf178 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf180 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_1, getattr_getattr_l__mod___blocks___8_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf174, buf175, buf176, primals_314, primals_315, buf177, buf178, buf180, primals_314, primals_315, 768, 64, grid=grid(768), stream=stream0)
        del primals_314
        del primals_315
        buf181 = reinterpret_tensor(buf172, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf172  # reuse
        # Source Nodes: [add_8, getattr_getattr_l__mod___blocks___8_____0___fn_1, getattr_getattr_l__mod___blocks___8_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf173, buf177, buf178, primals_71, primals_72, buf171, buf181, 6291456, grid=grid(6291456), stream=stream0)
        del primals_72
        # Source Nodes: [l__mod___blocks_8_1], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf183 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf182, primals_74, buf183, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_74
        buf184 = buf176; del buf176  # reuse
        buf185 = buf175; del buf175  # reuse
        buf186 = buf174; del buf174  # reuse
        # Source Nodes: [l__mod___blocks_8_2, l__mod___blocks_8_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf183, buf184, buf185, buf186, 49152, 128, grid=grid(49152), stream=stream0)
        buf187 = buf178; del buf178  # reuse
        buf188 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_2, l__mod___blocks_8_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf184, buf185, buf186, primals_317, primals_318, buf187, buf188, buf190, primals_317, primals_318, 768, 64, grid=grid(768), stream=stream0)
        del primals_317
        del primals_318
        buf191 = reinterpret_tensor(buf182, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf182  # reuse
        # Source Nodes: [l__mod___blocks_8_2, l__mod___blocks_8_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf183, buf187, buf188, primals_75, primals_76, buf191, 6291456, grid=grid(6291456), stream=stream0)
        del primals_76
        # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_0], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_77, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf192, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf193 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf192, primals_78, buf193, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_78
        buf194 = buf186; del buf186  # reuse
        buf195 = buf185; del buf185  # reuse
        buf196 = buf184; del buf184  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_1, getattr_getattr_l__mod___blocks___9_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf193, buf194, buf195, buf196, 49152, 128, grid=grid(49152), stream=stream0)
        buf197 = buf188; del buf188  # reuse
        buf198 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf200 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_1, getattr_getattr_l__mod___blocks___9_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf194, buf195, buf196, primals_320, primals_321, buf197, buf198, buf200, primals_320, primals_321, 768, 64, grid=grid(768), stream=stream0)
        del primals_320
        del primals_321
        buf201 = reinterpret_tensor(buf192, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf192  # reuse
        # Source Nodes: [add_9, getattr_getattr_l__mod___blocks___9_____0___fn_1, getattr_getattr_l__mod___blocks___9_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf193, buf197, buf198, primals_79, primals_80, buf191, buf201, 6291456, grid=grid(6291456), stream=stream0)
        del primals_80
        # Source Nodes: [l__mod___blocks_9_1], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf203 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf202, primals_82, buf203, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_82
        buf204 = buf196; del buf196  # reuse
        buf205 = buf195; del buf195  # reuse
        buf206 = buf194; del buf194  # reuse
        # Source Nodes: [l__mod___blocks_9_2, l__mod___blocks_9_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf203, buf204, buf205, buf206, 49152, 128, grid=grid(49152), stream=stream0)
        buf207 = buf198; del buf198  # reuse
        buf208 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf210 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_2, l__mod___blocks_9_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf204, buf205, buf206, primals_323, primals_324, buf207, buf208, buf210, primals_323, primals_324, 768, 64, grid=grid(768), stream=stream0)
        del primals_323
        del primals_324
        buf211 = reinterpret_tensor(buf202, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf202  # reuse
        # Source Nodes: [l__mod___blocks_9_2, l__mod___blocks_9_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf203, buf207, buf208, primals_83, primals_84, buf211, 6291456, grid=grid(6291456), stream=stream0)
        del primals_84
        # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_0], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_85, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf212, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf213 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf212, primals_86, buf213, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_86
        buf214 = buf206; del buf206  # reuse
        buf215 = buf205; del buf205  # reuse
        buf216 = buf204; del buf204  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_1, getattr_getattr_l__mod___blocks___10_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf213, buf214, buf215, buf216, 49152, 128, grid=grid(49152), stream=stream0)
        buf217 = buf208; del buf208  # reuse
        buf218 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf220 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_1, getattr_getattr_l__mod___blocks___10_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf214, buf215, buf216, primals_326, primals_327, buf217, buf218, buf220, primals_326, primals_327, 768, 64, grid=grid(768), stream=stream0)
        del primals_326
        del primals_327
        buf221 = reinterpret_tensor(buf212, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf212  # reuse
        # Source Nodes: [add_10, getattr_getattr_l__mod___blocks___10_____0___fn_1, getattr_getattr_l__mod___blocks___10_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf213, buf217, buf218, primals_87, primals_88, buf211, buf221, 6291456, grid=grid(6291456), stream=stream0)
        del primals_88
        # Source Nodes: [l__mod___blocks_10_1], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_89, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf223 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf222, primals_90, buf223, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_90
        buf224 = buf216; del buf216  # reuse
        buf225 = buf215; del buf215  # reuse
        buf226 = buf214; del buf214  # reuse
        # Source Nodes: [l__mod___blocks_10_2, l__mod___blocks_10_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf223, buf224, buf225, buf226, 49152, 128, grid=grid(49152), stream=stream0)
        buf227 = buf218; del buf218  # reuse
        buf228 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf230 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_2, l__mod___blocks_10_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf224, buf225, buf226, primals_329, primals_330, buf227, buf228, buf230, primals_329, primals_330, 768, 64, grid=grid(768), stream=stream0)
        del primals_329
        del primals_330
        buf231 = reinterpret_tensor(buf222, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf222  # reuse
        # Source Nodes: [l__mod___blocks_10_2, l__mod___blocks_10_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf223, buf227, buf228, primals_91, primals_92, buf231, 6291456, grid=grid(6291456), stream=stream0)
        del primals_92
        # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_0], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_93, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf232, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf233 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf232, primals_94, buf233, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_94
        buf234 = buf226; del buf226  # reuse
        buf235 = buf225; del buf225  # reuse
        buf236 = buf224; del buf224  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_1, getattr_getattr_l__mod___blocks___11_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf233, buf234, buf235, buf236, 49152, 128, grid=grid(49152), stream=stream0)
        buf237 = buf228; del buf228  # reuse
        buf238 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf240 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_1, getattr_getattr_l__mod___blocks___11_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf234, buf235, buf236, primals_332, primals_333, buf237, buf238, buf240, primals_332, primals_333, 768, 64, grid=grid(768), stream=stream0)
        del primals_332
        del primals_333
        buf241 = reinterpret_tensor(buf232, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf232  # reuse
        # Source Nodes: [add_11, getattr_getattr_l__mod___blocks___11_____0___fn_1, getattr_getattr_l__mod___blocks___11_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf233, buf237, buf238, primals_95, primals_96, buf231, buf241, 6291456, grid=grid(6291456), stream=stream0)
        del primals_96
        # Source Nodes: [l__mod___blocks_11_1], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf243 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf242, primals_98, buf243, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_98
        buf244 = buf236; del buf236  # reuse
        buf245 = buf235; del buf235  # reuse
        buf246 = buf234; del buf234  # reuse
        # Source Nodes: [l__mod___blocks_11_2, l__mod___blocks_11_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf243, buf244, buf245, buf246, 49152, 128, grid=grid(49152), stream=stream0)
        buf247 = buf238; del buf238  # reuse
        buf248 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf250 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_2, l__mod___blocks_11_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf244, buf245, buf246, primals_335, primals_336, buf247, buf248, buf250, primals_335, primals_336, 768, 64, grid=grid(768), stream=stream0)
        del primals_335
        del primals_336
        buf251 = reinterpret_tensor(buf242, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf242  # reuse
        # Source Nodes: [l__mod___blocks_11_2, l__mod___blocks_11_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf243, buf247, buf248, primals_99, primals_100, buf251, 6291456, grid=grid(6291456), stream=stream0)
        del primals_100
        # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_0], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_101, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf252, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf253 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf252, primals_102, buf253, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_102
        buf254 = buf246; del buf246  # reuse
        buf255 = buf245; del buf245  # reuse
        buf256 = buf244; del buf244  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_1, getattr_getattr_l__mod___blocks___12_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf253, buf254, buf255, buf256, 49152, 128, grid=grid(49152), stream=stream0)
        buf257 = buf248; del buf248  # reuse
        buf258 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf260 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_1, getattr_getattr_l__mod___blocks___12_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf254, buf255, buf256, primals_338, primals_339, buf257, buf258, buf260, primals_338, primals_339, 768, 64, grid=grid(768), stream=stream0)
        del primals_338
        del primals_339
        buf261 = reinterpret_tensor(buf252, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf252  # reuse
        # Source Nodes: [add_12, getattr_getattr_l__mod___blocks___12_____0___fn_1, getattr_getattr_l__mod___blocks___12_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf253, buf257, buf258, primals_103, primals_104, buf251, buf261, 6291456, grid=grid(6291456), stream=stream0)
        del primals_104
        # Source Nodes: [l__mod___blocks_12_1], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf263 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_12_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf262, primals_106, buf263, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_106
        buf264 = buf256; del buf256  # reuse
        buf265 = buf255; del buf255  # reuse
        buf266 = buf254; del buf254  # reuse
        # Source Nodes: [l__mod___blocks_12_2, l__mod___blocks_12_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf263, buf264, buf265, buf266, 49152, 128, grid=grid(49152), stream=stream0)
        buf267 = buf258; del buf258  # reuse
        buf268 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf270 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_12_2, l__mod___blocks_12_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf264, buf265, buf266, primals_341, primals_342, buf267, buf268, buf270, primals_341, primals_342, 768, 64, grid=grid(768), stream=stream0)
        del primals_341
        del primals_342
        buf271 = reinterpret_tensor(buf262, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf262  # reuse
        # Source Nodes: [l__mod___blocks_12_2, l__mod___blocks_12_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf263, buf267, buf268, primals_107, primals_108, buf271, 6291456, grid=grid(6291456), stream=stream0)
        del primals_108
        # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_0], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_109, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf272, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf273 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf272, primals_110, buf273, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_110
        buf274 = buf266; del buf266  # reuse
        buf275 = buf265; del buf265  # reuse
        buf276 = buf264; del buf264  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_1, getattr_getattr_l__mod___blocks___13_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf273, buf274, buf275, buf276, 49152, 128, grid=grid(49152), stream=stream0)
        buf277 = buf268; del buf268  # reuse
        buf278 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf280 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_1, getattr_getattr_l__mod___blocks___13_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf274, buf275, buf276, primals_344, primals_345, buf277, buf278, buf280, primals_344, primals_345, 768, 64, grid=grid(768), stream=stream0)
        del primals_344
        del primals_345
        buf281 = reinterpret_tensor(buf272, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf272  # reuse
        # Source Nodes: [add_13, getattr_getattr_l__mod___blocks___13_____0___fn_1, getattr_getattr_l__mod___blocks___13_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf273, buf277, buf278, primals_111, primals_112, buf271, buf281, 6291456, grid=grid(6291456), stream=stream0)
        del primals_112
        # Source Nodes: [l__mod___blocks_13_1], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf283 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_13_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf282, primals_114, buf283, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_114
        buf284 = buf276; del buf276  # reuse
        buf285 = buf275; del buf275  # reuse
        buf286 = buf274; del buf274  # reuse
        # Source Nodes: [l__mod___blocks_13_2, l__mod___blocks_13_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf283, buf284, buf285, buf286, 49152, 128, grid=grid(49152), stream=stream0)
        buf287 = buf278; del buf278  # reuse
        buf288 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf290 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_13_2, l__mod___blocks_13_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf284, buf285, buf286, primals_347, primals_348, buf287, buf288, buf290, primals_347, primals_348, 768, 64, grid=grid(768), stream=stream0)
        del primals_347
        del primals_348
        buf291 = reinterpret_tensor(buf282, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf282  # reuse
        # Source Nodes: [l__mod___blocks_13_2, l__mod___blocks_13_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf283, buf287, buf288, primals_115, primals_116, buf291, 6291456, grid=grid(6291456), stream=stream0)
        del primals_116
        # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_0], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_117, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf292, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf293 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf292, primals_118, buf293, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_118
        buf294 = buf286; del buf286  # reuse
        buf295 = buf285; del buf285  # reuse
        buf296 = buf284; del buf284  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_1, getattr_getattr_l__mod___blocks___14_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf293, buf294, buf295, buf296, 49152, 128, grid=grid(49152), stream=stream0)
        buf297 = buf288; del buf288  # reuse
        buf298 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf300 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_1, getattr_getattr_l__mod___blocks___14_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf294, buf295, buf296, primals_350, primals_351, buf297, buf298, buf300, primals_350, primals_351, 768, 64, grid=grid(768), stream=stream0)
        del primals_350
        del primals_351
        buf301 = reinterpret_tensor(buf292, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf292  # reuse
        # Source Nodes: [add_14, getattr_getattr_l__mod___blocks___14_____0___fn_1, getattr_getattr_l__mod___blocks___14_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf293, buf297, buf298, primals_119, primals_120, buf291, buf301, 6291456, grid=grid(6291456), stream=stream0)
        del primals_120
        # Source Nodes: [l__mod___blocks_14_1], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf303 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_14_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf302, primals_122, buf303, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_122
        buf304 = buf296; del buf296  # reuse
        buf305 = buf295; del buf295  # reuse
        buf306 = buf294; del buf294  # reuse
        # Source Nodes: [l__mod___blocks_14_2, l__mod___blocks_14_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf303, buf304, buf305, buf306, 49152, 128, grid=grid(49152), stream=stream0)
        buf307 = buf298; del buf298  # reuse
        buf308 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf310 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_14_2, l__mod___blocks_14_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf304, buf305, buf306, primals_353, primals_354, buf307, buf308, buf310, primals_353, primals_354, 768, 64, grid=grid(768), stream=stream0)
        del primals_353
        del primals_354
        buf311 = reinterpret_tensor(buf302, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf302  # reuse
        # Source Nodes: [l__mod___blocks_14_2, l__mod___blocks_14_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf303, buf307, buf308, primals_123, primals_124, buf311, 6291456, grid=grid(6291456), stream=stream0)
        del primals_124
        # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_0], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_125, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf312, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf313 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf312, primals_126, buf313, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_126
        buf314 = buf306; del buf306  # reuse
        buf315 = buf305; del buf305  # reuse
        buf316 = buf304; del buf304  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_1, getattr_getattr_l__mod___blocks___15_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf313, buf314, buf315, buf316, 49152, 128, grid=grid(49152), stream=stream0)
        buf317 = buf308; del buf308  # reuse
        buf318 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf320 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_1, getattr_getattr_l__mod___blocks___15_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf314, buf315, buf316, primals_356, primals_357, buf317, buf318, buf320, primals_356, primals_357, 768, 64, grid=grid(768), stream=stream0)
        del primals_356
        del primals_357
        buf321 = reinterpret_tensor(buf312, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf312  # reuse
        # Source Nodes: [add_15, getattr_getattr_l__mod___blocks___15_____0___fn_1, getattr_getattr_l__mod___blocks___15_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf313, buf317, buf318, primals_127, primals_128, buf311, buf321, 6291456, grid=grid(6291456), stream=stream0)
        del primals_128
        # Source Nodes: [l__mod___blocks_15_1], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf323 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_15_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf322, primals_130, buf323, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_130
        buf324 = buf316; del buf316  # reuse
        buf325 = buf315; del buf315  # reuse
        buf326 = buf314; del buf314  # reuse
        # Source Nodes: [l__mod___blocks_15_2, l__mod___blocks_15_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf323, buf324, buf325, buf326, 49152, 128, grid=grid(49152), stream=stream0)
        buf327 = buf318; del buf318  # reuse
        buf328 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf330 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_15_2, l__mod___blocks_15_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf324, buf325, buf326, primals_359, primals_360, buf327, buf328, buf330, primals_359, primals_360, 768, 64, grid=grid(768), stream=stream0)
        del primals_359
        del primals_360
        buf331 = reinterpret_tensor(buf322, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf322  # reuse
        # Source Nodes: [l__mod___blocks_15_2, l__mod___blocks_15_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf323, buf327, buf328, primals_131, primals_132, buf331, 6291456, grid=grid(6291456), stream=stream0)
        del primals_132
        # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_0], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_133, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf332, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf333 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf332, primals_134, buf333, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_134
        buf334 = buf326; del buf326  # reuse
        buf335 = buf325; del buf325  # reuse
        buf336 = buf324; del buf324  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_1, getattr_getattr_l__mod___blocks___16_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf333, buf334, buf335, buf336, 49152, 128, grid=grid(49152), stream=stream0)
        buf337 = buf328; del buf328  # reuse
        buf338 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf340 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_1, getattr_getattr_l__mod___blocks___16_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf334, buf335, buf336, primals_362, primals_363, buf337, buf338, buf340, primals_362, primals_363, 768, 64, grid=grid(768), stream=stream0)
        del primals_362
        del primals_363
        buf341 = reinterpret_tensor(buf332, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf332  # reuse
        # Source Nodes: [add_16, getattr_getattr_l__mod___blocks___16_____0___fn_1, getattr_getattr_l__mod___blocks___16_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf333, buf337, buf338, primals_135, primals_136, buf331, buf341, 6291456, grid=grid(6291456), stream=stream0)
        del primals_136
        # Source Nodes: [l__mod___blocks_16_1], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf343 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_16_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf342, primals_138, buf343, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_138
        buf344 = buf336; del buf336  # reuse
        buf345 = buf335; del buf335  # reuse
        buf346 = buf334; del buf334  # reuse
        # Source Nodes: [l__mod___blocks_16_2, l__mod___blocks_16_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf343, buf344, buf345, buf346, 49152, 128, grid=grid(49152), stream=stream0)
        buf347 = buf338; del buf338  # reuse
        buf348 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf350 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_16_2, l__mod___blocks_16_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf344, buf345, buf346, primals_365, primals_366, buf347, buf348, buf350, primals_365, primals_366, 768, 64, grid=grid(768), stream=stream0)
        del primals_365
        del primals_366
        buf351 = reinterpret_tensor(buf342, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf342  # reuse
        # Source Nodes: [l__mod___blocks_16_2, l__mod___blocks_16_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf343, buf347, buf348, primals_139, primals_140, buf351, 6291456, grid=grid(6291456), stream=stream0)
        del primals_140
        # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_0], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_141, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf352, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf353 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf352, primals_142, buf353, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_142
        buf354 = buf346; del buf346  # reuse
        buf355 = buf345; del buf345  # reuse
        buf356 = buf344; del buf344  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_1, getattr_getattr_l__mod___blocks___17_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf353, buf354, buf355, buf356, 49152, 128, grid=grid(49152), stream=stream0)
        buf357 = buf348; del buf348  # reuse
        buf358 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf360 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_1, getattr_getattr_l__mod___blocks___17_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf354, buf355, buf356, primals_368, primals_369, buf357, buf358, buf360, primals_368, primals_369, 768, 64, grid=grid(768), stream=stream0)
        del primals_368
        del primals_369
        buf361 = reinterpret_tensor(buf352, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf352  # reuse
        # Source Nodes: [add_17, getattr_getattr_l__mod___blocks___17_____0___fn_1, getattr_getattr_l__mod___blocks___17_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf353, buf357, buf358, primals_143, primals_144, buf351, buf361, 6291456, grid=grid(6291456), stream=stream0)
        del primals_144
        # Source Nodes: [l__mod___blocks_17_1], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf363 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_17_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf362, primals_146, buf363, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_146
        buf364 = buf356; del buf356  # reuse
        buf365 = buf355; del buf355  # reuse
        buf366 = buf354; del buf354  # reuse
        # Source Nodes: [l__mod___blocks_17_2, l__mod___blocks_17_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf363, buf364, buf365, buf366, 49152, 128, grid=grid(49152), stream=stream0)
        buf367 = buf358; del buf358  # reuse
        buf368 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf370 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_17_2, l__mod___blocks_17_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf364, buf365, buf366, primals_371, primals_372, buf367, buf368, buf370, primals_371, primals_372, 768, 64, grid=grid(768), stream=stream0)
        del primals_371
        del primals_372
        buf371 = reinterpret_tensor(buf362, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf362  # reuse
        # Source Nodes: [l__mod___blocks_17_2, l__mod___blocks_17_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf363, buf367, buf368, primals_147, primals_148, buf371, 6291456, grid=grid(6291456), stream=stream0)
        del primals_148
        # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_0], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_149, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf372, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf373 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf372, primals_150, buf373, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_150
        buf374 = buf366; del buf366  # reuse
        buf375 = buf365; del buf365  # reuse
        buf376 = buf364; del buf364  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_1, getattr_getattr_l__mod___blocks___18_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf373, buf374, buf375, buf376, 49152, 128, grid=grid(49152), stream=stream0)
        buf377 = buf368; del buf368  # reuse
        buf378 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf380 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_1, getattr_getattr_l__mod___blocks___18_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf374, buf375, buf376, primals_374, primals_375, buf377, buf378, buf380, primals_374, primals_375, 768, 64, grid=grid(768), stream=stream0)
        del primals_374
        del primals_375
        buf381 = reinterpret_tensor(buf372, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf372  # reuse
        # Source Nodes: [add_18, getattr_getattr_l__mod___blocks___18_____0___fn_1, getattr_getattr_l__mod___blocks___18_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf373, buf377, buf378, primals_151, primals_152, buf371, buf381, 6291456, grid=grid(6291456), stream=stream0)
        del primals_152
        # Source Nodes: [l__mod___blocks_18_1], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf383 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_18_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf382, primals_154, buf383, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_154
        buf384 = buf376; del buf376  # reuse
        buf385 = buf375; del buf375  # reuse
        buf386 = buf374; del buf374  # reuse
        # Source Nodes: [l__mod___blocks_18_2, l__mod___blocks_18_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf383, buf384, buf385, buf386, 49152, 128, grid=grid(49152), stream=stream0)
        buf387 = buf378; del buf378  # reuse
        buf388 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf390 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_18_2, l__mod___blocks_18_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf384, buf385, buf386, primals_377, primals_378, buf387, buf388, buf390, primals_377, primals_378, 768, 64, grid=grid(768), stream=stream0)
        del primals_377
        del primals_378
        buf391 = reinterpret_tensor(buf382, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf382  # reuse
        # Source Nodes: [l__mod___blocks_18_2, l__mod___blocks_18_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf383, buf387, buf388, primals_155, primals_156, buf391, 6291456, grid=grid(6291456), stream=stream0)
        del primals_156
        # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_0], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_157, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf392, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf393 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf392, primals_158, buf393, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_158
        buf394 = buf386; del buf386  # reuse
        buf395 = buf385; del buf385  # reuse
        buf396 = buf384; del buf384  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_1, getattr_getattr_l__mod___blocks___19_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf393, buf394, buf395, buf396, 49152, 128, grid=grid(49152), stream=stream0)
        buf397 = buf388; del buf388  # reuse
        buf398 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf400 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_1, getattr_getattr_l__mod___blocks___19_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf394, buf395, buf396, primals_380, primals_381, buf397, buf398, buf400, primals_380, primals_381, 768, 64, grid=grid(768), stream=stream0)
        del primals_380
        del primals_381
        buf401 = reinterpret_tensor(buf392, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf392  # reuse
        # Source Nodes: [add_19, getattr_getattr_l__mod___blocks___19_____0___fn_1, getattr_getattr_l__mod___blocks___19_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf393, buf397, buf398, primals_159, primals_160, buf391, buf401, 6291456, grid=grid(6291456), stream=stream0)
        del primals_160
        # Source Nodes: [l__mod___blocks_19_1], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf403 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_19_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf402, primals_162, buf403, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_162
        buf404 = buf396; del buf396  # reuse
        buf405 = buf395; del buf395  # reuse
        buf406 = buf394; del buf394  # reuse
        # Source Nodes: [l__mod___blocks_19_2, l__mod___blocks_19_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf403, buf404, buf405, buf406, 49152, 128, grid=grid(49152), stream=stream0)
        buf407 = buf398; del buf398  # reuse
        buf408 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf410 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_19_2, l__mod___blocks_19_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf404, buf405, buf406, primals_383, primals_384, buf407, buf408, buf410, primals_383, primals_384, 768, 64, grid=grid(768), stream=stream0)
        del primals_383
        del primals_384
        buf411 = reinterpret_tensor(buf402, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf402  # reuse
        # Source Nodes: [l__mod___blocks_19_2, l__mod___blocks_19_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf403, buf407, buf408, primals_163, primals_164, buf411, 6291456, grid=grid(6291456), stream=stream0)
        del primals_164
        # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_0], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_165, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf412, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf413 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf412, primals_166, buf413, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_166
        buf414 = buf406; del buf406  # reuse
        buf415 = buf405; del buf405  # reuse
        buf416 = buf404; del buf404  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_1, getattr_getattr_l__mod___blocks___20_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf413, buf414, buf415, buf416, 49152, 128, grid=grid(49152), stream=stream0)
        buf417 = buf408; del buf408  # reuse
        buf418 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf420 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_1, getattr_getattr_l__mod___blocks___20_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf414, buf415, buf416, primals_386, primals_387, buf417, buf418, buf420, primals_386, primals_387, 768, 64, grid=grid(768), stream=stream0)
        del primals_386
        del primals_387
        buf421 = reinterpret_tensor(buf412, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf412  # reuse
        # Source Nodes: [add_20, getattr_getattr_l__mod___blocks___20_____0___fn_1, getattr_getattr_l__mod___blocks___20_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf413, buf417, buf418, primals_167, primals_168, buf411, buf421, 6291456, grid=grid(6291456), stream=stream0)
        del primals_168
        # Source Nodes: [l__mod___blocks_20_1], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf423 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_20_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf422, primals_170, buf423, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_170
        buf424 = buf416; del buf416  # reuse
        buf425 = buf415; del buf415  # reuse
        buf426 = buf414; del buf414  # reuse
        # Source Nodes: [l__mod___blocks_20_2, l__mod___blocks_20_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf423, buf424, buf425, buf426, 49152, 128, grid=grid(49152), stream=stream0)
        buf427 = buf418; del buf418  # reuse
        buf428 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf430 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_20_2, l__mod___blocks_20_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf424, buf425, buf426, primals_389, primals_390, buf427, buf428, buf430, primals_389, primals_390, 768, 64, grid=grid(768), stream=stream0)
        del primals_389
        del primals_390
        buf431 = reinterpret_tensor(buf422, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf422  # reuse
        # Source Nodes: [l__mod___blocks_20_2, l__mod___blocks_20_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf423, buf427, buf428, primals_171, primals_172, buf431, 6291456, grid=grid(6291456), stream=stream0)
        del primals_172
        # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_0], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_173, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf432, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf433 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf432, primals_174, buf433, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_174
        buf434 = buf426; del buf426  # reuse
        buf435 = buf425; del buf425  # reuse
        buf436 = buf424; del buf424  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_1, getattr_getattr_l__mod___blocks___21_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf433, buf434, buf435, buf436, 49152, 128, grid=grid(49152), stream=stream0)
        buf437 = buf428; del buf428  # reuse
        buf438 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf440 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_1, getattr_getattr_l__mod___blocks___21_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf434, buf435, buf436, primals_392, primals_393, buf437, buf438, buf440, primals_392, primals_393, 768, 64, grid=grid(768), stream=stream0)
        del primals_392
        del primals_393
        buf441 = reinterpret_tensor(buf432, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf432  # reuse
        # Source Nodes: [add_21, getattr_getattr_l__mod___blocks___21_____0___fn_1, getattr_getattr_l__mod___blocks___21_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf433, buf437, buf438, primals_175, primals_176, buf431, buf441, 6291456, grid=grid(6291456), stream=stream0)
        del primals_176
        # Source Nodes: [l__mod___blocks_21_1], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf441, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf443 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_21_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf442, primals_178, buf443, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_178
        buf444 = buf436; del buf436  # reuse
        buf445 = buf435; del buf435  # reuse
        buf446 = buf434; del buf434  # reuse
        # Source Nodes: [l__mod___blocks_21_2, l__mod___blocks_21_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf443, buf444, buf445, buf446, 49152, 128, grid=grid(49152), stream=stream0)
        buf447 = buf438; del buf438  # reuse
        buf448 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf450 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_21_2, l__mod___blocks_21_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf444, buf445, buf446, primals_395, primals_396, buf447, buf448, buf450, primals_395, primals_396, 768, 64, grid=grid(768), stream=stream0)
        del primals_395
        del primals_396
        buf451 = reinterpret_tensor(buf442, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf442  # reuse
        # Source Nodes: [l__mod___blocks_21_2, l__mod___blocks_21_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf443, buf447, buf448, primals_179, primals_180, buf451, 6291456, grid=grid(6291456), stream=stream0)
        del primals_180
        # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_0], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf451, primals_181, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf452, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf453 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf452, primals_182, buf453, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_182
        buf454 = buf446; del buf446  # reuse
        buf455 = buf445; del buf445  # reuse
        buf456 = buf444; del buf444  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_1, getattr_getattr_l__mod___blocks___22_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf453, buf454, buf455, buf456, 49152, 128, grid=grid(49152), stream=stream0)
        buf457 = buf448; del buf448  # reuse
        buf458 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf460 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_1, getattr_getattr_l__mod___blocks___22_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf454, buf455, buf456, primals_398, primals_399, buf457, buf458, buf460, primals_398, primals_399, 768, 64, grid=grid(768), stream=stream0)
        del primals_398
        del primals_399
        buf461 = reinterpret_tensor(buf452, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf452  # reuse
        # Source Nodes: [add_22, getattr_getattr_l__mod___blocks___22_____0___fn_1, getattr_getattr_l__mod___blocks___22_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf453, buf457, buf458, primals_183, primals_184, buf451, buf461, 6291456, grid=grid(6291456), stream=stream0)
        del primals_184
        # Source Nodes: [l__mod___blocks_22_1], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf463 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_22_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf462, primals_186, buf463, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_186
        buf464 = buf456; del buf456  # reuse
        buf465 = buf455; del buf455  # reuse
        buf466 = buf454; del buf454  # reuse
        # Source Nodes: [l__mod___blocks_22_2, l__mod___blocks_22_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf463, buf464, buf465, buf466, 49152, 128, grid=grid(49152), stream=stream0)
        buf467 = buf458; del buf458  # reuse
        buf468 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf470 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_22_2, l__mod___blocks_22_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf464, buf465, buf466, primals_401, primals_402, buf467, buf468, buf470, primals_401, primals_402, 768, 64, grid=grid(768), stream=stream0)
        del primals_401
        del primals_402
        buf471 = reinterpret_tensor(buf462, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf462  # reuse
        # Source Nodes: [l__mod___blocks_22_2, l__mod___blocks_22_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf463, buf467, buf468, primals_187, primals_188, buf471, 6291456, grid=grid(6291456), stream=stream0)
        del primals_188
        # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_0], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, primals_189, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf472, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf473 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf472, primals_190, buf473, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_190
        buf474 = buf466; del buf466  # reuse
        buf475 = buf465; del buf465  # reuse
        buf476 = buf464; del buf464  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_1, getattr_getattr_l__mod___blocks___23_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf473, buf474, buf475, buf476, 49152, 128, grid=grid(49152), stream=stream0)
        buf477 = buf468; del buf468  # reuse
        buf478 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf480 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_1, getattr_getattr_l__mod___blocks___23_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf474, buf475, buf476, primals_404, primals_405, buf477, buf478, buf480, primals_404, primals_405, 768, 64, grid=grid(768), stream=stream0)
        del primals_404
        del primals_405
        buf481 = reinterpret_tensor(buf472, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf472  # reuse
        # Source Nodes: [add_23, getattr_getattr_l__mod___blocks___23_____0___fn_1, getattr_getattr_l__mod___blocks___23_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf473, buf477, buf478, primals_191, primals_192, buf471, buf481, 6291456, grid=grid(6291456), stream=stream0)
        del primals_192
        # Source Nodes: [l__mod___blocks_23_1], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf481, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf483 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_23_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf482, primals_194, buf483, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_194
        buf484 = buf476; del buf476  # reuse
        buf485 = buf475; del buf475  # reuse
        buf486 = buf474; del buf474  # reuse
        # Source Nodes: [l__mod___blocks_23_2, l__mod___blocks_23_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf483, buf484, buf485, buf486, 49152, 128, grid=grid(49152), stream=stream0)
        buf487 = buf478; del buf478  # reuse
        buf488 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf490 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_23_2, l__mod___blocks_23_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf484, buf485, buf486, primals_407, primals_408, buf487, buf488, buf490, primals_407, primals_408, 768, 64, grid=grid(768), stream=stream0)
        del primals_407
        del primals_408
        buf491 = reinterpret_tensor(buf482, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf482  # reuse
        # Source Nodes: [l__mod___blocks_23_2, l__mod___blocks_23_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf483, buf487, buf488, primals_195, primals_196, buf491, 6291456, grid=grid(6291456), stream=stream0)
        del primals_196
        # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_0], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf491, primals_197, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf492, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf493 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf492, primals_198, buf493, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_198
        buf494 = buf486; del buf486  # reuse
        buf495 = buf485; del buf485  # reuse
        buf496 = buf484; del buf484  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_1, getattr_getattr_l__mod___blocks___24_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf493, buf494, buf495, buf496, 49152, 128, grid=grid(49152), stream=stream0)
        buf497 = buf488; del buf488  # reuse
        buf498 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf500 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_1, getattr_getattr_l__mod___blocks___24_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf494, buf495, buf496, primals_410, primals_411, buf497, buf498, buf500, primals_410, primals_411, 768, 64, grid=grid(768), stream=stream0)
        del primals_410
        del primals_411
        buf501 = reinterpret_tensor(buf492, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf492  # reuse
        # Source Nodes: [add_24, getattr_getattr_l__mod___blocks___24_____0___fn_1, getattr_getattr_l__mod___blocks___24_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf493, buf497, buf498, primals_199, primals_200, buf491, buf501, 6291456, grid=grid(6291456), stream=stream0)
        del primals_200
        # Source Nodes: [l__mod___blocks_24_1], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf503 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_24_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf502, primals_202, buf503, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_202
        buf504 = buf496; del buf496  # reuse
        buf505 = buf495; del buf495  # reuse
        buf506 = buf494; del buf494  # reuse
        # Source Nodes: [l__mod___blocks_24_2, l__mod___blocks_24_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf503, buf504, buf505, buf506, 49152, 128, grid=grid(49152), stream=stream0)
        buf507 = buf498; del buf498  # reuse
        buf508 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf510 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_24_2, l__mod___blocks_24_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf504, buf505, buf506, primals_413, primals_414, buf507, buf508, buf510, primals_413, primals_414, 768, 64, grid=grid(768), stream=stream0)
        del primals_413
        del primals_414
        buf511 = reinterpret_tensor(buf502, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf502  # reuse
        # Source Nodes: [l__mod___blocks_24_2, l__mod___blocks_24_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf503, buf507, buf508, primals_203, primals_204, buf511, 6291456, grid=grid(6291456), stream=stream0)
        del primals_204
        # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_0], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, primals_205, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf512, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf513 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf512, primals_206, buf513, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_206
        buf514 = buf506; del buf506  # reuse
        buf515 = buf505; del buf505  # reuse
        buf516 = buf504; del buf504  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_1, getattr_getattr_l__mod___blocks___25_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf513, buf514, buf515, buf516, 49152, 128, grid=grid(49152), stream=stream0)
        buf517 = buf508; del buf508  # reuse
        buf518 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf520 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_1, getattr_getattr_l__mod___blocks___25_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf514, buf515, buf516, primals_416, primals_417, buf517, buf518, buf520, primals_416, primals_417, 768, 64, grid=grid(768), stream=stream0)
        del primals_416
        del primals_417
        buf521 = reinterpret_tensor(buf512, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf512  # reuse
        # Source Nodes: [add_25, getattr_getattr_l__mod___blocks___25_____0___fn_1, getattr_getattr_l__mod___blocks___25_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf513, buf517, buf518, primals_207, primals_208, buf511, buf521, 6291456, grid=grid(6291456), stream=stream0)
        del primals_208
        # Source Nodes: [l__mod___blocks_25_1], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_209, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf523 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_25_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf522, primals_210, buf523, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_210
        buf524 = buf516; del buf516  # reuse
        buf525 = buf515; del buf515  # reuse
        buf526 = buf514; del buf514  # reuse
        # Source Nodes: [l__mod___blocks_25_2, l__mod___blocks_25_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf523, buf524, buf525, buf526, 49152, 128, grid=grid(49152), stream=stream0)
        buf527 = buf518; del buf518  # reuse
        buf528 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf530 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_25_2, l__mod___blocks_25_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf524, buf525, buf526, primals_419, primals_420, buf527, buf528, buf530, primals_419, primals_420, 768, 64, grid=grid(768), stream=stream0)
        del primals_419
        del primals_420
        buf531 = reinterpret_tensor(buf522, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf522  # reuse
        # Source Nodes: [l__mod___blocks_25_2, l__mod___blocks_25_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf523, buf527, buf528, primals_211, primals_212, buf531, 6291456, grid=grid(6291456), stream=stream0)
        del primals_212
        # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_0], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf531, primals_213, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf532, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf533 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf532, primals_214, buf533, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_214
        buf534 = buf526; del buf526  # reuse
        buf535 = buf525; del buf525  # reuse
        buf536 = buf524; del buf524  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_1, getattr_getattr_l__mod___blocks___26_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf533, buf534, buf535, buf536, 49152, 128, grid=grid(49152), stream=stream0)
        buf537 = buf528; del buf528  # reuse
        buf538 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf540 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_1, getattr_getattr_l__mod___blocks___26_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf534, buf535, buf536, primals_422, primals_423, buf537, buf538, buf540, primals_422, primals_423, 768, 64, grid=grid(768), stream=stream0)
        del primals_422
        del primals_423
        buf541 = reinterpret_tensor(buf532, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf532  # reuse
        # Source Nodes: [add_26, getattr_getattr_l__mod___blocks___26_____0___fn_1, getattr_getattr_l__mod___blocks___26_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf533, buf537, buf538, primals_215, primals_216, buf531, buf541, 6291456, grid=grid(6291456), stream=stream0)
        del primals_216
        # Source Nodes: [l__mod___blocks_26_1], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf543 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_26_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf542, primals_218, buf543, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_218
        buf544 = buf536; del buf536  # reuse
        buf545 = buf535; del buf535  # reuse
        buf546 = buf534; del buf534  # reuse
        # Source Nodes: [l__mod___blocks_26_2, l__mod___blocks_26_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf543, buf544, buf545, buf546, 49152, 128, grid=grid(49152), stream=stream0)
        buf547 = buf538; del buf538  # reuse
        buf548 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf550 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_26_2, l__mod___blocks_26_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf544, buf545, buf546, primals_425, primals_426, buf547, buf548, buf550, primals_425, primals_426, 768, 64, grid=grid(768), stream=stream0)
        del primals_425
        del primals_426
        buf551 = reinterpret_tensor(buf542, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf542  # reuse
        # Source Nodes: [l__mod___blocks_26_2, l__mod___blocks_26_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf543, buf547, buf548, primals_219, primals_220, buf551, 6291456, grid=grid(6291456), stream=stream0)
        del primals_220
        # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_0], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_221, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf552, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf553 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf552, primals_222, buf553, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_222
        buf554 = buf546; del buf546  # reuse
        buf555 = buf545; del buf545  # reuse
        buf556 = buf544; del buf544  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_1, getattr_getattr_l__mod___blocks___27_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf553, buf554, buf555, buf556, 49152, 128, grid=grid(49152), stream=stream0)
        buf557 = buf548; del buf548  # reuse
        buf558 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf560 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_1, getattr_getattr_l__mod___blocks___27_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf554, buf555, buf556, primals_428, primals_429, buf557, buf558, buf560, primals_428, primals_429, 768, 64, grid=grid(768), stream=stream0)
        del primals_428
        del primals_429
        buf561 = reinterpret_tensor(buf552, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf552  # reuse
        # Source Nodes: [add_27, getattr_getattr_l__mod___blocks___27_____0___fn_1, getattr_getattr_l__mod___blocks___27_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf553, buf557, buf558, primals_223, primals_224, buf551, buf561, 6291456, grid=grid(6291456), stream=stream0)
        del primals_224
        # Source Nodes: [l__mod___blocks_27_1], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf561, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf563 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_27_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf562, primals_226, buf563, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_226
        buf564 = buf556; del buf556  # reuse
        buf565 = buf555; del buf555  # reuse
        buf566 = buf554; del buf554  # reuse
        # Source Nodes: [l__mod___blocks_27_2, l__mod___blocks_27_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf563, buf564, buf565, buf566, 49152, 128, grid=grid(49152), stream=stream0)
        buf567 = buf558; del buf558  # reuse
        buf568 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf570 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_27_2, l__mod___blocks_27_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf564, buf565, buf566, primals_431, primals_432, buf567, buf568, buf570, primals_431, primals_432, 768, 64, grid=grid(768), stream=stream0)
        del primals_431
        del primals_432
        buf571 = reinterpret_tensor(buf562, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf562  # reuse
        # Source Nodes: [l__mod___blocks_27_2, l__mod___blocks_27_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf563, buf567, buf568, primals_227, primals_228, buf571, 6291456, grid=grid(6291456), stream=stream0)
        del primals_228
        # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_0], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, primals_229, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf572, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf573 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf572, primals_230, buf573, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_230
        buf574 = buf566; del buf566  # reuse
        buf575 = buf565; del buf565  # reuse
        buf576 = buf564; del buf564  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_1, getattr_getattr_l__mod___blocks___28_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf573, buf574, buf575, buf576, 49152, 128, grid=grid(49152), stream=stream0)
        buf577 = buf568; del buf568  # reuse
        buf578 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf580 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_1, getattr_getattr_l__mod___blocks___28_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf574, buf575, buf576, primals_434, primals_435, buf577, buf578, buf580, primals_434, primals_435, 768, 64, grid=grid(768), stream=stream0)
        del primals_434
        del primals_435
        buf581 = reinterpret_tensor(buf572, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf572  # reuse
        # Source Nodes: [add_28, getattr_getattr_l__mod___blocks___28_____0___fn_1, getattr_getattr_l__mod___blocks___28_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf573, buf577, buf578, primals_231, primals_232, buf571, buf581, 6291456, grid=grid(6291456), stream=stream0)
        del primals_232
        # Source Nodes: [l__mod___blocks_28_1], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf581, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf582, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf583 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_28_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf582, primals_234, buf583, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_234
        buf584 = buf576; del buf576  # reuse
        buf585 = buf575; del buf575  # reuse
        buf586 = buf574; del buf574  # reuse
        # Source Nodes: [l__mod___blocks_28_2, l__mod___blocks_28_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf583, buf584, buf585, buf586, 49152, 128, grid=grid(49152), stream=stream0)
        buf587 = buf578; del buf578  # reuse
        buf588 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf590 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_28_2, l__mod___blocks_28_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf584, buf585, buf586, primals_437, primals_438, buf587, buf588, buf590, primals_437, primals_438, 768, 64, grid=grid(768), stream=stream0)
        del primals_437
        del primals_438
        buf591 = reinterpret_tensor(buf582, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf582  # reuse
        # Source Nodes: [l__mod___blocks_28_2, l__mod___blocks_28_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf583, buf587, buf588, primals_235, primals_236, buf591, 6291456, grid=grid(6291456), stream=stream0)
        del primals_236
        # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_0], Original ATen: [aten.convolution]
        buf592 = extern_kernels.convolution(buf591, primals_237, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf592, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf593 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf592, primals_238, buf593, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_238
        buf594 = buf586; del buf586  # reuse
        buf595 = buf585; del buf585  # reuse
        buf596 = buf584; del buf584  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_1, getattr_getattr_l__mod___blocks___29_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf593, buf594, buf595, buf596, 49152, 128, grid=grid(49152), stream=stream0)
        buf597 = buf588; del buf588  # reuse
        buf598 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf600 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_1, getattr_getattr_l__mod___blocks___29_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf594, buf595, buf596, primals_440, primals_441, buf597, buf598, buf600, primals_440, primals_441, 768, 64, grid=grid(768), stream=stream0)
        del primals_440
        del primals_441
        buf601 = reinterpret_tensor(buf592, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf592  # reuse
        # Source Nodes: [add_29, getattr_getattr_l__mod___blocks___29_____0___fn_1, getattr_getattr_l__mod___blocks___29_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf593, buf597, buf598, primals_239, primals_240, buf591, buf601, 6291456, grid=grid(6291456), stream=stream0)
        del primals_240
        # Source Nodes: [l__mod___blocks_29_1], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf601, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf603 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_29_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf602, primals_242, buf603, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_242
        buf604 = buf596; del buf596  # reuse
        buf605 = buf595; del buf595  # reuse
        buf606 = buf594; del buf594  # reuse
        # Source Nodes: [l__mod___blocks_29_2, l__mod___blocks_29_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf603, buf604, buf605, buf606, 49152, 128, grid=grid(49152), stream=stream0)
        buf607 = buf598; del buf598  # reuse
        buf608 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf610 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_29_2, l__mod___blocks_29_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf604, buf605, buf606, primals_443, primals_444, buf607, buf608, buf610, primals_443, primals_444, 768, 64, grid=grid(768), stream=stream0)
        del primals_443
        del primals_444
        buf611 = reinterpret_tensor(buf602, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf602  # reuse
        # Source Nodes: [l__mod___blocks_29_2, l__mod___blocks_29_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf603, buf607, buf608, primals_243, primals_244, buf611, 6291456, grid=grid(6291456), stream=stream0)
        del primals_244
        # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_0], Original ATen: [aten.convolution]
        buf612 = extern_kernels.convolution(buf611, primals_245, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf612, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf613 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf612, primals_246, buf613, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_246
        buf614 = buf606; del buf606  # reuse
        buf615 = buf605; del buf605  # reuse
        buf616 = buf604; del buf604  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_1, getattr_getattr_l__mod___blocks___30_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf613, buf614, buf615, buf616, 49152, 128, grid=grid(49152), stream=stream0)
        buf617 = buf608; del buf608  # reuse
        buf618 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf620 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_1, getattr_getattr_l__mod___blocks___30_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf614, buf615, buf616, primals_446, primals_447, buf617, buf618, buf620, primals_446, primals_447, 768, 64, grid=grid(768), stream=stream0)
        del primals_446
        del primals_447
        buf621 = reinterpret_tensor(buf612, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf612  # reuse
        # Source Nodes: [add_30, getattr_getattr_l__mod___blocks___30_____0___fn_1, getattr_getattr_l__mod___blocks___30_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf613, buf617, buf618, primals_247, primals_248, buf611, buf621, 6291456, grid=grid(6291456), stream=stream0)
        del primals_248
        # Source Nodes: [l__mod___blocks_30_1], Original ATen: [aten.convolution]
        buf622 = extern_kernels.convolution(buf621, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf622, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf623 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_30_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf622, primals_250, buf623, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_250
        buf624 = buf616; del buf616  # reuse
        buf625 = buf615; del buf615  # reuse
        buf626 = buf614; del buf614  # reuse
        # Source Nodes: [l__mod___blocks_30_2, l__mod___blocks_30_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf623, buf624, buf625, buf626, 49152, 128, grid=grid(49152), stream=stream0)
        buf627 = buf618; del buf618  # reuse
        buf628 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf630 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_30_2, l__mod___blocks_30_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf624, buf625, buf626, primals_449, primals_450, buf627, buf628, buf630, primals_449, primals_450, 768, 64, grid=grid(768), stream=stream0)
        del primals_449
        del primals_450
        buf631 = reinterpret_tensor(buf622, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf622  # reuse
        # Source Nodes: [l__mod___blocks_30_2, l__mod___blocks_30_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf623, buf627, buf628, primals_251, primals_252, buf631, 6291456, grid=grid(6291456), stream=stream0)
        del primals_252
        # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_0], Original ATen: [aten.convolution]
        buf632 = extern_kernels.convolution(buf631, primals_253, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf632, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf633 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf632, primals_254, buf633, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del primals_254
        buf634 = buf626; del buf626  # reuse
        buf635 = buf625; del buf625  # reuse
        buf636 = buf624; del buf624  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1, getattr_getattr_l__mod___blocks___31_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf633, buf634, buf635, buf636, 49152, 128, grid=grid(49152), stream=stream0)
        buf637 = buf628; del buf628  # reuse
        buf638 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf640 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1, getattr_getattr_l__mod___blocks___31_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf634, buf635, buf636, primals_452, primals_453, buf637, buf638, buf640, primals_452, primals_453, 768, 64, grid=grid(768), stream=stream0)
        del primals_452
        del primals_453
        buf641 = reinterpret_tensor(buf632, (8, 768, 32, 32), (786432, 1, 24576, 768), 0); del buf632  # reuse
        # Source Nodes: [add_31, getattr_getattr_l__mod___blocks___31_____0___fn_1, getattr_getattr_l__mod___blocks___31_____0___fn_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_6.run(buf633, buf637, buf638, primals_255, primals_256, buf631, buf641, 6291456, grid=grid(6291456), stream=stream0)
        del primals_256
        # Source Nodes: [l__mod___blocks_31_1], Original ATen: [aten.convolution]
        buf642 = extern_kernels.convolution(buf641, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (8, 768, 32, 32), (786432, 1024, 32, 1))
        buf643 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_31_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf642, primals_258, buf643, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf642
        del primals_258
        buf644 = buf636; del buf636  # reuse
        buf645 = buf635; del buf635  # reuse
        buf646 = buf634; del buf634  # reuse
        # Source Nodes: [l__mod___blocks_31_2, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_relu_3.run(buf643, buf644, buf645, buf646, 49152, 128, grid=grid(49152), stream=stream0)
        buf647 = buf638; del buf638  # reuse
        buf648 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf650 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_31_2, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_relu_4.run(buf644, buf645, buf646, primals_455, primals_456, buf647, buf648, buf650, primals_455, primals_456, 768, 64, grid=grid(768), stream=stream0)
        del buf644
        del buf645
        del primals_455
        del primals_456
        buf651 = reinterpret_tensor(buf646, (8, 768, 1, 1, 8), (6144, 1, 49152, 49152, 768), 0); del buf646  # reuse
        # Source Nodes: [l__mod___blocks_31_2, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_mean_relu_7.run(buf643, buf647, buf648, primals_259, primals_260, buf651, 49152, 128, grid=grid(49152), stream=stream0)
        del buf648
        del primals_260
        buf652 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf653 = reinterpret_tensor(buf652, (8, 768), (768, 1), 0); del buf652  # reuse
        # Source Nodes: [l__mod___blocks_31_2, x_2, x_3, x_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu, aten.view]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_view_8.run(buf653, buf651, 6144, 8, grid=grid(6144), stream=stream0)
        del buf651
        buf654 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_262, buf653, reinterpret_tensor(primals_261, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf654)
        del primals_262
        # Source Nodes: [x], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_265, primals_265, 1, grid=grid(1), stream=stream0)
        del primals_265
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_268, primals_268, 1, grid=grid(1), stream=stream0)
        del primals_268
        # Source Nodes: [l__mod___blocks_0_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_271, primals_271, 1, grid=grid(1), stream=stream0)
        del primals_271
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_274, primals_274, 1, grid=grid(1), stream=stream0)
        del primals_274
        # Source Nodes: [l__mod___blocks_1_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_277, primals_277, 1, grid=grid(1), stream=stream0)
        del primals_277
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_280, primals_280, 1, grid=grid(1), stream=stream0)
        del primals_280
        # Source Nodes: [l__mod___blocks_2_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_283, primals_283, 1, grid=grid(1), stream=stream0)
        del primals_283
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_286, primals_286, 1, grid=grid(1), stream=stream0)
        del primals_286
        # Source Nodes: [l__mod___blocks_3_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_289, primals_289, 1, grid=grid(1), stream=stream0)
        del primals_289
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_292, primals_292, 1, grid=grid(1), stream=stream0)
        del primals_292
        # Source Nodes: [l__mod___blocks_4_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_295, primals_295, 1, grid=grid(1), stream=stream0)
        del primals_295
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_298, primals_298, 1, grid=grid(1), stream=stream0)
        del primals_298
        # Source Nodes: [l__mod___blocks_5_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_301, primals_301, 1, grid=grid(1), stream=stream0)
        del primals_301
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_304, primals_304, 1, grid=grid(1), stream=stream0)
        del primals_304
        # Source Nodes: [l__mod___blocks_6_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_307, primals_307, 1, grid=grid(1), stream=stream0)
        del primals_307
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_310, primals_310, 1, grid=grid(1), stream=stream0)
        del primals_310
        # Source Nodes: [l__mod___blocks_7_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_313, primals_313, 1, grid=grid(1), stream=stream0)
        del primals_313
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_316, primals_316, 1, grid=grid(1), stream=stream0)
        del primals_316
        # Source Nodes: [l__mod___blocks_8_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_319, primals_319, 1, grid=grid(1), stream=stream0)
        del primals_319
        # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_322, primals_322, 1, grid=grid(1), stream=stream0)
        del primals_322
        # Source Nodes: [l__mod___blocks_9_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_325, primals_325, 1, grid=grid(1), stream=stream0)
        del primals_325
        # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_328, primals_328, 1, grid=grid(1), stream=stream0)
        del primals_328
        # Source Nodes: [l__mod___blocks_10_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_331, primals_331, 1, grid=grid(1), stream=stream0)
        del primals_331
        # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_334, primals_334, 1, grid=grid(1), stream=stream0)
        del primals_334
        # Source Nodes: [l__mod___blocks_11_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_337, primals_337, 1, grid=grid(1), stream=stream0)
        del primals_337
        # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_340, primals_340, 1, grid=grid(1), stream=stream0)
        del primals_340
        # Source Nodes: [l__mod___blocks_12_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_343, primals_343, 1, grid=grid(1), stream=stream0)
        del primals_343
        # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_346, primals_346, 1, grid=grid(1), stream=stream0)
        del primals_346
        # Source Nodes: [l__mod___blocks_13_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_349, primals_349, 1, grid=grid(1), stream=stream0)
        del primals_349
        # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_352, primals_352, 1, grid=grid(1), stream=stream0)
        del primals_352
        # Source Nodes: [l__mod___blocks_14_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_355, primals_355, 1, grid=grid(1), stream=stream0)
        del primals_355
        # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_358, primals_358, 1, grid=grid(1), stream=stream0)
        del primals_358
        # Source Nodes: [l__mod___blocks_15_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_361, primals_361, 1, grid=grid(1), stream=stream0)
        del primals_361
        # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_364, primals_364, 1, grid=grid(1), stream=stream0)
        del primals_364
        # Source Nodes: [l__mod___blocks_16_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_367, primals_367, 1, grid=grid(1), stream=stream0)
        del primals_367
        # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_370, primals_370, 1, grid=grid(1), stream=stream0)
        del primals_370
        # Source Nodes: [l__mod___blocks_17_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_373, primals_373, 1, grid=grid(1), stream=stream0)
        del primals_373
        # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_376, primals_376, 1, grid=grid(1), stream=stream0)
        del primals_376
        # Source Nodes: [l__mod___blocks_18_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_379, primals_379, 1, grid=grid(1), stream=stream0)
        del primals_379
        # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_382, primals_382, 1, grid=grid(1), stream=stream0)
        del primals_382
        # Source Nodes: [l__mod___blocks_19_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_385, primals_385, 1, grid=grid(1), stream=stream0)
        del primals_385
        # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_388, primals_388, 1, grid=grid(1), stream=stream0)
        del primals_388
        # Source Nodes: [l__mod___blocks_20_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_391, primals_391, 1, grid=grid(1), stream=stream0)
        del primals_391
        # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_394, primals_394, 1, grid=grid(1), stream=stream0)
        del primals_394
        # Source Nodes: [l__mod___blocks_21_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_397, primals_397, 1, grid=grid(1), stream=stream0)
        del primals_397
        # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_400, primals_400, 1, grid=grid(1), stream=stream0)
        del primals_400
        # Source Nodes: [l__mod___blocks_22_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_403, primals_403, 1, grid=grid(1), stream=stream0)
        del primals_403
        # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_406, primals_406, 1, grid=grid(1), stream=stream0)
        del primals_406
        # Source Nodes: [l__mod___blocks_23_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_409, primals_409, 1, grid=grid(1), stream=stream0)
        del primals_409
        # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_412, primals_412, 1, grid=grid(1), stream=stream0)
        del primals_412
        # Source Nodes: [l__mod___blocks_24_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_415, primals_415, 1, grid=grid(1), stream=stream0)
        del primals_415
        # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_418, primals_418, 1, grid=grid(1), stream=stream0)
        del primals_418
        # Source Nodes: [l__mod___blocks_25_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_421, primals_421, 1, grid=grid(1), stream=stream0)
        del primals_421
        # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_424, primals_424, 1, grid=grid(1), stream=stream0)
        del primals_424
        # Source Nodes: [l__mod___blocks_26_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_427, primals_427, 1, grid=grid(1), stream=stream0)
        del primals_427
        # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_430, primals_430, 1, grid=grid(1), stream=stream0)
        del primals_430
        # Source Nodes: [l__mod___blocks_27_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_433, primals_433, 1, grid=grid(1), stream=stream0)
        del primals_433
        # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_436, primals_436, 1, grid=grid(1), stream=stream0)
        del primals_436
        # Source Nodes: [l__mod___blocks_28_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_439, primals_439, 1, grid=grid(1), stream=stream0)
        del primals_439
        # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_442, primals_442, 1, grid=grid(1), stream=stream0)
        del primals_442
        # Source Nodes: [l__mod___blocks_29_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_445, primals_445, 1, grid=grid(1), stream=stream0)
        del primals_445
        # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_448, primals_448, 1, grid=grid(1), stream=stream0)
        del primals_448
        # Source Nodes: [l__mod___blocks_30_3], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_451, primals_451, 1, grid=grid(1), stream=stream0)
        del primals_451
        # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_454, primals_454, 1, grid=grid(1), stream=stream0)
        del primals_454
        # Source Nodes: [x_2], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(primals_457, primals_457, 1, grid=grid(1), stream=stream0)
        del primals_457
        return (buf654, buf0, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, buf1, buf3, buf10, buf11, buf13, buf20, buf21, buf23, buf30, buf31, buf33, buf40, buf41, buf43, buf50, buf51, buf53, buf60, buf61, buf63, buf70, buf71, buf73, buf80, buf81, buf83, buf90, buf91, buf93, buf100, buf101, buf103, buf110, buf111, buf113, buf120, buf121, buf123, buf130, buf131, buf133, buf140, buf141, buf143, buf150, buf151, buf153, buf160, buf161, buf163, buf170, buf171, buf173, buf180, buf181, buf183, buf190, buf191, buf193, buf200, buf201, buf203, buf210, buf211, buf213, buf220, buf221, buf223, buf230, buf231, buf233, buf240, buf241, buf243, buf250, buf251, buf253, buf260, buf261, buf263, buf270, buf271, buf273, buf280, buf281, buf283, buf290, buf291, buf293, buf300, buf301, buf303, buf310, buf311, buf313, buf320, buf321, buf323, buf330, buf331, buf333, buf340, buf341, buf343, buf350, buf351, buf353, buf360, buf361, buf363, buf370, buf371, buf373, buf380, buf381, buf383, buf390, buf391, buf393, buf400, buf401, buf403, buf410, buf411, buf413, buf420, buf421, buf423, buf430, buf431, buf433, buf440, buf441, buf443, buf450, buf451, buf453, buf460, buf461, buf463, buf470, buf471, buf473, buf480, buf481, buf483, buf490, buf491, buf493, buf500, buf501, buf503, buf510, buf511, buf513, buf520, buf521, buf523, buf530, buf531, buf533, buf540, buf541, buf543, buf550, buf551, buf553, buf560, buf561, buf563, buf570, buf571, buf573, buf580, buf581, buf583, buf590, buf591, buf593, buf600, buf601, buf603, buf610, buf611, buf613, buf620, buf621, buf623, buf630, buf631, buf633, buf640, buf641, buf643, buf650, buf653, reinterpret_tensor(primals_261, (1000, 768), (768, 1), 0), reinterpret_tensor(buf647, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf637, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf627, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf617, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf607, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf597, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf587, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf577, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf567, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf557, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf547, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf537, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf527, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf517, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf507, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf497, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf487, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf477, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf467, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf457, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf447, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf437, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf427, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf417, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf407, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf397, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf387, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf377, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf367, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf357, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf347, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf337, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf327, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf317, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf307, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf297, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf287, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf277, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf267, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf257, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf247, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf237, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf227, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf217, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf207, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf197, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf187, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf177, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf167, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf157, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf147, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf137, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf127, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf117, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf107, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf97, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf87, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf77, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf67, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf57, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf47, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf37, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf27, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf17, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf7, (1, 768, 1, 1), (768, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_266 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_269 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_272 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_275 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_278 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_281 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_284 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_287 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_290 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_293 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_296 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_299 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_302 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_305 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_308 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_311 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_314 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_317 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_320 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_323 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_326 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_329 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_332 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_335 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_338 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_341 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_344 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_347 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_350 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_353 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_356 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_359 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_362 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_365 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_368 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_371 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_374 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_377 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_380 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_383 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_386 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_389 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_392 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_395 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_398 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_401 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_404 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_407 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_410 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_413 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_416 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_419 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_422 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_425 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_428 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_431 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_434 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_437 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_440 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_443 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_446 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_449 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_452 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_455 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_458 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convmixer_768_32', benchmark_compiled_module)
