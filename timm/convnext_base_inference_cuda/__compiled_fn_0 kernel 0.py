
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


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdh3hfruxaef6icv5pktywqty73ma3mmkmf4i25jjsxiljnhk6p.py
# Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
# l__mod___stem_0 => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 82944
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
    tmp0 = tl.load(in_ptr0 + (x2 + (82944*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (248832*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpakbvdagktyjvjuk5canfajxlq2bricgdynh5cqmeydm3i2pzf.py
# Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
# l__mod___stem_0 => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (48*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tnsigfexqtkm7vrhemsm3whwkhsqeep4jrgxtjxj7gsuftdohw.py
# Source Nodes: [x_1], Original ATen: [aten.native_layer_norm]
# x_1 => add, add_1, clone, mul, mul_1, rsqrt, sub, var_mean
triton_red_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 41472
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5184
    x1 = (xindex // 5184)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (5184*r2) + (663552*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (x0 + (5184*r2) + (663552*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 128.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-06
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp20, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnr6dnoi4fjvsow25eojdmhsswpj4s2pgb5lozt3qaa3d22gkyz.py
# Source Nodes: [x_10], Original ATen: [aten.gelu]
# x_10 => add_4, erf, mul_4, mul_5, mul_6
triton_poi_fused_gelu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21233664
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


# kernel path: /tmp/torchinductor_youkaichao/by/cbyieavguvzpltv5o7bcndvn5zfv77nezio5g6gksdfrgqdcsbay.py
# Source Nodes: [shortcut_1, x_17], Original ATen: [aten.add, aten.mul]
# shortcut_1 => add_5
# x_17 => mul_7
triton_poi_fused_add_mul_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sn/csncracivmiwoz7cufee6kx4gybrb3vt7ifyeqjtkgjo77vtvswf.py
# Source Nodes: [shortcut_2, x_31], Original ATen: [aten.add, aten.mul]
# shortcut_2 => add_9
# x_31 => mul_13
triton_poi_fused_add_mul_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pw/cpw7vtm33lgotj4eymzfiwqsygmzgugvcbwu6ym2smjkmucllfv2.py
# Source Nodes: [x_49], Original ATen: [aten.native_layer_norm]
# x_49 => add_14, add_15, mul_20, mul_21, rsqrt_4, sub_4, var_mean_4
triton_per_fused_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 41472
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp6 - tmp16
    tmp24 = 128.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2wq6xlv7vglbvd7xvpzakyaxkqm6ysecmpsewmstf2bsrlciyu.py
# Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
# shortcut_3 => convolution_4
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (512*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dddel5qcg6un6oogtqmienxcmjq7lzdyrd3rju3rtct4bkk2ak.py
# Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
# shortcut_3 => convolution_4
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1296
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1296*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (256*x2) + (331776*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/om/com3oblqoovbnc46zfkgyvhuodfbysiqrzdx6xgfhc764yji2t4i.py
# Source Nodes: [x_55], Original ATen: [aten.native_layer_norm]
# x_55 => add_16, add_17, clone_10, mul_22, mul_23, rsqrt_5, sub_5, var_mean_5
triton_red_fused_native_layer_norm_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10368
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1296
    x1 = (xindex // 1296)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1296*r2) + (331776*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (x0 + (1296*r2) + (331776*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 256.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-06
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp20, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktqifffpwt5wccjwyxiz47j4zquzzf34znummqlnbehxyd66hzc.py
# Source Nodes: [x_57], Original ATen: [aten.gelu]
# x_57 => add_18, erf_3, mul_24, mul_25, mul_26
triton_poi_fused_gelu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
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


# kernel path: /tmp/torchinductor_youkaichao/gs/cgscjzh4fwjd6ysutzlfdonu2fipfeoumnb5o2qom74bblz7fs4x.py
# Source Nodes: [shortcut_4, x_64], Original ATen: [aten.add, aten.mul]
# shortcut_4 => add_19
# x_64 => mul_27
triton_poi_fused_add_mul_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3wsbqonojori33luondls5t2sbmrx5vkmdxpag6jr7dmdcvc4u.py
# Source Nodes: [x_96], Original ATen: [aten.native_layer_norm]
# x_96 => add_28, add_29, mul_40, mul_41, rsqrt_8, sub_8, var_mean_8
triton_per_fused_native_layer_norm_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 10368
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 256, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 256.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3i/c3ih6bz7uojegke65cetqwi4ry3vonnfcq26q52767wtm66rbmk3.py
# Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
# shortcut_6 => convolution_8
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (1024*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cviyxrb5s5yuwn73bvhojb5agk2x34oihoezmiglzlfmfdnczxp5.py
# Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
# shortcut_6 => convolution_8
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 324
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (324*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (512*x2) + (165888*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cowlnflkgjxtoi7lcwpq5pkppwtdfwdq43ikfu3zb7gyhtlbj7.py
# Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
# x_102 => clone_19, var_mean_9
triton_red_fused_native_layer_norm_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10368
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 324
    x4 = (xindex // 324)
    x1 = (xindex // 324) % 4
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (324*r3) + (41472*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ag/caghpj6qtkrzp6olyjx2xqdsswm4lgmiqrrizpgnujom4lzvqgg2.py
# Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
# x_102 => clone_19, var_mean_9
triton_per_fused_native_layer_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2592
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 324
    x1 = (xindex // 324)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (324*r2) + (1296*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (324*r2) + (1296*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (324*r2) + (1296*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ab/cabodj6yzubdtmriehm57vvaqt6qfkfboy4p7vurr7lnmtvd3oyz.py
# Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
# x_102 => add_30, add_31, clone_19, mul_42, mul_43, rsqrt_9, sub_9, var_mean_9
triton_poi_fused_native_layer_norm_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2592
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 324
    y1 = (yindex // 324)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (324*x2) + (165888*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/xd/cxd2zgxmzzlwhfxk67qnnhjxagkq2ytq4o6wnlwgo3b4pp2mtq6t.py
# Source Nodes: [x_104], Original ATen: [aten.gelu]
# x_104 => add_32, erf_6, mul_44, mul_45, mul_46
triton_poi_fused_gelu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
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


# kernel path: /tmp/torchinductor_youkaichao/dd/cddruje5hw3phwi4scu2cpybi76u4tc5hhvw633coz3q6uqwvftw.py
# Source Nodes: [shortcut_7, x_111], Original ATen: [aten.add, aten.mul]
# shortcut_7 => add_33
# x_111 => mul_47
triton_poi_fused_add_mul_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/cclh2qe3qf2m36e4ziknsjsqpawcldkjvviuhtqq2dsxkkri37un.py
# Source Nodes: [shortcut_9, x_139], Original ATen: [aten.add, aten.mul]
# shortcut_9 => add_41
# x_139 => mul_59
triton_poi_fused_add_mul_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjsywhhgka67kmg43qzwipdantovsvggjzhetmiymllarc7732z.py
# Source Nodes: [x_479], Original ATen: [aten.native_layer_norm]
# x_479 => add_138, add_139, mul_204, mul_205, rsqrt_36, sub_36, var_mean_36
triton_per_fused_native_layer_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2592
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 512, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 512.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbywrnmi2hl7pbprvbtwsauc3hjkyfdgfvcfrxfi4uhoaq3vlh6.py
# Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
# shortcut_33 => convolution_36
triton_poi_fused_convolution_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (2048*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jn62gkc6nwso3qd5sceb77hvj336zyztqsah3dudck4vtay2sk.py
# Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
# shortcut_33 => convolution_36
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 81
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
    tmp0 = tl.load(in_ptr0 + (x2 + (81*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (1024*x2) + (82944*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4zndfeguybfjy57waxvebmh3djuuaav3mwkrs2eh7g6yep36kps.py
# Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
# x_485 => clone_100, var_mean_37
triton_red_fused_native_layer_norm_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5184
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 81
    x4 = (xindex // 81)
    x1 = (xindex // 81) % 8
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (81*r3) + (10368*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sl/cslm4budlt5lkiqzcwojqoc7axi6rpjh3jdi2tgtxbgwke4fnqxp.py
# Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
# x_485 => clone_100, var_mean_37
triton_per_fused_native_layer_norm_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 648
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 81
    x1 = (xindex // 81)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (81*r2) + (648*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (81*r2) + (648*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (81*r2) + (648*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/po/cpoaac2xqkhetrxevanosd3n5eqloijeuontfmrdri3rxiamuguw.py
# Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
# x_485 => add_140, add_141, clone_100, mul_206, mul_207, rsqrt_37, sub_37, var_mean_37
triton_poi_fused_native_layer_norm_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 648
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 81
    y1 = (yindex // 81)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (81*x2) + (82944*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1024.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chffxikzijfoefrxtwrky32j5gt7uummks5lukoyfpfz3b6bltxj.py
# Source Nodes: [x_487], Original ATen: [aten.gelu]
# x_487 => add_142, erf_33, mul_208, mul_209, mul_210
triton_poi_fused_gelu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
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


# kernel path: /tmp/torchinductor_youkaichao/od/cod5jpuffexbrbeyye43w533vbuhz2zejcsxcba6le3fsz6vafal.py
# Source Nodes: [shortcut_34, x_494], Original ATen: [aten.add, aten.mul]
# shortcut_34 => add_143
# x_494 => mul_211
triton_poi_fused_add_mul_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/col6kaz4xqyzdowvbu57xpgdjjeyo6wn7vbg7waomq7lzs7y4j7s.py
# Source Nodes: [x_522, x_525, x_528], Original ATen: [aten.add, aten.mean, aten.mul]
# x_522 => mul_223
# x_525 => add_151
# x_528 => mean
triton_red_fused_add_mean_mul_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_mul_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 81
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (82944*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (1024*r2) + (82944*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = 81.0
    tmp11 = tmp8 / tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp23r723olutnehixi5si53ycd56z6pvnmibp4eld5gj76nrpbso.py
# Source Nodes: [x_532], Original ATen: [aten.native_layer_norm]
# x_532 => add_152, add_153, mul_224, mul_225, rsqrt_40, sub_40, var_mean_40
triton_per_fused_native_layer_norm_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 1024.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp27, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, ), (1, ))
    assert_size_stride(arg1_1, (128, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
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
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (1024, ), (1, ))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (512, 128), (128, 1))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (128, 512), (512, 1))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (512, 128), (128, 1))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (128, 512), (512, 1))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (512, 128), (128, 1))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (128, 512), (512, 1))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (256, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(arg139_1, (256, ), (1, ))
    assert_size_stride(arg140_1, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (1024, 256), (256, 1))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (256, 1024), (1024, 1))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg147_1, (256, ), (1, ))
    assert_size_stride(arg148_1, (1024, 256), (256, 1))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (256, 1024), (1024, 1))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (1024, 256), (256, 1))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (256, 1024), (1024, 1))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (512, 256, 2, 2), (1024, 4, 2, 1))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (2048, 512), (512, 1))
    assert_size_stride(arg163_1, (2048, ), (1, ))
    assert_size_stride(arg164_1, (512, 2048), (2048, 1))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg167_1, (512, ), (1, ))
    assert_size_stride(arg168_1, (2048, 512), (512, 1))
    assert_size_stride(arg169_1, (2048, ), (1, ))
    assert_size_stride(arg170_1, (512, 2048), (2048, 1))
    assert_size_stride(arg171_1, (512, ), (1, ))
    assert_size_stride(arg172_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (2048, 512), (512, 1))
    assert_size_stride(arg175_1, (2048, ), (1, ))
    assert_size_stride(arg176_1, (512, 2048), (2048, 1))
    assert_size_stride(arg177_1, (512, ), (1, ))
    assert_size_stride(arg178_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (2048, 512), (512, 1))
    assert_size_stride(arg181_1, (2048, ), (1, ))
    assert_size_stride(arg182_1, (512, 2048), (2048, 1))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (2048, 512), (512, 1))
    assert_size_stride(arg187_1, (2048, ), (1, ))
    assert_size_stride(arg188_1, (512, 2048), (2048, 1))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg191_1, (512, ), (1, ))
    assert_size_stride(arg192_1, (2048, 512), (512, 1))
    assert_size_stride(arg193_1, (2048, ), (1, ))
    assert_size_stride(arg194_1, (512, 2048), (2048, 1))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg197_1, (512, ), (1, ))
    assert_size_stride(arg198_1, (2048, 512), (512, 1))
    assert_size_stride(arg199_1, (2048, ), (1, ))
    assert_size_stride(arg200_1, (512, 2048), (2048, 1))
    assert_size_stride(arg201_1, (512, ), (1, ))
    assert_size_stride(arg202_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (2048, 512), (512, 1))
    assert_size_stride(arg205_1, (2048, ), (1, ))
    assert_size_stride(arg206_1, (512, 2048), (2048, 1))
    assert_size_stride(arg207_1, (512, ), (1, ))
    assert_size_stride(arg208_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (2048, 512), (512, 1))
    assert_size_stride(arg211_1, (2048, ), (1, ))
    assert_size_stride(arg212_1, (512, 2048), (2048, 1))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg215_1, (512, ), (1, ))
    assert_size_stride(arg216_1, (2048, 512), (512, 1))
    assert_size_stride(arg217_1, (2048, ), (1, ))
    assert_size_stride(arg218_1, (512, 2048), (2048, 1))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (2048, 512), (512, 1))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (512, 2048), (2048, 1))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg227_1, (512, ), (1, ))
    assert_size_stride(arg228_1, (2048, 512), (512, 1))
    assert_size_stride(arg229_1, (2048, ), (1, ))
    assert_size_stride(arg230_1, (512, 2048), (2048, 1))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg233_1, (512, ), (1, ))
    assert_size_stride(arg234_1, (2048, 512), (512, 1))
    assert_size_stride(arg235_1, (2048, ), (1, ))
    assert_size_stride(arg236_1, (512, 2048), (2048, 1))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (2048, 512), (512, 1))
    assert_size_stride(arg241_1, (2048, ), (1, ))
    assert_size_stride(arg242_1, (512, 2048), (2048, 1))
    assert_size_stride(arg243_1, (512, ), (1, ))
    assert_size_stride(arg244_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg245_1, (512, ), (1, ))
    assert_size_stride(arg246_1, (2048, 512), (512, 1))
    assert_size_stride(arg247_1, (2048, ), (1, ))
    assert_size_stride(arg248_1, (512, 2048), (2048, 1))
    assert_size_stride(arg249_1, (512, ), (1, ))
    assert_size_stride(arg250_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg251_1, (512, ), (1, ))
    assert_size_stride(arg252_1, (2048, 512), (512, 1))
    assert_size_stride(arg253_1, (2048, ), (1, ))
    assert_size_stride(arg254_1, (512, 2048), (2048, 1))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (2048, 512), (512, 1))
    assert_size_stride(arg259_1, (2048, ), (1, ))
    assert_size_stride(arg260_1, (512, 2048), (2048, 1))
    assert_size_stride(arg261_1, (512, ), (1, ))
    assert_size_stride(arg262_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg263_1, (512, ), (1, ))
    assert_size_stride(arg264_1, (2048, 512), (512, 1))
    assert_size_stride(arg265_1, (2048, ), (1, ))
    assert_size_stride(arg266_1, (512, 2048), (2048, 1))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg269_1, (512, ), (1, ))
    assert_size_stride(arg270_1, (2048, 512), (512, 1))
    assert_size_stride(arg271_1, (2048, ), (1, ))
    assert_size_stride(arg272_1, (512, 2048), (2048, 1))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (2048, 512), (512, 1))
    assert_size_stride(arg277_1, (2048, ), (1, ))
    assert_size_stride(arg278_1, (512, 2048), (2048, 1))
    assert_size_stride(arg279_1, (512, ), (1, ))
    assert_size_stride(arg280_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg281_1, (512, ), (1, ))
    assert_size_stride(arg282_1, (2048, 512), (512, 1))
    assert_size_stride(arg283_1, (2048, ), (1, ))
    assert_size_stride(arg284_1, (512, 2048), (2048, 1))
    assert_size_stride(arg285_1, (512, ), (1, ))
    assert_size_stride(arg286_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg287_1, (512, ), (1, ))
    assert_size_stride(arg288_1, (2048, 512), (512, 1))
    assert_size_stride(arg289_1, (2048, ), (1, ))
    assert_size_stride(arg290_1, (512, 2048), (2048, 1))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (2048, 512), (512, 1))
    assert_size_stride(arg295_1, (2048, ), (1, ))
    assert_size_stride(arg296_1, (512, 2048), (2048, 1))
    assert_size_stride(arg297_1, (512, ), (1, ))
    assert_size_stride(arg298_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (2048, 512), (512, 1))
    assert_size_stride(arg301_1, (2048, ), (1, ))
    assert_size_stride(arg302_1, (512, 2048), (2048, 1))
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg305_1, (512, ), (1, ))
    assert_size_stride(arg306_1, (2048, 512), (512, 1))
    assert_size_stride(arg307_1, (2048, ), (1, ))
    assert_size_stride(arg308_1, (512, 2048), (2048, 1))
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg311_1, (512, ), (1, ))
    assert_size_stride(arg312_1, (2048, 512), (512, 1))
    assert_size_stride(arg313_1, (2048, ), (1, ))
    assert_size_stride(arg314_1, (512, 2048), (2048, 1))
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg317_1, (512, ), (1, ))
    assert_size_stride(arg318_1, (2048, 512), (512, 1))
    assert_size_stride(arg319_1, (2048, ), (1, ))
    assert_size_stride(arg320_1, (512, 2048), (2048, 1))
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (1024, 512, 2, 2), (2048, 4, 2, 1))
    assert_size_stride(arg323_1, (1024, ), (1, ))
    assert_size_stride(arg324_1, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg327_1, (4096, ), (1, ))
    assert_size_stride(arg328_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg329_1, (1024, ), (1, ))
    assert_size_stride(arg330_1, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg333_1, (4096, ), (1, ))
    assert_size_stride(arg334_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg337_1, (1024, ), (1, ))
    assert_size_stride(arg338_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg339_1, (4096, ), (1, ))
    assert_size_stride(arg340_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg343_1, (1000, ), (1, ))
    assert_size_stride(arg344_1, (8, 3, 288, 288), (248832, 82944, 288, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 288, 288), (248832, 1, 864, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg344_1, buf0, 24, 82944, grid=grid(24, 82944), stream=stream0)
        del arg344_1
        buf1 = empty_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg118_1, buf1, 384, 16, grid=grid(384, 16), stream=stream0)
        del arg118_1
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del buf0
        del buf1
        buf6 = empty((8, 72, 72, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_2.run(buf2, arg119_1, arg0_1, arg1_1, buf6, 41472, 128, grid=grid(41472), stream=stream0)
        del arg0_1
        del arg119_1
        del arg1_1
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 128, 72, 72), (663552, 1, 9216, 128), 0), arg120_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf7, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg120_1
        buf11 = reinterpret_tensor(buf2, (8, 72, 72, 128), (663552, 9216, 128, 1), 0); del buf2  # reuse
        # Source Nodes: [x_8], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_2.run(buf7, arg121_1, arg2_1, arg3_1, buf11, 41472, 128, grid=grid(41472), stream=stream0)
        del arg121_1
        del arg2_1
        del arg3_1
        del buf7
        buf12 = empty((41472, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (41472, 128), (128, 1), 0), reinterpret_tensor(arg122_1, (128, 512), (1, 128), 0), out=buf12)
        del arg122_1
        buf13 = reinterpret_tensor(buf12, (8, 72, 72, 512), (2654208, 36864, 512, 1), 0); del buf12  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf13, arg123_1, 21233664, grid=grid(21233664), stream=stream0)
        del arg123_1
        buf14 = reinterpret_tensor(buf11, (41472, 128), (128, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (41472, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 128), (1, 512), 0), out=buf14)
        del arg124_1
        buf15 = reinterpret_tensor(buf14, (8, 128, 72, 72), (663552, 1, 9216, 128), 0); del buf14  # reuse
        # Source Nodes: [shortcut_1, x_17], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_4.run(buf15, arg125_1, arg4_1, buf6, 5308416, grid=grid(5308416), stream=stream0)
        del arg125_1
        del arg4_1
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg126_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf16, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg126_1
        buf20 = buf6; del buf6  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_2.run(buf16, arg127_1, arg5_1, arg6_1, buf20, 41472, 128, grid=grid(41472), stream=stream0)
        del arg127_1
        del arg5_1
        del arg6_1
        del buf16
        buf21 = reinterpret_tensor(buf13, (41472, 512), (512, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (41472, 128), (128, 1), 0), reinterpret_tensor(arg128_1, (128, 512), (1, 128), 0), out=buf21)
        del arg128_1
        buf22 = reinterpret_tensor(buf21, (8, 72, 72, 512), (2654208, 36864, 512, 1), 0); del buf21  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf22, arg129_1, 21233664, grid=grid(21233664), stream=stream0)
        del arg129_1
        buf23 = reinterpret_tensor(buf20, (41472, 128), (128, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (41472, 512), (512, 1), 0), reinterpret_tensor(arg130_1, (512, 128), (1, 512), 0), out=buf23)
        del arg130_1
        buf24 = buf15; del buf15  # reuse
        # Source Nodes: [shortcut_2, x_31], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf24, buf23, arg131_1, arg7_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg131_1
        del arg7_1
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg132_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf25, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del arg132_1
        buf29 = reinterpret_tensor(buf23, (8, 72, 72, 128), (663552, 9216, 128, 1), 0); del buf23  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_2.run(buf25, arg133_1, arg8_1, arg9_1, buf29, 41472, 128, grid=grid(41472), stream=stream0)
        del arg133_1
        del arg8_1
        del arg9_1
        buf30 = reinterpret_tensor(buf22, (41472, 512), (512, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (41472, 128), (128, 1), 0), reinterpret_tensor(arg134_1, (128, 512), (1, 128), 0), out=buf30)
        del arg134_1
        buf31 = reinterpret_tensor(buf30, (8, 72, 72, 512), (2654208, 36864, 512, 1), 0); del buf30  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf31, arg135_1, 21233664, grid=grid(21233664), stream=stream0)
        del arg135_1
        buf32 = reinterpret_tensor(buf29, (41472, 128), (128, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (41472, 512), (512, 1), 0), reinterpret_tensor(arg136_1, (512, 128), (1, 512), 0), out=buf32)
        del arg136_1
        del buf31
        buf36 = reinterpret_tensor(buf25, (8, 72, 72, 128), (663552, 9216, 128, 1), 0); del buf25  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf32, arg137_1, arg10_1, buf24, arg11_1, arg12_1, buf36, 41472, 128, grid=grid(41472), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg137_1
        del buf24
        del buf32
        buf37 = empty_strided((256, 128, 2, 2), (512, 1, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(arg138_1, buf37, 32768, 4, grid=grid(32768, 4), stream=stream0)
        del arg138_1
        # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(reinterpret_tensor(buf36, (8, 128, 72, 72), (663552, 1, 9216, 128), 0), buf37, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 256, 36, 36), (331776, 1296, 36, 1))
        del buf37
        buf39 = empty_strided((8, 256, 36, 36), (331776, 1, 9216, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf38, arg139_1, buf39, 2048, 1296, grid=grid(2048, 1296), stream=stream0)
        del arg139_1
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg140_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf40, (8, 256, 36, 36), (331776, 1296, 36, 1))
        del arg140_1
        buf44 = reinterpret_tensor(buf38, (8, 36, 36, 256), (331776, 9216, 256, 1), 0); del buf38  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_9.run(buf40, arg141_1, arg13_1, arg14_1, buf44, 10368, 256, grid=grid(10368), stream=stream0)
        del arg13_1
        del arg141_1
        del arg14_1
        del buf40
        buf45 = empty((10368, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (10368, 256), (256, 1), 0), reinterpret_tensor(arg142_1, (256, 1024), (1, 256), 0), out=buf45)
        del arg142_1
        buf46 = reinterpret_tensor(buf45, (8, 36, 36, 1024), (1327104, 36864, 1024, 1), 0); del buf45  # reuse
        # Source Nodes: [x_57], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf46, arg143_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg143_1
        buf47 = reinterpret_tensor(buf44, (10368, 256), (256, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (10368, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 256), (1, 1024), 0), out=buf47)
        del arg144_1
        buf48 = buf39; del buf39  # reuse
        # Source Nodes: [shortcut_4, x_64], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_11.run(buf48, buf47, arg145_1, arg15_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg145_1
        del arg15_1
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg146_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf49, (8, 256, 36, 36), (331776, 1296, 36, 1))
        del arg146_1
        buf53 = reinterpret_tensor(buf47, (8, 36, 36, 256), (331776, 9216, 256, 1), 0); del buf47  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_9.run(buf49, arg147_1, arg16_1, arg17_1, buf53, 10368, 256, grid=grid(10368), stream=stream0)
        del arg147_1
        del arg16_1
        del arg17_1
        del buf49
        buf54 = reinterpret_tensor(buf46, (10368, 1024), (1024, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (10368, 256), (256, 1), 0), reinterpret_tensor(arg148_1, (256, 1024), (1, 256), 0), out=buf54)
        del arg148_1
        buf55 = reinterpret_tensor(buf54, (8, 36, 36, 1024), (1327104, 36864, 1024, 1), 0); del buf54  # reuse
        # Source Nodes: [x_71], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf55, arg149_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg149_1
        buf56 = reinterpret_tensor(buf53, (10368, 256), (256, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (10368, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 256), (1, 1024), 0), out=buf56)
        del arg150_1
        buf57 = buf48; del buf48  # reuse
        # Source Nodes: [shortcut_5, x_78], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_11.run(buf57, buf56, arg151_1, arg18_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg151_1
        del arg18_1
        # Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg152_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf58, (8, 256, 36, 36), (331776, 1296, 36, 1))
        del arg152_1
        buf62 = reinterpret_tensor(buf56, (8, 36, 36, 256), (331776, 9216, 256, 1), 0); del buf56  # reuse
        # Source Nodes: [x_83], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_9.run(buf58, arg153_1, arg19_1, arg20_1, buf62, 10368, 256, grid=grid(10368), stream=stream0)
        del arg153_1
        del arg19_1
        del arg20_1
        buf63 = reinterpret_tensor(buf55, (10368, 1024), (1024, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf62, (10368, 256), (256, 1), 0), reinterpret_tensor(arg154_1, (256, 1024), (1, 256), 0), out=buf63)
        del arg154_1
        buf64 = reinterpret_tensor(buf63, (8, 36, 36, 1024), (1327104, 36864, 1024, 1), 0); del buf63  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf64, arg155_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg155_1
        buf65 = reinterpret_tensor(buf62, (10368, 256), (256, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (10368, 1024), (1024, 1), 0), reinterpret_tensor(arg156_1, (1024, 256), (1, 1024), 0), out=buf65)
        del arg156_1
        del buf64
        buf69 = reinterpret_tensor(buf58, (8, 36, 36, 256), (331776, 9216, 256, 1), 0); del buf58  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_12.run(buf65, arg157_1, arg21_1, buf57, arg22_1, arg23_1, buf69, 10368, 256, grid=grid(10368), stream=stream0)
        del arg157_1
        del arg21_1
        del arg22_1
        del arg23_1
        del buf57
        del buf65
        buf70 = empty_strided((512, 256, 2, 2), (1024, 1, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg158_1, buf70, 131072, 4, grid=grid(131072, 4), stream=stream0)
        del arg158_1
        # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(reinterpret_tensor(buf69, (8, 256, 36, 36), (331776, 1, 9216, 256), 0), buf70, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 512, 18, 18), (165888, 324, 18, 1))
        del buf70
        buf72 = empty_strided((8, 512, 18, 18), (165888, 1, 9216, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf71, arg159_1, buf72, 4096, 324, grid=grid(4096, 324), stream=stream0)
        del arg159_1
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg160_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf73, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg160_1
        buf74 = empty_strided((8, 18, 18, 1, 4), (1296, 18, 1, 10368, 324), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((8, 18, 18, 1, 4), (1296, 18, 1, 10368, 324), device='cuda', dtype=torch.float32)
        buf76 = empty_strided((8, 18, 18, 1, 4), (1296, 18, 1, 10368, 324), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf73, arg161_1, buf74, buf75, buf76, 10368, 128, grid=grid(10368), stream=stream0)
        buf77 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf74, buf75, buf76, buf77, buf78, 2592, 4, grid=grid(2592), stream=stream0)
        buf80 = reinterpret_tensor(buf71, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf71  # reuse
        # Source Nodes: [x_102], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf73, arg161_1, buf77, buf78, arg24_1, arg25_1, buf80, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg161_1
        del arg24_1
        del arg25_1
        del buf73
        buf81 = reinterpret_tensor(buf36, (2592, 2048), (2048, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (2592, 512), (512, 1), 0), reinterpret_tensor(arg162_1, (512, 2048), (1, 512), 0), out=buf81)
        del arg162_1
        buf82 = reinterpret_tensor(buf81, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf81  # reuse
        # Source Nodes: [x_104], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf82, arg163_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg163_1
        buf83 = reinterpret_tensor(buf80, (2592, 512), (512, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg164_1, (2048, 512), (1, 2048), 0), out=buf83)
        del arg164_1
        buf84 = buf72; del buf72  # reuse
        # Source Nodes: [shortcut_7, x_111], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf84, buf83, arg165_1, arg26_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg165_1
        del arg26_1
        # Source Nodes: [x_113], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg166_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf85, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg166_1
        buf86 = buf76; del buf76  # reuse
        buf87 = buf75; del buf75  # reuse
        buf88 = buf74; del buf74  # reuse
        # Source Nodes: [x_116], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf85, arg167_1, buf86, buf87, buf88, 10368, 128, grid=grid(10368), stream=stream0)
        buf89 = buf78; del buf78  # reuse
        buf90 = buf77; del buf77  # reuse
        # Source Nodes: [x_116], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf86, buf87, buf88, buf89, buf90, 2592, 4, grid=grid(2592), stream=stream0)
        buf92 = reinterpret_tensor(buf83, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf83  # reuse
        # Source Nodes: [x_116], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf85, arg167_1, buf89, buf90, arg27_1, arg28_1, buf92, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg167_1
        del arg27_1
        del arg28_1
        del buf85
        buf93 = reinterpret_tensor(buf82, (2592, 2048), (2048, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (2592, 512), (512, 1), 0), reinterpret_tensor(arg168_1, (512, 2048), (1, 512), 0), out=buf93)
        del arg168_1
        buf94 = reinterpret_tensor(buf93, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf93  # reuse
        # Source Nodes: [x_118], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf94, arg169_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg169_1
        buf95 = reinterpret_tensor(buf92, (2592, 512), (512, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg170_1, (2048, 512), (1, 2048), 0), out=buf95)
        del arg170_1
        buf96 = buf84; del buf84  # reuse
        # Source Nodes: [shortcut_8, x_125], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf96, buf95, arg171_1, arg29_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg171_1
        del arg29_1
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg172_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf97, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg172_1
        buf98 = buf88; del buf88  # reuse
        buf99 = buf87; del buf87  # reuse
        buf100 = buf86; del buf86  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf97, arg173_1, buf98, buf99, buf100, 10368, 128, grid=grid(10368), stream=stream0)
        buf101 = buf90; del buf90  # reuse
        buf102 = buf89; del buf89  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf98, buf99, buf100, buf101, buf102, 2592, 4, grid=grid(2592), stream=stream0)
        buf104 = reinterpret_tensor(buf95, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf95  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf97, arg173_1, buf101, buf102, arg30_1, arg31_1, buf104, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg173_1
        del arg30_1
        del arg31_1
        del buf97
        buf105 = reinterpret_tensor(buf94, (2592, 2048), (2048, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (2592, 512), (512, 1), 0), reinterpret_tensor(arg174_1, (512, 2048), (1, 512), 0), out=buf105)
        del arg174_1
        buf106 = reinterpret_tensor(buf105, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf105  # reuse
        # Source Nodes: [x_132], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf106, arg175_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg175_1
        buf107 = reinterpret_tensor(buf104, (2592, 512), (512, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg176_1, (2048, 512), (1, 2048), 0), out=buf107)
        del arg176_1
        buf108 = reinterpret_tensor(buf107, (8, 512, 18, 18), (165888, 1, 9216, 512), 0); del buf107  # reuse
        # Source Nodes: [shortcut_9, x_139], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf108, arg177_1, arg32_1, buf96, 1327104, grid=grid(1327104), stream=stream0)
        del arg177_1
        del arg32_1
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg178_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf109, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg178_1
        buf110 = buf99; del buf99  # reuse
        buf111 = buf98; del buf98  # reuse
        buf112 = buf100; del buf100  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf109, arg179_1, buf110, buf111, buf112, 10368, 128, grid=grid(10368), stream=stream0)
        buf113 = buf102; del buf102  # reuse
        buf114 = buf101; del buf101  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf110, buf111, buf112, buf113, buf114, 2592, 4, grid=grid(2592), stream=stream0)
        buf116 = reinterpret_tensor(buf96, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf96  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf109, arg179_1, buf113, buf114, arg33_1, arg34_1, buf116, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg179_1
        del arg33_1
        del arg34_1
        del buf109
        buf117 = reinterpret_tensor(buf106, (2592, 2048), (2048, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (2592, 512), (512, 1), 0), reinterpret_tensor(arg180_1, (512, 2048), (1, 512), 0), out=buf117)
        del arg180_1
        buf118 = reinterpret_tensor(buf117, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf117  # reuse
        # Source Nodes: [x_146], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf118, arg181_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg181_1
        buf119 = reinterpret_tensor(buf116, (2592, 512), (512, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg182_1, (2048, 512), (1, 2048), 0), out=buf119)
        del arg182_1
        buf120 = buf108; del buf108  # reuse
        # Source Nodes: [shortcut_10, x_153], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf120, buf119, arg183_1, arg35_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg183_1
        del arg35_1
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg184_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf121, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg184_1
        buf122 = buf112; del buf112  # reuse
        buf123 = buf111; del buf111  # reuse
        buf124 = buf110; del buf110  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf121, arg185_1, buf122, buf123, buf124, 10368, 128, grid=grid(10368), stream=stream0)
        buf125 = buf114; del buf114  # reuse
        buf126 = buf113; del buf113  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf122, buf123, buf124, buf125, buf126, 2592, 4, grid=grid(2592), stream=stream0)
        buf128 = reinterpret_tensor(buf119, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf119  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf121, arg185_1, buf125, buf126, arg36_1, arg37_1, buf128, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg185_1
        del arg36_1
        del arg37_1
        del buf121
        buf129 = reinterpret_tensor(buf118, (2592, 2048), (2048, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (2592, 512), (512, 1), 0), reinterpret_tensor(arg186_1, (512, 2048), (1, 512), 0), out=buf129)
        del arg186_1
        buf130 = reinterpret_tensor(buf129, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf129  # reuse
        # Source Nodes: [x_160], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf130, arg187_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg187_1
        buf131 = reinterpret_tensor(buf128, (2592, 512), (512, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg188_1, (2048, 512), (1, 2048), 0), out=buf131)
        del arg188_1
        buf132 = buf120; del buf120  # reuse
        # Source Nodes: [shortcut_11, x_167], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf132, buf131, arg189_1, arg38_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg189_1
        del arg38_1
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg190_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf133, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg190_1
        buf134 = buf124; del buf124  # reuse
        buf135 = buf123; del buf123  # reuse
        buf136 = buf122; del buf122  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf133, arg191_1, buf134, buf135, buf136, 10368, 128, grid=grid(10368), stream=stream0)
        buf137 = buf126; del buf126  # reuse
        buf138 = buf125; del buf125  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf134, buf135, buf136, buf137, buf138, 2592, 4, grid=grid(2592), stream=stream0)
        buf140 = reinterpret_tensor(buf131, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf131  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf133, arg191_1, buf137, buf138, arg39_1, arg40_1, buf140, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg191_1
        del arg39_1
        del arg40_1
        del buf133
        buf141 = reinterpret_tensor(buf130, (2592, 2048), (2048, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (2592, 512), (512, 1), 0), reinterpret_tensor(arg192_1, (512, 2048), (1, 512), 0), out=buf141)
        del arg192_1
        buf142 = reinterpret_tensor(buf141, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf141  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf142, arg193_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg193_1
        buf143 = reinterpret_tensor(buf140, (2592, 512), (512, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg194_1, (2048, 512), (1, 2048), 0), out=buf143)
        del arg194_1
        buf144 = buf132; del buf132  # reuse
        # Source Nodes: [shortcut_12, x_181], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf144, buf143, arg195_1, arg41_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg195_1
        del arg41_1
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg196_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf145, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg196_1
        buf146 = buf136; del buf136  # reuse
        buf147 = buf135; del buf135  # reuse
        buf148 = buf134; del buf134  # reuse
        # Source Nodes: [x_186], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf145, arg197_1, buf146, buf147, buf148, 10368, 128, grid=grid(10368), stream=stream0)
        buf149 = buf138; del buf138  # reuse
        buf150 = buf137; del buf137  # reuse
        # Source Nodes: [x_186], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf146, buf147, buf148, buf149, buf150, 2592, 4, grid=grid(2592), stream=stream0)
        buf152 = reinterpret_tensor(buf143, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf143  # reuse
        # Source Nodes: [x_186], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf145, arg197_1, buf149, buf150, arg42_1, arg43_1, buf152, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg197_1
        del arg42_1
        del arg43_1
        del buf145
        buf153 = reinterpret_tensor(buf142, (2592, 2048), (2048, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (2592, 512), (512, 1), 0), reinterpret_tensor(arg198_1, (512, 2048), (1, 512), 0), out=buf153)
        del arg198_1
        buf154 = reinterpret_tensor(buf153, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf153  # reuse
        # Source Nodes: [x_188], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf154, arg199_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg199_1
        buf155 = reinterpret_tensor(buf152, (2592, 512), (512, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg200_1, (2048, 512), (1, 2048), 0), out=buf155)
        del arg200_1
        buf156 = buf144; del buf144  # reuse
        # Source Nodes: [shortcut_13, x_195], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf156, buf155, arg201_1, arg44_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg201_1
        del arg44_1
        # Source Nodes: [x_197], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, arg202_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf157, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg202_1
        buf158 = buf148; del buf148  # reuse
        buf159 = buf147; del buf147  # reuse
        buf160 = buf146; del buf146  # reuse
        # Source Nodes: [x_200], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf157, arg203_1, buf158, buf159, buf160, 10368, 128, grid=grid(10368), stream=stream0)
        buf161 = buf150; del buf150  # reuse
        buf162 = buf149; del buf149  # reuse
        # Source Nodes: [x_200], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf158, buf159, buf160, buf161, buf162, 2592, 4, grid=grid(2592), stream=stream0)
        buf164 = reinterpret_tensor(buf155, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf155  # reuse
        # Source Nodes: [x_200], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf157, arg203_1, buf161, buf162, arg45_1, arg46_1, buf164, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg203_1
        del arg45_1
        del arg46_1
        del buf157
        buf165 = reinterpret_tensor(buf154, (2592, 2048), (2048, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (2592, 512), (512, 1), 0), reinterpret_tensor(arg204_1, (512, 2048), (1, 512), 0), out=buf165)
        del arg204_1
        buf166 = reinterpret_tensor(buf165, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf165  # reuse
        # Source Nodes: [x_202], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf166, arg205_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg205_1
        buf167 = reinterpret_tensor(buf164, (2592, 512), (512, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg206_1, (2048, 512), (1, 2048), 0), out=buf167)
        del arg206_1
        buf168 = buf156; del buf156  # reuse
        # Source Nodes: [shortcut_14, x_209], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf168, buf167, arg207_1, arg47_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg207_1
        del arg47_1
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, arg208_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf169, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg208_1
        buf170 = buf160; del buf160  # reuse
        buf171 = buf159; del buf159  # reuse
        buf172 = buf158; del buf158  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf169, arg209_1, buf170, buf171, buf172, 10368, 128, grid=grid(10368), stream=stream0)
        buf173 = buf162; del buf162  # reuse
        buf174 = buf161; del buf161  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf170, buf171, buf172, buf173, buf174, 2592, 4, grid=grid(2592), stream=stream0)
        buf176 = reinterpret_tensor(buf167, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf167  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf169, arg209_1, buf173, buf174, arg48_1, arg49_1, buf176, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg209_1
        del arg48_1
        del arg49_1
        del buf169
        buf177 = reinterpret_tensor(buf166, (2592, 2048), (2048, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (2592, 512), (512, 1), 0), reinterpret_tensor(arg210_1, (512, 2048), (1, 512), 0), out=buf177)
        del arg210_1
        buf178 = reinterpret_tensor(buf177, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf177  # reuse
        # Source Nodes: [x_216], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf178, arg211_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg211_1
        buf179 = reinterpret_tensor(buf176, (2592, 512), (512, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg212_1, (2048, 512), (1, 2048), 0), out=buf179)
        del arg212_1
        buf180 = buf168; del buf168  # reuse
        # Source Nodes: [shortcut_15, x_223], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf180, buf179, arg213_1, arg50_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg213_1
        del arg50_1
        # Source Nodes: [x_225], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg214_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf181, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg214_1
        buf182 = buf172; del buf172  # reuse
        buf183 = buf171; del buf171  # reuse
        buf184 = buf170; del buf170  # reuse
        # Source Nodes: [x_228], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf181, arg215_1, buf182, buf183, buf184, 10368, 128, grid=grid(10368), stream=stream0)
        buf185 = buf174; del buf174  # reuse
        buf186 = buf173; del buf173  # reuse
        # Source Nodes: [x_228], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf182, buf183, buf184, buf185, buf186, 2592, 4, grid=grid(2592), stream=stream0)
        buf188 = reinterpret_tensor(buf179, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf179  # reuse
        # Source Nodes: [x_228], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf181, arg215_1, buf185, buf186, arg51_1, arg52_1, buf188, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg215_1
        del arg51_1
        del arg52_1
        del buf181
        buf189 = reinterpret_tensor(buf178, (2592, 2048), (2048, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (2592, 512), (512, 1), 0), reinterpret_tensor(arg216_1, (512, 2048), (1, 512), 0), out=buf189)
        del arg216_1
        buf190 = reinterpret_tensor(buf189, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf189  # reuse
        # Source Nodes: [x_230], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf190, arg217_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg217_1
        buf191 = reinterpret_tensor(buf188, (2592, 512), (512, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg218_1, (2048, 512), (1, 2048), 0), out=buf191)
        del arg218_1
        buf192 = buf180; del buf180  # reuse
        # Source Nodes: [shortcut_16, x_237], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf192, buf191, arg219_1, arg53_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg219_1
        del arg53_1
        # Source Nodes: [x_239], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg220_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf193, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg220_1
        buf194 = buf184; del buf184  # reuse
        buf195 = buf183; del buf183  # reuse
        buf196 = buf182; del buf182  # reuse
        # Source Nodes: [x_242], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf193, arg221_1, buf194, buf195, buf196, 10368, 128, grid=grid(10368), stream=stream0)
        buf197 = buf186; del buf186  # reuse
        buf198 = buf185; del buf185  # reuse
        # Source Nodes: [x_242], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf194, buf195, buf196, buf197, buf198, 2592, 4, grid=grid(2592), stream=stream0)
        buf200 = reinterpret_tensor(buf191, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf191  # reuse
        # Source Nodes: [x_242], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf193, arg221_1, buf197, buf198, arg54_1, arg55_1, buf200, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg221_1
        del arg54_1
        del arg55_1
        del buf193
        buf201 = reinterpret_tensor(buf190, (2592, 2048), (2048, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (2592, 512), (512, 1), 0), reinterpret_tensor(arg222_1, (512, 2048), (1, 512), 0), out=buf201)
        del arg222_1
        buf202 = reinterpret_tensor(buf201, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf201  # reuse
        # Source Nodes: [x_244], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf202, arg223_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg223_1
        buf203 = reinterpret_tensor(buf200, (2592, 512), (512, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg224_1, (2048, 512), (1, 2048), 0), out=buf203)
        del arg224_1
        buf204 = buf192; del buf192  # reuse
        # Source Nodes: [shortcut_17, x_251], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf204, buf203, arg225_1, arg56_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg225_1
        del arg56_1
        # Source Nodes: [x_253], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, arg226_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf205, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg226_1
        buf206 = buf196; del buf196  # reuse
        buf207 = buf195; del buf195  # reuse
        buf208 = buf194; del buf194  # reuse
        # Source Nodes: [x_256], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf205, arg227_1, buf206, buf207, buf208, 10368, 128, grid=grid(10368), stream=stream0)
        buf209 = buf198; del buf198  # reuse
        buf210 = buf197; del buf197  # reuse
        # Source Nodes: [x_256], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf206, buf207, buf208, buf209, buf210, 2592, 4, grid=grid(2592), stream=stream0)
        buf212 = reinterpret_tensor(buf203, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf203  # reuse
        # Source Nodes: [x_256], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf205, arg227_1, buf209, buf210, arg57_1, arg58_1, buf212, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg227_1
        del arg57_1
        del arg58_1
        del buf205
        buf213 = reinterpret_tensor(buf202, (2592, 2048), (2048, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (2592, 512), (512, 1), 0), reinterpret_tensor(arg228_1, (512, 2048), (1, 512), 0), out=buf213)
        del arg228_1
        buf214 = reinterpret_tensor(buf213, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf213  # reuse
        # Source Nodes: [x_258], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf214, arg229_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg229_1
        buf215 = reinterpret_tensor(buf212, (2592, 512), (512, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg230_1, (2048, 512), (1, 2048), 0), out=buf215)
        del arg230_1
        buf216 = buf204; del buf204  # reuse
        # Source Nodes: [shortcut_18, x_265], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf216, buf215, arg231_1, arg59_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg231_1
        del arg59_1
        # Source Nodes: [x_267], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, arg232_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf217, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg232_1
        buf218 = buf208; del buf208  # reuse
        buf219 = buf207; del buf207  # reuse
        buf220 = buf206; del buf206  # reuse
        # Source Nodes: [x_270], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf217, arg233_1, buf218, buf219, buf220, 10368, 128, grid=grid(10368), stream=stream0)
        buf221 = buf210; del buf210  # reuse
        buf222 = buf209; del buf209  # reuse
        # Source Nodes: [x_270], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf218, buf219, buf220, buf221, buf222, 2592, 4, grid=grid(2592), stream=stream0)
        buf224 = reinterpret_tensor(buf215, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf215  # reuse
        # Source Nodes: [x_270], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf217, arg233_1, buf221, buf222, arg60_1, arg61_1, buf224, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg233_1
        del arg60_1
        del arg61_1
        del buf217
        buf225 = reinterpret_tensor(buf214, (2592, 2048), (2048, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf224, (2592, 512), (512, 1), 0), reinterpret_tensor(arg234_1, (512, 2048), (1, 512), 0), out=buf225)
        del arg234_1
        buf226 = reinterpret_tensor(buf225, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf225  # reuse
        # Source Nodes: [x_272], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf226, arg235_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg235_1
        buf227 = reinterpret_tensor(buf224, (2592, 512), (512, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg236_1, (2048, 512), (1, 2048), 0), out=buf227)
        del arg236_1
        buf228 = buf216; del buf216  # reuse
        # Source Nodes: [shortcut_19, x_279], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf228, buf227, arg237_1, arg62_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg237_1
        del arg62_1
        # Source Nodes: [x_281], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, arg238_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf229, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg238_1
        buf230 = buf220; del buf220  # reuse
        buf231 = buf219; del buf219  # reuse
        buf232 = buf218; del buf218  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf229, arg239_1, buf230, buf231, buf232, 10368, 128, grid=grid(10368), stream=stream0)
        buf233 = buf222; del buf222  # reuse
        buf234 = buf221; del buf221  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf230, buf231, buf232, buf233, buf234, 2592, 4, grid=grid(2592), stream=stream0)
        buf236 = reinterpret_tensor(buf227, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf227  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf229, arg239_1, buf233, buf234, arg63_1, arg64_1, buf236, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg239_1
        del arg63_1
        del arg64_1
        del buf229
        buf237 = reinterpret_tensor(buf226, (2592, 2048), (2048, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (2592, 512), (512, 1), 0), reinterpret_tensor(arg240_1, (512, 2048), (1, 512), 0), out=buf237)
        del arg240_1
        buf238 = reinterpret_tensor(buf237, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf237  # reuse
        # Source Nodes: [x_286], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf238, arg241_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg241_1
        buf239 = reinterpret_tensor(buf236, (2592, 512), (512, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg242_1, (2048, 512), (1, 2048), 0), out=buf239)
        del arg242_1
        buf240 = buf228; del buf228  # reuse
        # Source Nodes: [shortcut_20, x_293], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf240, buf239, arg243_1, arg65_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg243_1
        del arg65_1
        # Source Nodes: [x_295], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, arg244_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf241, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg244_1
        buf242 = buf232; del buf232  # reuse
        buf243 = buf231; del buf231  # reuse
        buf244 = buf230; del buf230  # reuse
        # Source Nodes: [x_298], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf241, arg245_1, buf242, buf243, buf244, 10368, 128, grid=grid(10368), stream=stream0)
        buf245 = buf234; del buf234  # reuse
        buf246 = buf233; del buf233  # reuse
        # Source Nodes: [x_298], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf242, buf243, buf244, buf245, buf246, 2592, 4, grid=grid(2592), stream=stream0)
        buf248 = reinterpret_tensor(buf239, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf239  # reuse
        # Source Nodes: [x_298], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf241, arg245_1, buf245, buf246, arg66_1, arg67_1, buf248, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg245_1
        del arg66_1
        del arg67_1
        del buf241
        buf249 = reinterpret_tensor(buf238, (2592, 2048), (2048, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (2592, 512), (512, 1), 0), reinterpret_tensor(arg246_1, (512, 2048), (1, 512), 0), out=buf249)
        del arg246_1
        buf250 = reinterpret_tensor(buf249, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf249  # reuse
        # Source Nodes: [x_300], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf250, arg247_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg247_1
        buf251 = reinterpret_tensor(buf248, (2592, 512), (512, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg248_1, (2048, 512), (1, 2048), 0), out=buf251)
        del arg248_1
        buf252 = buf240; del buf240  # reuse
        # Source Nodes: [shortcut_21, x_307], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf252, buf251, arg249_1, arg68_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg249_1
        del arg68_1
        # Source Nodes: [x_309], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, arg250_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf253, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg250_1
        buf254 = buf244; del buf244  # reuse
        buf255 = buf243; del buf243  # reuse
        buf256 = buf242; del buf242  # reuse
        # Source Nodes: [x_312], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf253, arg251_1, buf254, buf255, buf256, 10368, 128, grid=grid(10368), stream=stream0)
        buf257 = buf246; del buf246  # reuse
        buf258 = buf245; del buf245  # reuse
        # Source Nodes: [x_312], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf254, buf255, buf256, buf257, buf258, 2592, 4, grid=grid(2592), stream=stream0)
        buf260 = reinterpret_tensor(buf251, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf251  # reuse
        # Source Nodes: [x_312], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf253, arg251_1, buf257, buf258, arg69_1, arg70_1, buf260, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg251_1
        del arg69_1
        del arg70_1
        del buf253
        buf261 = reinterpret_tensor(buf250, (2592, 2048), (2048, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (2592, 512), (512, 1), 0), reinterpret_tensor(arg252_1, (512, 2048), (1, 512), 0), out=buf261)
        del arg252_1
        buf262 = reinterpret_tensor(buf261, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf261  # reuse
        # Source Nodes: [x_314], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf262, arg253_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg253_1
        buf263 = reinterpret_tensor(buf260, (2592, 512), (512, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf262, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg254_1, (2048, 512), (1, 2048), 0), out=buf263)
        del arg254_1
        buf264 = buf252; del buf252  # reuse
        # Source Nodes: [shortcut_22, x_321], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf264, buf263, arg255_1, arg71_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg255_1
        del arg71_1
        # Source Nodes: [x_323], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, arg256_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf265, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg256_1
        buf266 = buf256; del buf256  # reuse
        buf267 = buf255; del buf255  # reuse
        buf268 = buf254; del buf254  # reuse
        # Source Nodes: [x_326], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf265, arg257_1, buf266, buf267, buf268, 10368, 128, grid=grid(10368), stream=stream0)
        buf269 = buf258; del buf258  # reuse
        buf270 = buf257; del buf257  # reuse
        # Source Nodes: [x_326], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf266, buf267, buf268, buf269, buf270, 2592, 4, grid=grid(2592), stream=stream0)
        buf272 = reinterpret_tensor(buf263, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf263  # reuse
        # Source Nodes: [x_326], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf265, arg257_1, buf269, buf270, arg72_1, arg73_1, buf272, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg257_1
        del arg72_1
        del arg73_1
        del buf265
        buf273 = reinterpret_tensor(buf262, (2592, 2048), (2048, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (2592, 512), (512, 1), 0), reinterpret_tensor(arg258_1, (512, 2048), (1, 512), 0), out=buf273)
        del arg258_1
        buf274 = reinterpret_tensor(buf273, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf273  # reuse
        # Source Nodes: [x_328], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf274, arg259_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg259_1
        buf275 = reinterpret_tensor(buf272, (2592, 512), (512, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg260_1, (2048, 512), (1, 2048), 0), out=buf275)
        del arg260_1
        buf276 = buf264; del buf264  # reuse
        # Source Nodes: [shortcut_23, x_335], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf276, buf275, arg261_1, arg74_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg261_1
        del arg74_1
        # Source Nodes: [x_337], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, arg262_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf277, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg262_1
        buf278 = buf268; del buf268  # reuse
        buf279 = buf267; del buf267  # reuse
        buf280 = buf266; del buf266  # reuse
        # Source Nodes: [x_340], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf277, arg263_1, buf278, buf279, buf280, 10368, 128, grid=grid(10368), stream=stream0)
        buf281 = buf270; del buf270  # reuse
        buf282 = buf269; del buf269  # reuse
        # Source Nodes: [x_340], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf278, buf279, buf280, buf281, buf282, 2592, 4, grid=grid(2592), stream=stream0)
        buf284 = reinterpret_tensor(buf275, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf275  # reuse
        # Source Nodes: [x_340], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf277, arg263_1, buf281, buf282, arg75_1, arg76_1, buf284, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg263_1
        del arg75_1
        del arg76_1
        del buf277
        buf285 = reinterpret_tensor(buf274, (2592, 2048), (2048, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (2592, 512), (512, 1), 0), reinterpret_tensor(arg264_1, (512, 2048), (1, 512), 0), out=buf285)
        del arg264_1
        buf286 = reinterpret_tensor(buf285, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf285  # reuse
        # Source Nodes: [x_342], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf286, arg265_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg265_1
        buf287 = reinterpret_tensor(buf284, (2592, 512), (512, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg266_1, (2048, 512), (1, 2048), 0), out=buf287)
        del arg266_1
        buf288 = buf276; del buf276  # reuse
        # Source Nodes: [shortcut_24, x_349], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf288, buf287, arg267_1, arg77_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg267_1
        del arg77_1
        # Source Nodes: [x_351], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, arg268_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf289, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg268_1
        buf290 = buf280; del buf280  # reuse
        buf291 = buf279; del buf279  # reuse
        buf292 = buf278; del buf278  # reuse
        # Source Nodes: [x_354], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf289, arg269_1, buf290, buf291, buf292, 10368, 128, grid=grid(10368), stream=stream0)
        buf293 = buf282; del buf282  # reuse
        buf294 = buf281; del buf281  # reuse
        # Source Nodes: [x_354], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf290, buf291, buf292, buf293, buf294, 2592, 4, grid=grid(2592), stream=stream0)
        buf296 = reinterpret_tensor(buf287, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf287  # reuse
        # Source Nodes: [x_354], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf289, arg269_1, buf293, buf294, arg78_1, arg79_1, buf296, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg269_1
        del arg78_1
        del arg79_1
        del buf289
        buf297 = reinterpret_tensor(buf286, (2592, 2048), (2048, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (2592, 512), (512, 1), 0), reinterpret_tensor(arg270_1, (512, 2048), (1, 512), 0), out=buf297)
        del arg270_1
        buf298 = reinterpret_tensor(buf297, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf297  # reuse
        # Source Nodes: [x_356], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf298, arg271_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg271_1
        buf299 = reinterpret_tensor(buf296, (2592, 512), (512, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg272_1, (2048, 512), (1, 2048), 0), out=buf299)
        del arg272_1
        buf300 = buf288; del buf288  # reuse
        # Source Nodes: [shortcut_25, x_363], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf300, buf299, arg273_1, arg80_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg273_1
        del arg80_1
        # Source Nodes: [x_365], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, arg274_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf301, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg274_1
        buf302 = buf292; del buf292  # reuse
        buf303 = buf291; del buf291  # reuse
        buf304 = buf290; del buf290  # reuse
        # Source Nodes: [x_368], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf301, arg275_1, buf302, buf303, buf304, 10368, 128, grid=grid(10368), stream=stream0)
        buf305 = buf294; del buf294  # reuse
        buf306 = buf293; del buf293  # reuse
        # Source Nodes: [x_368], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf302, buf303, buf304, buf305, buf306, 2592, 4, grid=grid(2592), stream=stream0)
        buf308 = reinterpret_tensor(buf299, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf299  # reuse
        # Source Nodes: [x_368], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf301, arg275_1, buf305, buf306, arg81_1, arg82_1, buf308, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg275_1
        del arg81_1
        del arg82_1
        del buf301
        buf309 = reinterpret_tensor(buf298, (2592, 2048), (2048, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (2592, 512), (512, 1), 0), reinterpret_tensor(arg276_1, (512, 2048), (1, 512), 0), out=buf309)
        del arg276_1
        buf310 = reinterpret_tensor(buf309, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf309  # reuse
        # Source Nodes: [x_370], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf310, arg277_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg277_1
        buf311 = reinterpret_tensor(buf308, (2592, 512), (512, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg278_1, (2048, 512), (1, 2048), 0), out=buf311)
        del arg278_1
        buf312 = buf300; del buf300  # reuse
        # Source Nodes: [shortcut_26, x_377], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf312, buf311, arg279_1, arg83_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg279_1
        del arg83_1
        # Source Nodes: [x_379], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, arg280_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf313, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg280_1
        buf314 = buf304; del buf304  # reuse
        buf315 = buf303; del buf303  # reuse
        buf316 = buf302; del buf302  # reuse
        # Source Nodes: [x_382], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf313, arg281_1, buf314, buf315, buf316, 10368, 128, grid=grid(10368), stream=stream0)
        buf317 = buf306; del buf306  # reuse
        buf318 = buf305; del buf305  # reuse
        # Source Nodes: [x_382], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf314, buf315, buf316, buf317, buf318, 2592, 4, grid=grid(2592), stream=stream0)
        buf320 = reinterpret_tensor(buf311, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf311  # reuse
        # Source Nodes: [x_382], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf313, arg281_1, buf317, buf318, arg84_1, arg85_1, buf320, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg281_1
        del arg84_1
        del arg85_1
        del buf313
        buf321 = reinterpret_tensor(buf310, (2592, 2048), (2048, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf320, (2592, 512), (512, 1), 0), reinterpret_tensor(arg282_1, (512, 2048), (1, 512), 0), out=buf321)
        del arg282_1
        buf322 = reinterpret_tensor(buf321, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf321  # reuse
        # Source Nodes: [x_384], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf322, arg283_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg283_1
        buf323 = reinterpret_tensor(buf320, (2592, 512), (512, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg284_1, (2048, 512), (1, 2048), 0), out=buf323)
        del arg284_1
        buf324 = buf312; del buf312  # reuse
        # Source Nodes: [shortcut_27, x_391], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf324, buf323, arg285_1, arg86_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg285_1
        del arg86_1
        # Source Nodes: [x_393], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, arg286_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf325, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg286_1
        buf326 = buf316; del buf316  # reuse
        buf327 = buf315; del buf315  # reuse
        buf328 = buf314; del buf314  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf325, arg287_1, buf326, buf327, buf328, 10368, 128, grid=grid(10368), stream=stream0)
        buf329 = buf318; del buf318  # reuse
        buf330 = buf317; del buf317  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf326, buf327, buf328, buf329, buf330, 2592, 4, grid=grid(2592), stream=stream0)
        buf332 = reinterpret_tensor(buf323, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf323  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf325, arg287_1, buf329, buf330, arg87_1, arg88_1, buf332, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg287_1
        del arg87_1
        del arg88_1
        del buf325
        buf333 = reinterpret_tensor(buf322, (2592, 2048), (2048, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (2592, 512), (512, 1), 0), reinterpret_tensor(arg288_1, (512, 2048), (1, 512), 0), out=buf333)
        del arg288_1
        buf334 = reinterpret_tensor(buf333, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf333  # reuse
        # Source Nodes: [x_398], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf334, arg289_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg289_1
        buf335 = reinterpret_tensor(buf332, (2592, 512), (512, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg290_1, (2048, 512), (1, 2048), 0), out=buf335)
        del arg290_1
        buf336 = buf324; del buf324  # reuse
        # Source Nodes: [shortcut_28, x_405], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf336, buf335, arg291_1, arg89_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg291_1
        del arg89_1
        # Source Nodes: [x_407], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, arg292_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf337, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg292_1
        buf338 = buf328; del buf328  # reuse
        buf339 = buf327; del buf327  # reuse
        buf340 = buf326; del buf326  # reuse
        # Source Nodes: [x_410], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf337, arg293_1, buf338, buf339, buf340, 10368, 128, grid=grid(10368), stream=stream0)
        buf341 = buf330; del buf330  # reuse
        buf342 = buf329; del buf329  # reuse
        # Source Nodes: [x_410], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf338, buf339, buf340, buf341, buf342, 2592, 4, grid=grid(2592), stream=stream0)
        buf344 = reinterpret_tensor(buf335, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf335  # reuse
        # Source Nodes: [x_410], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf337, arg293_1, buf341, buf342, arg90_1, arg91_1, buf344, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg293_1
        del arg90_1
        del arg91_1
        del buf337
        buf345 = reinterpret_tensor(buf334, (2592, 2048), (2048, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf344, (2592, 512), (512, 1), 0), reinterpret_tensor(arg294_1, (512, 2048), (1, 512), 0), out=buf345)
        del arg294_1
        buf346 = reinterpret_tensor(buf345, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf345  # reuse
        # Source Nodes: [x_412], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf346, arg295_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg295_1
        buf347 = reinterpret_tensor(buf344, (2592, 512), (512, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg296_1, (2048, 512), (1, 2048), 0), out=buf347)
        del arg296_1
        buf348 = buf336; del buf336  # reuse
        # Source Nodes: [shortcut_29, x_419], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf348, buf347, arg297_1, arg92_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg297_1
        del arg92_1
        # Source Nodes: [x_421], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, arg298_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf349, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg298_1
        buf350 = buf340; del buf340  # reuse
        buf351 = buf339; del buf339  # reuse
        buf352 = buf338; del buf338  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf349, arg299_1, buf350, buf351, buf352, 10368, 128, grid=grid(10368), stream=stream0)
        buf353 = buf342; del buf342  # reuse
        buf354 = buf341; del buf341  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf350, buf351, buf352, buf353, buf354, 2592, 4, grid=grid(2592), stream=stream0)
        buf356 = reinterpret_tensor(buf347, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf347  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf349, arg299_1, buf353, buf354, arg93_1, arg94_1, buf356, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg299_1
        del arg93_1
        del arg94_1
        del buf349
        buf357 = reinterpret_tensor(buf346, (2592, 2048), (2048, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (2592, 512), (512, 1), 0), reinterpret_tensor(arg300_1, (512, 2048), (1, 512), 0), out=buf357)
        del arg300_1
        buf358 = reinterpret_tensor(buf357, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf357  # reuse
        # Source Nodes: [x_426], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf358, arg301_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg301_1
        buf359 = reinterpret_tensor(buf356, (2592, 512), (512, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg302_1, (2048, 512), (1, 2048), 0), out=buf359)
        del arg302_1
        buf360 = buf348; del buf348  # reuse
        # Source Nodes: [shortcut_30, x_433], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf360, buf359, arg303_1, arg95_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg303_1
        del arg95_1
        # Source Nodes: [x_435], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, arg304_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf361, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg304_1
        buf362 = buf352; del buf352  # reuse
        buf363 = buf351; del buf351  # reuse
        buf364 = buf350; del buf350  # reuse
        # Source Nodes: [x_438], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf361, arg305_1, buf362, buf363, buf364, 10368, 128, grid=grid(10368), stream=stream0)
        buf365 = buf354; del buf354  # reuse
        buf366 = buf353; del buf353  # reuse
        # Source Nodes: [x_438], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf362, buf363, buf364, buf365, buf366, 2592, 4, grid=grid(2592), stream=stream0)
        buf368 = reinterpret_tensor(buf359, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf359  # reuse
        # Source Nodes: [x_438], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf361, arg305_1, buf365, buf366, arg96_1, arg97_1, buf368, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg305_1
        del arg96_1
        del arg97_1
        del buf361
        buf369 = reinterpret_tensor(buf358, (2592, 2048), (2048, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf368, (2592, 512), (512, 1), 0), reinterpret_tensor(arg306_1, (512, 2048), (1, 512), 0), out=buf369)
        del arg306_1
        buf370 = reinterpret_tensor(buf369, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf369  # reuse
        # Source Nodes: [x_440], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf370, arg307_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg307_1
        buf371 = reinterpret_tensor(buf368, (2592, 512), (512, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg308_1, (2048, 512), (1, 2048), 0), out=buf371)
        del arg308_1
        buf372 = buf360; del buf360  # reuse
        # Source Nodes: [shortcut_31, x_447], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf372, buf371, arg309_1, arg98_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg309_1
        del arg98_1
        # Source Nodes: [x_449], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, arg310_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf373, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg310_1
        buf374 = buf364; del buf364  # reuse
        buf375 = buf363; del buf363  # reuse
        buf376 = buf362; del buf362  # reuse
        # Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf373, arg311_1, buf374, buf375, buf376, 10368, 128, grid=grid(10368), stream=stream0)
        buf377 = buf366; del buf366  # reuse
        buf378 = buf365; del buf365  # reuse
        # Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf374, buf375, buf376, buf377, buf378, 2592, 4, grid=grid(2592), stream=stream0)
        buf380 = reinterpret_tensor(buf371, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf371  # reuse
        # Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf373, arg311_1, buf377, buf378, arg99_1, arg100_1, buf380, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg100_1
        del arg311_1
        del arg99_1
        del buf373
        buf381 = reinterpret_tensor(buf370, (2592, 2048), (2048, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf380, (2592, 512), (512, 1), 0), reinterpret_tensor(arg312_1, (512, 2048), (1, 512), 0), out=buf381)
        del arg312_1
        buf382 = reinterpret_tensor(buf381, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf381  # reuse
        # Source Nodes: [x_454], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf382, arg313_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg313_1
        buf383 = reinterpret_tensor(buf380, (2592, 512), (512, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg314_1, (2048, 512), (1, 2048), 0), out=buf383)
        del arg314_1
        buf384 = buf372; del buf372  # reuse
        # Source Nodes: [shortcut_32, x_461], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf384, buf383, arg315_1, arg101_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg101_1
        del arg315_1
        # Source Nodes: [x_463], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, arg316_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf385, (8, 512, 18, 18), (165888, 324, 18, 1))
        del arg316_1
        buf386 = buf376; del buf376  # reuse
        buf387 = buf375; del buf375  # reuse
        buf388 = buf374; del buf374  # reuse
        # Source Nodes: [x_466], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf385, arg317_1, buf386, buf387, buf388, 10368, 128, grid=grid(10368), stream=stream0)
        buf389 = buf378; del buf378  # reuse
        buf390 = buf377; del buf377  # reuse
        # Source Nodes: [x_466], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_16.run(buf386, buf387, buf388, buf389, buf390, 2592, 4, grid=grid(2592), stream=stream0)
        del buf386
        del buf387
        del buf388
        buf392 = reinterpret_tensor(buf383, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf383  # reuse
        # Source Nodes: [x_466], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_17.run(buf385, arg317_1, buf389, buf390, arg102_1, arg103_1, buf392, 2592, 512, grid=grid(2592, 512), stream=stream0)
        del arg102_1
        del arg103_1
        del arg317_1
        del buf389
        del buf390
        buf393 = reinterpret_tensor(buf382, (2592, 2048), (2048, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (2592, 512), (512, 1), 0), reinterpret_tensor(arg318_1, (512, 2048), (1, 512), 0), out=buf393)
        del arg318_1
        buf394 = reinterpret_tensor(buf393, (8, 18, 18, 2048), (663552, 36864, 2048, 1), 0); del buf393  # reuse
        # Source Nodes: [x_468], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_18.run(buf394, arg319_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg319_1
        buf395 = reinterpret_tensor(buf392, (2592, 512), (512, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (2592, 2048), (2048, 1), 0), reinterpret_tensor(arg320_1, (2048, 512), (1, 2048), 0), out=buf395)
        del arg320_1
        del buf394
        buf399 = reinterpret_tensor(buf385, (8, 18, 18, 512), (165888, 9216, 512, 1), 0); del buf385  # reuse
        # Source Nodes: [x_479], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_21.run(buf395, arg321_1, arg104_1, buf384, arg105_1, arg106_1, buf399, 2592, 512, grid=grid(2592), stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        del arg321_1
        del buf384
        del buf395
        buf400 = empty_strided((1024, 512, 2, 2), (2048, 1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(arg322_1, buf400, 524288, 4, grid=grid(524288, 4), stream=stream0)
        del arg322_1
        # Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(reinterpret_tensor(buf399, (8, 512, 18, 18), (165888, 1, 9216, 512), 0), buf400, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 1024, 9, 9), (82944, 81, 9, 1))
        del buf399
        del buf400
        buf402 = empty_strided((8, 1024, 9, 9), (82944, 1, 9216, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf401, arg323_1, buf402, 8192, 81, grid=grid(8192, 81), stream=stream0)
        del arg323_1
        # Source Nodes: [x_482], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, arg324_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf403, (8, 1024, 9, 9), (82944, 81, 9, 1))
        del arg324_1
        buf404 = empty_strided((8, 9, 9, 1, 8), (648, 9, 1, 5184, 81), device='cuda', dtype=torch.float32)
        buf405 = empty_strided((8, 9, 9, 1, 8), (648, 9, 1, 5184, 81), device='cuda', dtype=torch.float32)
        buf406 = empty_strided((8, 9, 9, 1, 8), (648, 9, 1, 5184, 81), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf403, arg325_1, buf404, buf405, buf406, 5184, 128, grid=grid(5184), stream=stream0)
        buf407 = empty_strided((8, 9, 9, 1), (81, 9, 1, 648), device='cuda', dtype=torch.float32)
        buf408 = empty_strided((8, 9, 9, 1), (81, 9, 1, 648), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_25.run(buf404, buf405, buf406, buf407, buf408, 648, 8, grid=grid(648), stream=stream0)
        buf410 = reinterpret_tensor(buf401, (8, 9, 9, 1024), (82944, 9216, 1024, 1), 0); del buf401  # reuse
        # Source Nodes: [x_485], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf403, arg325_1, buf407, buf408, arg107_1, arg108_1, buf410, 648, 1024, grid=grid(648, 1024), stream=stream0)
        del arg107_1
        del arg108_1
        del arg325_1
        del buf403
        buf411 = reinterpret_tensor(buf69, (648, 4096), (4096, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (648, 1024), (1024, 1), 0), reinterpret_tensor(arg326_1, (1024, 4096), (1, 1024), 0), out=buf411)
        del arg326_1
        buf412 = reinterpret_tensor(buf411, (8, 9, 9, 4096), (331776, 36864, 4096, 1), 0); del buf411  # reuse
        # Source Nodes: [x_487], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf412, arg327_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg327_1
        buf413 = reinterpret_tensor(buf410, (648, 1024), (1024, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (648, 4096), (4096, 1), 0), reinterpret_tensor(arg328_1, (4096, 1024), (1, 4096), 0), out=buf413)
        del arg328_1
        buf414 = buf402; del buf402  # reuse
        # Source Nodes: [shortcut_34, x_494], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_28.run(buf414, buf413, arg329_1, arg109_1, 663552, grid=grid(663552), stream=stream0)
        del arg109_1
        del arg329_1
        # Source Nodes: [x_496], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf414, arg330_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf415, (8, 1024, 9, 9), (82944, 81, 9, 1))
        del arg330_1
        buf416 = buf406; del buf406  # reuse
        buf417 = buf405; del buf405  # reuse
        buf418 = buf404; del buf404  # reuse
        # Source Nodes: [x_499], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf415, arg331_1, buf416, buf417, buf418, 5184, 128, grid=grid(5184), stream=stream0)
        buf419 = buf408; del buf408  # reuse
        buf420 = buf407; del buf407  # reuse
        # Source Nodes: [x_499], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_25.run(buf416, buf417, buf418, buf419, buf420, 648, 8, grid=grid(648), stream=stream0)
        buf422 = reinterpret_tensor(buf413, (8, 9, 9, 1024), (82944, 9216, 1024, 1), 0); del buf413  # reuse
        # Source Nodes: [x_499], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf415, arg331_1, buf419, buf420, arg110_1, arg111_1, buf422, 648, 1024, grid=grid(648, 1024), stream=stream0)
        del arg110_1
        del arg111_1
        del arg331_1
        del buf415
        buf423 = reinterpret_tensor(buf412, (648, 4096), (4096, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf422, (648, 1024), (1024, 1), 0), reinterpret_tensor(arg332_1, (1024, 4096), (1, 1024), 0), out=buf423)
        del arg332_1
        buf424 = reinterpret_tensor(buf423, (8, 9, 9, 4096), (331776, 36864, 4096, 1), 0); del buf423  # reuse
        # Source Nodes: [x_501], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf424, arg333_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg333_1
        buf425 = reinterpret_tensor(buf422, (648, 1024), (1024, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf424, (648, 4096), (4096, 1), 0), reinterpret_tensor(arg334_1, (4096, 1024), (1, 4096), 0), out=buf425)
        del arg334_1
        buf426 = buf414; del buf414  # reuse
        # Source Nodes: [shortcut_35, x_508], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_28.run(buf426, buf425, arg335_1, arg112_1, 663552, grid=grid(663552), stream=stream0)
        del arg112_1
        del arg335_1
        # Source Nodes: [x_510], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, arg336_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf427, (8, 1024, 9, 9), (82944, 81, 9, 1))
        del arg336_1
        buf428 = buf418; del buf418  # reuse
        buf429 = buf417; del buf417  # reuse
        buf430 = buf416; del buf416  # reuse
        # Source Nodes: [x_513], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_24.run(buf427, arg337_1, buf428, buf429, buf430, 5184, 128, grid=grid(5184), stream=stream0)
        buf431 = buf420; del buf420  # reuse
        buf432 = buf419; del buf419  # reuse
        # Source Nodes: [x_513], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_25.run(buf428, buf429, buf430, buf431, buf432, 648, 8, grid=grid(648), stream=stream0)
        del buf428
        del buf429
        del buf430
        buf434 = reinterpret_tensor(buf425, (8, 9, 9, 1024), (82944, 9216, 1024, 1), 0); del buf425  # reuse
        # Source Nodes: [x_513], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_26.run(buf427, arg337_1, buf431, buf432, arg113_1, arg114_1, buf434, 648, 1024, grid=grid(648, 1024), stream=stream0)
        del arg113_1
        del arg114_1
        del arg337_1
        del buf427
        del buf431
        del buf432
        buf435 = reinterpret_tensor(buf424, (648, 4096), (4096, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf434, (648, 1024), (1024, 1), 0), reinterpret_tensor(arg338_1, (1024, 4096), (1, 1024), 0), out=buf435)
        del arg338_1
        buf436 = reinterpret_tensor(buf435, (8, 9, 9, 4096), (331776, 36864, 4096, 1), 0); del buf435  # reuse
        # Source Nodes: [x_515], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_27.run(buf436, arg339_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg339_1
        buf437 = reinterpret_tensor(buf434, (648, 1024), (1024, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (648, 4096), (4096, 1), 0), reinterpret_tensor(arg340_1, (4096, 1024), (1, 4096), 0), out=buf437)
        del arg340_1
        del buf436
        buf438 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf439 = reinterpret_tensor(buf438, (8, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf438  # reuse
        # Source Nodes: [x_522, x_525, x_528], Original ATen: [aten.add, aten.mean, aten.mul]
        triton_red_fused_add_mean_mul_29.run(buf439, buf437, arg341_1, arg115_1, buf426, 8192, 81, grid=grid(8192), stream=stream0)
        del arg115_1
        del arg341_1
        del buf426
        del buf437
        buf443 = empty((8, 1, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_532], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_30.run(buf439, arg116_1, arg117_1, buf443, 8, 1024, grid=grid(8), stream=stream0)
        del arg116_1
        del arg117_1
        del buf439
        buf444 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_539], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg343_1, reinterpret_tensor(buf443, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg342_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf444)
        del arg342_1
        del arg343_1
        return (buf444, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, 512, 2, 2), (2048, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((8, 3, 288, 288), (248832, 82944, 288, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convnext_base', benchmark_compiled_module)
